"""Person mattes (CPU, ONNX) for the pop-out effects.

Model: MODNet (Apache-2.0, 25 MB), run on the photo and its mirror image and averaged.
BiRefNet-lite (MIT) was tried too: 40-70x slower on CPU for no visible gain on people, and a
bigger input size pulled in tree roots and leaves, so neither is used.

The raw matte is cleaned with plain image processing:
1. keep only regions that contain a detected face (drops passers-by, roots, leaves);
2. fill holes fully inside a person (a knee or a sleeve the model missed);
3. remove thin spikes (a pole behind a head) with an opening, then give back detail only within
   a few pixels of what survived, so hair and fingers keep their outline;
4. a colour guided filter (He et al.) snaps the soft edge to the photo's own edges (hair);
5. foreground colours are estimated near the edge ("blur fusion", Forte & Pitie 2021) so the
   cut-out doesn't carry a fringe of background colour onto a new backdrop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..immich import FaceBox

MODNET_SHORT_EDGE = 512


@dataclass
class Matte:
    alpha: np.ndarray  # H x W float32, 0..1
    solid: np.ndarray  # H x W bool: the cleaned silhouette (no soft edge)
    foreground: np.ndarray  # H x W x 3 float32 0..1: colours with the background removed at edges
    face_width: float  # typical face width in pixels (sets the scale of the clean-up)


class Matter:
    def __init__(self, model_path: Path, threads: int = 0) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def raw(self, img: Image.Image) -> np.ndarray:
        """Model output at the photo's size, averaged with the mirrored pass."""
        w, h = img.size
        s = MODNET_SHORT_EDGE / min(w, h)
        nw, nh = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
        arr = (np.asarray(img.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR), np.float32) / 255.0 - 0.5) / 0.5
        x = arr.transpose(2, 0, 1)
        batch = np.ascontiguousarray(np.stack([x, x[:, :, ::-1]]).astype(np.float32))
        try:
            out = self.session.run(None, {self.input_name: batch})[0]
            a, b = out[0, 0], out[1, 0][:, ::-1]
        except Exception:  # noqa: BLE001 - a model exported for batch size 1
            a = self.session.run(None, {self.input_name: batch[:1]})[0][0, 0]
            b = self.session.run(None, {self.input_name: batch[1:]})[0][0, 0][:, ::-1]
        m = (a + b) / 2
        return np.clip(cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)

    def matte(self, img: Image.Image, faces: Sequence[FaceBox] = ()) -> Matte:
        return clean(self.raw(img), img, faces)


# ---- clean-up ------------------------------------------------------------------------------


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that don't touch the image border."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = np.pad(m, 1, constant_values=0) * 255
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = flood[1:-1, 1:-1] == 0
    return (m.astype(bool) | holes)


def keep_people(mask: np.ndarray, faces: Sequence[FaceBox]) -> np.ndarray:
    """Keep connected regions with a face centre inside (or within half a face height)."""
    if not faces:
        return mask
    h, w = mask.shape
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for f in faces:
        cx, cy = int((f.x1 + f.x2) / 2 * w), int((f.y1 + f.y2) / 2 * h)
        reach = max(3, int((f.y2 - f.y1) * h * 0.5))
        region = labels[max(0, cy - reach) : cy + reach, max(0, cx - reach) : cx + reach]
        keep[np.unique(region)] = True
    keep[0] = False
    return keep[labels]


def guided_filter_rgb(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter with a colour guide (3x3 covariance per pixel)."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)  # noqa: E731
    mI = [mean(I[..., c]) for c in range(3)]
    mp = mean(p)
    cov = np.stack([mean(I[..., c] * p) - mI[c] * mp for c in range(3)], -1)
    var = np.empty(p.shape + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = mean(I[..., i] * I[..., j]) - mI[i] * mI[j]
            var[..., i, j] = var[..., j, i] = v
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov[..., None])[..., 0]
    b = mp - (a * np.stack(mI, -1)).sum(-1)
    return (np.stack([mean(a[..., c]) for c in range(3)], -1) * I).sum(-1) + mean(b)


def estimate_foreground(image: np.ndarray, alpha: np.ndarray, r1: int = 90, r2: int = 6) -> np.ndarray:
    """Approximate fast foreground colour estimation (Forte & Pitie 2021), two passes."""
    a = alpha[..., None]

    def one(F: np.ndarray, B: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
        k = (r, r)
        ba = cv2.blur(alpha, k)[..., None]
        bF = cv2.blur(F * a, k) / (ba + 1e-5)
        bB = cv2.blur(B * (1 - a), k) / ((1 - ba) + 1e-5)
        F = bF + a * (image - a * bF - (1 - a) * bB)
        return np.clip(F, 0, 1), bB

    F, B = one(image, image, r1)
    F, _ = one(F, B, r2)
    return F.astype(np.float32)


def trim_above_heads(mask: np.ndarray, faces: Sequence[FaceBox], width: int, height: int) -> np.ndarray:
    """Remove things standing up out of a head (a pole, a lamp, a branch behind it).

    Above each face, the silhouette may reach at most ~0.55 face heights over the face box
    (hair, a bun, a cap). Higher pixels in that column are dropped unless they belong to another
    person's head region.
    """
    if not faces:
        return mask
    # Per column, the highest allowed row: the top of the tallest head over that column. A child
    # in front of an adult must not cut into the adult's shoulders, so the highest one wins.
    limit_col = np.full(width, -1, np.int32)
    for f in faces:
        fwid, fhei = (f.x2 - f.x1) * width, (f.y2 - f.y1) * height
        cx = (f.x1 + f.x2) / 2 * width
        x1, x2 = int(max(0, cx - fwid * 0.9)), int(min(width, cx + fwid * 0.9))
        limit = int(max(0, f.y1 * height - fhei * 0.55))
        seg = limit_col[x1:x2]
        limit_col[x1:x2] = np.where((seg < 0) | (seg > limit), limit, seg)
    rows = np.arange(height)[:, None]
    cut = (limit_col[None, :] >= 0) & (rows < limit_col[None, :])
    kept = mask & ~cut
    candidate = (mask & cut).astype(np.uint8)
    if not candidate.any():
        return kept
    # Only drop pieces that stand straight up out of a head. A piece that also joins the kept
    # silhouette sideways at body height (below the top of the tallest head) is somebody's
    # shoulder, e.g. an adult standing behind a child, so it stays.
    n, labels = cv2.connectedComponents(candidate, connectivity=8)
    side = np.zeros_like(kept)
    side[:, 1:] |= kept[:, :-1]
    side[:, :-1] |= kept[:, 1:]
    side[: int(limit_col[limit_col >= 0].min())] = False
    joined = np.zeros(n, bool)
    joined[np.unique(labels[(labels > 0) & side])] = True
    joined[0] = False
    return kept | joined[labels]


def typical_face_width(faces: Sequence[FaceBox], width: int, height: int) -> float:
    if not faces:
        return max(width, height) / 12
    ws = sorted((f.x2 - f.x1) * width for f in faces)
    return float(ws[len(ws) // 2])


def clean(raw: np.ndarray, img: Image.Image, faces: Sequence[FaceBox] = ()) -> Matte:
    w, h = img.size
    fw = typical_face_width(faces, w, h)
    solid = keep_people(raw > 0.5, faces)
    solid = fill_holes(solid)
    # Thin spikes: open with ~1/6 of a face width, then restore detail within ~1/4 face width.
    r_open = max(2, int(fw / 6))
    opened = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_OPEN, _disk(r_open)).astype(bool)
    opened = keep_people(opened, faces) if faces else opened
    near = cv2.dilate(opened.astype(np.uint8), _disk(max(2, int(fw / 4)))).astype(bool)
    solid = solid & near
    solid = fill_holes(cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, _disk(max(2, int(fw / 10)))).astype(bool))
    solid = trim_above_heads(solid, faces, w, h)

    # Soft edge: inside a band around the silhouette, let the colour guided filter decide.
    I = np.asarray(img.convert("RGB"), np.float32) / 255.0
    band_r = max(3, int(fw / 8))
    inner = cv2.erode(solid.astype(np.uint8), _disk(band_r)).astype(bool)
    outer = cv2.dilate(solid.astype(np.uint8), _disk(band_r)).astype(bool)
    p = np.where(inner, 1.0, np.where(outer, np.maximum(raw, solid * 0.5), 0.0)).astype(np.float32)
    alpha = guided_filter_rgb(I, p, r=max(3, int(fw / 10)), eps=2e-4)
    alpha = np.where(inner, 1.0, np.where(outer, np.clip(alpha, 0, 1), 0.0)).astype(np.float32)
    fg = estimate_foreground(I, alpha, r1=max(15, int(fw * 1.2)), r2=max(3, int(fw / 12)))
    return Matte(alpha=alpha, solid=solid, foreground=fg, face_width=fw)


def smooth_silhouette(solid: np.ndarray, face_width: float, grow: float) -> np.ndarray:
    """A die-cut shape: concave gaps closed, outline smoothed, grown by `grow` px. Float 0..1."""
    m = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE, _disk(max(3, int(face_width * 0.35))))
    m = cv2.dilate(m, _disk(max(1, int(grow))))
    s = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.5, face_width / 10))
    return np.clip((s - 0.5) * 6 + 0.5, 0, 1).astype(np.float32)
