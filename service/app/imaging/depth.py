"""Monocular depth with Depth Anything V2 Small (ONNX, CPU) plus the edge cleanup Google describes.

Output convention: a float32 map in 0..1 where **1 = nearest**. The model predicts relative
inverse depth with unknown scale and shift, so it's normalised per photo (2nd-98th percentile).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..immich import FaceBox

INPUT_SIZE = 518
MULTIPLE = 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class DepthResult:
    depth: np.ndarray  # H x W float32, 0..1, 1 = near (same size as the input image)
    spread: float  # how much depth variation the scene has (0 = flat)
    subject: float  # depth value of the main subject (faces, else the nearest large region)


class DepthModel:
    def __init__(self, model_path: Path, threads: int = 0) -> None:
        import onnxruntime as ort  # imported here so the rest of the app works without it

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def raw(self, img: Image.Image) -> np.ndarray:
        """Model output (relative inverse depth) at the model's resolution."""
        w, h = img.size
        scale = INPUT_SIZE / min(w, h)
        nw = max(MULTIPLE, round(w * scale / MULTIPLE) * MULTIPLE)
        nh = max(MULTIPLE, round(h * scale / MULTIPLE) * MULTIPLE)
        arr = np.asarray(img.convert("RGB").resize((nw, nh), Image.Resampling.BICUBIC), np.float32) / 255.0
        arr = (arr - MEAN) / STD
        out = self.session.run(None, {self.input_name: arr.transpose(2, 0, 1)[None]})[0]
        return np.asarray(out, np.float32).reshape(nh, nw)


def _body_mask(shape: tuple[int, int], faces: Sequence[FaceBox]) -> np.ndarray:
    """Rough person regions: each face box widened and extended down to where a body would be."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for f in faces:
        fw, fh = (f.x2 - f.x1) * w, (f.y2 - f.y1) * h
        cx = (f.x1 + f.x2) / 2 * w
        x1, x2 = int(cx - fw * 1.6), int(cx + fw * 1.6)
        y1, y2 = int(f.y1 * h - fh * 0.3), int(f.y2 * h + fh * 5)
        cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), 1, -1)
    return mask.astype(bool)


def estimate(model: DepthModel, img: Image.Image, faces: Sequence[FaceBox] = ()) -> DepthResult:
    raw = model.raw(img)
    lo, hi = np.percentile(raw, [2, 98])
    p10, p90 = np.percentile(raw, [10, 90])
    spread = float((p90 - p10) / (abs(p90) + 1e-6))
    d = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    # Edge cleanup: a median filter removes speckle at depth edges.
    d = cv2.medianBlur((d * 255).astype(np.uint8), 5).astype(np.float32) / 255.0

    # People are often placed too far back at their edges; pull their regions forward.
    if faces:
        body = _body_mask(d.shape, faces)
        face_only = np.zeros_like(body)
        for f in faces:
            face_only[int(f.y1 * d.shape[0]) : int(f.y2 * d.shape[0]) + 1, int(f.x1 * d.shape[1]) : int(f.x2 * d.shape[1]) + 1] = True
        subject = float(np.median(d[face_only])) if face_only.any() else float(np.percentile(d, 80))
        # Only pixels already near the person's depth are pulled forward; genuine background
        # inside the box (trees between people) stays put. Feathered so no box edges show.
        near_subject = (d >= subject - 0.15) & body
        feather = cv2.GaussianBlur(near_subject.astype(np.float32), (0, 0), sigmaX=max(2.0, d.shape[1] / 150))
        d = d + feather * np.clip(subject * 0.95 - d, 0, None)
    else:
        subject = float(np.percentile(d, 85))

    # Back to the photo's size.
    d = cv2.resize(d, img.size, interpolation=cv2.INTER_CUBIC)
    # Grow near layers by a few pixels (grey dilation) so the outline of a face or shoulder
    # moves *with* the person. Without it the outermost pixels stay with the background and
    # are smeared into lines when the camera moves.
    r = max(2, min(img.size) // 250)
    d = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    k = max(3, (min(img.size) // 200) | 1)
    d = cv2.GaussianBlur(d, (k, k), 0)
    return DepthResult(np.clip(d, 0, 1).astype(np.float32), spread, subject)


def depth_png(depth: np.ndarray) -> Image.Image:
    """16-bit greyscale PNG (white = near)."""
    return Image.fromarray((depth * 65535).astype(np.uint16))
