"""2.5D parallax ("cinematic photo") renderer with OpenCV.

Each output pixel looks up the source image through a virtual camera. Points nearer than the
subject move more than points behind it, which gives parallax. Moving the camera exposes
"stretch" where depth jumps: neighbouring output pixels then sample source points far apart.
Following Google's approach, every preset is scored by how much stretch is visible (connected
clusters count more, stretch on people counts triple) and "auto" keeps the lowest score.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

PRESETS = ("zoom", "dolly", "circle")
FIXED_POINT_PASSES = 3


@dataclass(frozen=True)
class Motion:
    preset: str
    amplitude: float  # parallax strength as a fraction of the output width
    zoom: float  # base zoom; > 1 keeps the borders covered while the camera moves


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)  # smoothstep: starts and ends gently


def camera(preset: str, t: float, amp: float) -> tuple[float, float, float]:
    """(shift_x, shift_y, push) at time t in 0..1. Shifts are fractions of width; push scales near points."""
    e = _ease(t)
    if preset == "dolly":
        return amp * (2 * e - 1), 0.0, 0.0
    if preset == "circle":
        a = 2 * math.pi * t
        return amp * math.cos(a), amp * 0.6 * math.sin(a), 0.0
    if preset == "zoom":
        return 0.0, -amp * 0.35 * e, amp * 2.4 * e
    raise ValueError(f"unknown preset {preset}")


def _maps(depth: np.ndarray, out_size: tuple[int, int], focus: float, motion: Motion, t: float, centre: tuple[float, float]):
    """Remap grids (map_x, map_y) into the source for one frame."""
    H, W = depth.shape
    ow, oh = out_size
    sx, sy, push = camera(motion.preset, t, motion.amplitude)
    # Source window that the output covers at base zoom, centred on `centre` (a fraction of the source).
    scale = min(W / ow, H / oh) / motion.zoom
    zoom_t = 1.0 + 0.12 * _ease(t) if motion.preset == "zoom" else 1.0
    u = (np.arange(ow, dtype=np.float32) - ow / 2) * scale / zoom_t
    v = (np.arange(oh, dtype=np.float32) - oh / 2) * scale / zoom_t
    cx = min(max(centre[0] * W, ow * scale / 2), W - ow * scale / 2)
    cy = min(max(centre[1] * H, oh * scale / 2), H - oh * scale / 2)
    bx, by = np.meshgrid(u + cx, v + cy)
    px = W * sx  # shift in source pixels for a point one depth unit from the focus plane
    mx, my = bx, by
    # Fixed-point refinement: look depth up where the pixel actually comes from, not where it
    # lands. One pass (depth at the landing spot) ghosts people's edges: background pixels
    # next to a person sample the person again. Three passes settle on the right layer.
    for _ in range(FIXED_POINT_PASSES):
        d = cv2.remap(depth, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) - focus
        mx = bx - d * px - (bx - cx) * d * push
        my = by - d * W * sy - (by - cy) * d * push
    return mx.astype(np.float32), my.astype(np.float32)


def stretch_score(depth: np.ndarray, out_size: tuple[int, int], focus: float, motion: Motion, subject_mask: np.ndarray | None, centre=(0.5, 0.5)) -> float:
    """How visible stretching is at the extremes of the move (lower is better)."""
    score = 0.0
    small = (max(64, out_size[0] // 4), max(64, out_size[1] // 4))
    d_small = cv2.resize(depth, (depth.shape[1] // 4 or 1, depth.shape[0] // 4 or 1), interpolation=cv2.INTER_AREA)
    for t in (0.0, 0.5, 1.0):
        mx, my = _maps(d_small, small, focus, motion, t, centre)
        gx = np.abs(np.diff(mx, axis=1, append=mx[:, -1:]))
        gy = np.abs(np.diff(my, axis=0, append=my[-1:, :]))
        base = min(d_small.shape[1] / small[0], d_small.shape[0] / small[1]) / motion.zoom
        stretched = ((gx > base * 2.2) | (gy > base * 2.2)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(stretched, connectivity=8)
        weights = np.ones(small[::-1], np.float32)
        if subject_mask is not None:
            weights += 2 * cv2.resize(subject_mask.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            w = float(weights[labels == k].mean())
            score += w * area**1.5  # big connected tears are worse than scattered pixels
    return score / (small[0] * small[1])


STRETCH_OK = 0.10  # below this, stretch isn't noticeable at normal viewing size
MIN_AMP = 0.03  # weaker than this and the parallax isn't worth a video
# History: 0.04 (too subtle) -> 0.065 ("move more", but streaks at face edges) -> 0.05.
BASE_AMP = 0.05


def choose_motion(depth: np.ndarray, out_size: tuple[int, int], focus: float, preset: str, subject_mask, base_amp: float = BASE_AMP) -> tuple[Motion, dict[str, float]]:
    """Pick the preset (or use the given one) and back off the amplitude if stretch is too visible.

    Scores are compared per unit of motion, so a preset can't win just by barely moving.
    """
    zoom = 1.0 + 2.2 * base_amp + 0.04
    candidates = PRESETS if preset == "auto" else (preset,)
    scores: dict[str, float] = {}
    best = None
    for p in candidates:
        amp = base_amp
        while True:
            m = Motion(p, amp, zoom)
            s = stretch_score(depth, out_size, focus, m, subject_mask)
            if s < STRETCH_OK or amp * 0.8 < MIN_AMP:
                break
            amp *= 0.8
        per_motion = s / amp
        scores[p] = round(float(per_motion), 3)
        if best is None or per_motion < best[1]:
            best = (m, per_motion)
    return best[0], scores


def frames(image: np.ndarray, depth: np.ndarray, out_size: tuple[int, int], focus: float, motion: Motion, n_frames: int, centre=(0.5, 0.5)) -> Iterator[np.ndarray]:
    """RGB uint8 frames (out_h x out_w x 3), rendered by forward splatting.

    Every source pixel is moved to where the camera sees it, far pixels first and near ones
    last, so a person always covers what is behind them (no doubled faces). Gaps opened
    behind people ("disocclusions") are filled from their surroundings with OpenCV's inpaint.
    """
    H, W = depth.shape
    ow, oh = out_size
    order = np.argsort(depth, axis=None, kind="stable")  # far -> near
    ys, xs = np.unravel_index(order, depth.shape)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    dd = depth.reshape(-1)[order] - focus
    colours = image.reshape(-1, 3)[order]
    base_scale = min(W / ow, H / oh) / motion.zoom
    cx0 = min(max(centre[0] * W, ow * base_scale / 2), W - ow * base_scale / 2)
    cy0 = min(max(centre[1] * H, oh * base_scale / 2), H - oh * base_scale / 2)
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        sx, sy, push = camera(motion.preset, t, motion.amplitude)
        zoom_t = 1.0 + 0.12 * _ease(t) if motion.preset == "zoom" else 1.0
        scale = base_scale / zoom_t
        # Inverse of the backward model in _maps: source (x, y) -> base view coordinates.
        k = 1.0 - dd * push
        bx = (xs + dd * W * sx - cx0 * dd * push) / k
        by = (ys + dd * W * sy - cy0 * dd * push) / k
        u = np.rint((bx - cx0) / scale + ow / 2).astype(np.int32)
        v = np.rint((by - cy0) / scale + oh / 2).astype(np.int32)
        ok = (u >= 0) & (u < ow) & (v >= 0) & (v < oh)
        out = np.zeros((oh, ow, 3), np.uint8)
        hit = np.zeros((oh, ow), np.uint8)
        zbuf = np.full((oh, ow), -1.0, np.float32)
        # Repeated targets: numpy keeps the last write, i.e. the nearest pixel.
        out[v[ok], u[ok]] = colours[ok]
        hit[v[ok], u[ok]] = 1
        zbuf[v[ok], u[ok]] = dd[ok]
        holes = (1 - hit).astype(np.uint8)
        if holes.any():
            # Single-pixel cracks from the rounding: average only the *real* neighbours
            # (normalised box filter), so empty black pixels never leak in as dark specks.
            hf = hit.astype(np.float32)
            num = cv2.boxFilter(out.astype(np.float32) * hf[..., None], -1, (3, 3), normalize=False)
            den = cv2.boxFilter(hf, -1, (3, 3), normalize=False)
            crack = (holes == 1) & (den >= 3)  # enough real pixels around to trust the average
            out[crack] = (num[crack] / den[crack][:, None]).astype(np.uint8)
            zbuf[crack] = (cv2.boxFilter(zbuf * hf, -1, (3, 3), normalize=False)[crack] / den[crack])
            hit[crack] = 1
            holes[crack] = 0
            if holes.any():
                # Gaps open *behind* a person, so fill them from the background only: hide
                # the near pixels around each gap from the inpainter as well. Plain inpainting
                # drags the person's colours into the gap, which shows as lines at face edges.
                band = cv2.dilate(holes, np.ones((9, 9), np.uint8)).astype(bool) & (hit == 1)
                if band.any():
                    cut = np.percentile(zbuf[band], 50)
                    near = band & (zbuf > cut)
                else:
                    near = np.zeros_like(band)
                mask = ((holes == 1) | near).astype(np.uint8)
                filled = cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA)
                out[holes == 1] = filled[holes == 1]
        yield out


def output_size(width: int, height: int, resolution: str) -> tuple[int, int]:
    """1080p/720p output in the photo's orientation (even sizes for H.264)."""
    long_side = 1920 if resolution == "1080p" else 1280
    short = 1080 if resolution == "1080p" else 720
    if abs(width / height - 1) < 0.08:
        return short, short
    return (long_side, short) if width > height else (short, long_side)
