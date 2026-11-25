"""Face-aware crop to a target aspect ratio."""

from __future__ import annotations

from collections.abc import Sequence

from ..immich import FaceBox


def crop_box(
    width: int,
    height: int,
    target_aspect: float,
    faces: Sequence[FaceBox] = (),
    margin: float = 0.25,
) -> tuple[int, int, int, int]:
    """Largest box with `target_aspect` (w/h) inside the image, placed to keep faces in view.

    - With faces: centre on the union of all faces (grown by `margin`); if that union doesn't
      fit, centre on the largest face instead.
    - Without faces: centre horizontally, slightly above the middle vertically (people and
      horizons tend to sit in the upper half).
    Returns (left, top, right, bottom) in pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError("empty image")
    img_aspect = width / height
    if img_aspect > target_aspect:
        cw, ch = height * target_aspect, float(height)
    else:
        cw, ch = float(width), width / target_aspect

    if faces:
        ux1 = min(f.x1 for f in faces) * width
        uy1 = min(f.y1 for f in faces) * height
        ux2 = max(f.x2 for f in faces) * width
        uy2 = max(f.y2 for f in faces) * height
        gw, gh = (ux2 - ux1) * margin, (uy2 - uy1) * margin
        ux1, uy1, ux2, uy2 = ux1 - gw, uy1 - gh, ux2 + gw, uy2 + gh
        if ux2 - ux1 > cw or uy2 - uy1 > ch:
            big = max(faces, key=lambda f: f.area)
            ux1, uy1, ux2, uy2 = big.x1 * width, big.y1 * height, big.x2 * width, big.y2 * height
        cx, cy = (ux1 + ux2) / 2, (uy1 + uy2) / 2
    else:
        cx, cy = width / 2, height * 0.45

    left = min(max(cx - cw / 2, 0), width - cw)
    top = min(max(cy - ch / 2, 0), height - ch)
    return round(left), round(top), round(left + cw), round(top + ch)
