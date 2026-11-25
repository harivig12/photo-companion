"""Draw a planned collage: face-aware crops into slots, uniform gutters, rounded corners."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageColor, ImageDraw

from ..immich import FaceBox
from .crop import crop_box
from .layout import Layout


def canvas_size(ratio: float, long_side: int) -> tuple[int, int]:
    if ratio >= 1:
        return long_side, round(long_side / ratio)
    return round(long_side * ratio), long_side


def slot_pixels(layout: Layout, size: tuple[int, int], gutter: int) -> list[tuple[int, int, int, int]]:
    """Slot rectangles in pixels (left, top, right, bottom) with a `gutter` between and around cells."""
    W, H = size
    unit = W  # layout units: canvas is 1 wide
    inner_w, inner_h = W - gutter, H - gutter  # outer margin of gutter/2, plus gutter/2 per cell side
    sx, sy = inner_w / unit, inner_h / (unit * (H / W))
    out = []
    for s in layout.slots:
        x1 = gutter / 2 + s.x * unit * sx + gutter / 2
        y1 = gutter / 2 + s.y * unit * sy + gutter / 2
        x2 = gutter / 2 + (s.x + s.w) * unit * sx - gutter / 2
        y2 = gutter / 2 + (s.y + s.h) * unit * sy - gutter / 2
        out.append((round(x1), round(y1), round(x2), round(y2)))
    return out


def render(
    images: Sequence[Image.Image],
    faces: Sequence[Sequence[FaceBox]],
    layout: Layout,
    ratio: float,
    long_side: int = 3000,
    gutter: int = 8,
    radius: int = 12,
    colour: str = "#FFFFFF",
    face_aware: bool = True,
) -> Image.Image:
    size = canvas_size(ratio, long_side)
    scale = long_side / 1000  # gutter and radius settings are per 1000 px of the long side
    g, r = round(gutter * scale), round(radius * scale)
    canvas = Image.new("RGB", size, ImageColor.getrgb(colour))
    for img, f, (x1, y1, x2, y2) in zip(images, faces, slot_pixels(layout, size, g)):
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        box = crop_box(img.width, img.height, w / h, f if face_aware else ())
        cell = img.crop(box).resize((w, h), Image.Resampling.LANCZOS)
        if r > 0:
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=min(r, w // 2, h // 2), fill=255)
            canvas.paste(cell, (x1, y1), mask)
        else:
            canvas.paste(cell, (x1, y1))
    return canvas
