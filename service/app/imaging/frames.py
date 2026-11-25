"""Framed collage styles: photos as tilted prints with soft shadows.

Three styles (references: design-library tabnav/osmo "scattered tilted cards",
tabnav/gamma "tilted card row", patterns/card-art-fan-marquee.html "fan"):

- `scatter`: polaroid prints scattered on warm paper, tilted up to ~9°, the best photo
  larger and on top. Lively.
- `fan`: 5-7 portrait cards fanned from a pivot below the canvas on a dark gallery
  background, with a title. Classy.
- `prints`: the regular least-crop layout, but every cell is a bordered print with a small
  tilt (±2.5°) and a shadow on paper. Tidy but alive; works for any count.

All randomness is seeded, so the same job always renders the same picture.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..immich import FaceBox
from .crop import crop_box
from .layout import plan

PAPER = (245, 239, 225)  # design-library "champagne" --bg
INK = (28, 25, 23)
OBSIDIAN = (7, 8, 12)
SKY = (125, 211, 252)
LIGHT_INK = (232, 236, 244)


@dataclass
class Photo:
    image: Image.Image
    faces: Sequence[FaceBox] = ()
    score: float = 1.0


# ---- building blocks ---------------------------------------------------------------------


def _fit(photo: Photo, w: int, h: int) -> Image.Image:
    box = crop_box(photo.image.width, photo.image.height, w / h, photo.faces)
    return photo.image.crop(box).resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)


def _paper(size: tuple[int, int], colour=PAPER, grain: float = 6.0, seed: int = 0) -> Image.Image:
    """Flat colour with fine grain and a gentle vignette, so it reads as paper, not a fill."""
    w, h = size
    rng = np.random.default_rng(seed)
    base = np.empty((h, w, 3), np.float32)
    base[:] = colour
    base += rng.normal(0, grain, (h, w, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    base *= (1 - 0.07 * np.clip(d - 0.4, 0, 1))[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def _gallery(size: tuple[int, int]) -> Image.Image:
    """Obsidian with a soft cyan glow behind the cards."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h * 0.62) / h) ** 2)
    glow = np.clip(1 - d * 1.8, 0, 1) ** 2
    base = np.empty((h, w, 3), np.float32)
    base[:] = OBSIDIAN
    base += glow[..., None] * np.array([18, 40, 60], np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def _print(photo_img: Image.Image, border: tuple[int, int, int, int], colour=(255, 255, 255), radius: int = 0, rim: bool = False) -> Image.Image:
    """Mount an image on a card. border = (left, top, right, bottom) in px. Returns RGBA."""
    l, t, r, b = border
    w, h = photo_img.width + l + r, photo_img.height + t + b
    card = Image.new("RGBA", (w, h), (*colour, 255))
    card.paste(photo_img, (l, t))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    card.putalpha(mask)
    if rim:
        ImageDraw.Draw(card).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, outline=(255, 255, 255, 60), width=max(2, w // 300))
    return card


def _place(
    canvas: Image.Image, card: Image.Image, cx: float, cy: float, angle: float,
    shadow: float = 0.35, lift: float = 1.0, keep_inside: float | None = None,
) -> None:
    """Rotate a card, drop a soft shadow and paste it centred on (cx, cy).

    keep_inside: margin in px; when set, the card is nudged so it stays fully on the canvas.
    """
    rot = card.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    if keep_inside is not None:
        m = keep_inside
        cx = min(max(cx, m + rot.width / 2), canvas.width - m - rot.width / 2)
        cy = min(max(cy, m + rot.height / 2), canvas.height - m - rot.height / 2)
    blur = max(4, int(min(card.size) * 0.035 * lift))
    off = int(min(card.size) * 0.025 * lift)
    sh = Image.new("RGBA", (rot.width + blur * 4, rot.height + blur * 4), (0, 0, 0, 0))
    alpha = rot.getchannel("A").point(lambda a: int(a * shadow))
    sh.paste((0, 0, 0, 255), (blur * 2, blur * 2), alpha)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    x, y = int(cx - rot.width / 2), int(cy - rot.height / 2)
    canvas.alpha_composite(sh, (x - blur * 2 + off // 2, y - blur * 2 + off))
    canvas.alpha_composite(rot, (x, y))


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSerif.ttf", "georgia.ttf", "Georgia.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# ---- styles ------------------------------------------------------------------------------


def scatter(photos: Sequence[Photo], size: tuple[int, int], seed: int = 1, caption: str | None = None) -> Image.Image:
    """Polaroids on paper. Up to 9 photos; the best one is bigger and on top."""
    rng = random.Random(seed)
    W, H = size
    canvas = _paper(size, seed=seed).convert("RGBA")
    n = len(photos)
    hero = max(range(n), key=lambda i: photos[i].score)
    others = [i for i in range(n) if i != hero]
    cols = math.ceil(math.sqrt(len(others) + 1))
    rows = math.ceil((len(others) + 1) / cols)
    margin = int(min(W, H) * 0.03)
    cell = min((W - 2 * margin) / cols, (H - 2 * margin) / rows)
    side = int(cell * 0.74)  # photo area of a small print (square, like a Polaroid)
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    centre = (cols // 2, rows // 2)
    cells.remove(centre) if centre in cells else None
    rng.shuffle(cells)
    gx, gy = (W - cols * cell) / 2, (H - rows * cell) / 2

    def polaroid(i: int, s: int) -> Image.Image:
        pad = int(s * 0.06)
        return _print(_fit(photos[i], s, s), (pad, pad, pad, int(s * 0.22)))

    for k, i in enumerate(others):
        c, r = cells[k % len(cells)]
        cx = gx + (c + 0.5) * cell + rng.uniform(-0.12, 0.12) * cell
        cy = gy + (r + 0.5) * cell + rng.uniform(-0.10, 0.10) * cell
        angle = rng.uniform(3, 9) * (1 if k % 2 else -1)
        _place(canvas, polaroid(i, side), cx, cy, angle, keep_inside=margin)

    hs = int(side * 1.4)
    hero_card = polaroid(hero, hs)
    if caption:
        d = ImageDraw.Draw(hero_card)
        f = _font(int(hs * 0.075))
        tw = d.textlength(caption, font=f)
        d.text(((hero_card.width - tw) / 2, hs + int(hs * 0.06) + int(hs * 0.05)), caption, font=f, fill=INK)
    _place(canvas, hero_card, W / 2, H / 2, rng.uniform(-3, 3), shadow=0.45, lift=1.6, keep_inside=margin)
    return canvas.convert("RGB")


def fan(photos: Sequence[Photo], size: tuple[int, int], title: str | None = None, subtitle: str | None = None) -> Image.Image:
    """Portrait cards fanned from a pivot far below the canvas (like card-art-fan-marquee).

    The spread angle is chosen so the outer cards stay on the canvas for 3-7 photos.
    """
    W, H = size
    canvas = _gallery(size).convert("RGBA")
    k = len(photos)
    cw = int(W * (0.30 if k <= 5 else 0.25))
    ch = int(cw * 1.3)
    radius = int(cw * 0.06)
    pivot = (W / 2, H * 1.70)
    reach = pivot[1] - H * 0.58
    # Outer card centre at most 0.30 W from the middle.
    step_deg = 9.0 if k < 2 else min(9.0, 2 * math.degrees(math.asin(min(1.0, 0.30 * W / reach))) / (k - 1))
    order = sorted(range(k), key=lambda i: -abs(i - (k - 1) / 2))  # outer cards first, centre on top
    for i in order:
        a = math.radians((i - (k - 1) / 2) * step_deg)
        cx, cy = pivot[0] + reach * math.sin(a), pivot[1] - reach * math.cos(a)
        pad = int(cw * 0.025)
        card = _print(_fit(photos[i], cw - 2 * pad, ch - 2 * pad), (pad,) * 4, colour=(24, 28, 39), radius=radius, rim=True)
        _place(canvas, card, cx, cy, -math.degrees(a), shadow=0.7, lift=1.4)
    d = ImageDraw.Draw(canvas)
    if subtitle:
        f2 = _font(int(W * 0.022))
        sub = subtitle.upper()
        d.text(((W - d.textlength(sub, font=f2)) / 2, H * 0.12), sub, font=f2, fill=SKY)
    if title:
        f = _font(int(W * 0.06))
        d.text(((W - d.textlength(title, font=f)) / 2, H * 0.155), title, font=f, fill=LIGHT_INK)
    return canvas.convert("RGB")


def prints(photos: Sequence[Photo], size: tuple[int, int], ratio: float, seed: int = 1, max_tilt: float = 2.5) -> Image.Image:
    """The regular least-crop layout, each cell a slightly tilted bordered print."""
    rng = random.Random(seed)
    W, H = size
    canvas = _paper(size, seed=seed).convert("RGBA")
    layout = plan([p.image.width / p.image.height for p in photos], [p.score for p in photos], ratio)
    margin = int(min(W, H) * 0.045)
    gap = int(min(W, H) * 0.03)
    iw, ih = W - 2 * margin, H - 2 * margin
    for p, s in zip(photos, layout.slots):
        x1, y1 = margin + s.x * iw, margin + s.y * ratio * ih
        x2, y2 = margin + (s.x + s.w) * iw, margin + (s.y + s.h) * ratio * ih
        w, h = int(x2 - x1 - gap), int(y2 - y1 - gap)
        pad = max(6, int(min(w, h) * 0.035))
        card = _print(_fit(p, w - 2 * pad, h - 2 * pad), (pad,) * 4)
        _place(canvas, card, (x1 + x2) / 2, (y1 + y2) / 2, rng.uniform(-max_tilt, max_tilt), shadow=0.3, lift=0.8)
    return canvas.convert("RGB")


STYLES = ("grid", "prints", "scatter", "fan")
