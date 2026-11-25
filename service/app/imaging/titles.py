"""Animated title cards for highlight videos (CPU, Pillow).

Animation ideas from tz-thantzin/my_animated_text (MIT, Flutter): fade-up, typewriter and a
letter-by-letter stagger. Re-done here for video frames; no code from it is used.

The card: the film's first photo, blurred and darkened, with the title and a small date line.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .frames import _font

TITLE_STYLES = ("fade-up", "typewriter", "letters")
INK = (245, 239, 225)
ACCENT = (125, 211, 252)


def pick_title_style(requested: str, seed: int) -> str:
    return random.Random(seed).choice(TITLE_STYLES) if requested == "random" else requested


def _ease_out(t: float) -> float:
    return 1 - (1 - min(max(t, 0.0), 1.0)) ** 3


def _backdrop(photo: Image.Image, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    img = photo.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.Resampling.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    a = np.asarray(img.crop((x, y, x + w, y + h)), np.float32)
    a = cv2.GaussianBlur(a, (0, 0), max(w, h) / 90)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return a * (0.42 - 0.12 * np.abs(yy - 0.55))  # darker towards top and bottom


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, start: int) -> ImageFont.ImageFont:
    size = start
    while size > 24:
        f = _font(size)
        if draw.textlength(text, font=f) <= max_w:
            return f
        size = int(size * 0.9)
    return _font(size)


def title_frames(photo: Image.Image, title: str, subtitle: str, size: tuple[int, int], seconds: float, fps: int, style: str) -> Iterator[np.ndarray]:
    """Frames of the title card; text animates in during the first ~60 %, then holds."""
    w, h = size
    base = _backdrop(photo, size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    f_title = _fit_font(probe, title, int(w * 0.84), int(min(w, h) * 0.11))
    f_sub = _font(max(18, int(min(w, h) * 0.034)))
    tw = probe.textlength(title, font=f_title)
    sw = probe.textlength(subtitle.upper(), font=f_sub) if subtitle else 0
    ty = int(h * 0.44)
    sy = ty + int(f_title.size * 1.35)
    n = round(seconds * fps)
    for i in range(n):
        t = i / max(1, n - 1)
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        p = _ease_out(t / 0.6)
        if style == "typewriter":
            k = int(round(len(title) * min(1.0, t / 0.55)))
            shown = title[:k]
            d.text(((w - tw) / 2, ty), shown, font=f_title, fill=(*INK, 255))
            if t < 0.75 and (i // max(1, fps // 3)) % 2 == 0:  # blinking caret
                cx = (w - tw) / 2 + probe.textlength(shown, font=f_title) + 6
                d.rectangle((cx, ty + f_title.size * 0.15, cx + max(3, f_title.size // 14), ty + f_title.size * 1.05), fill=(*ACCENT, 255))
        elif style == "letters":
            x = (w - tw) / 2
            for j, ch in enumerate(title):
                lt = _ease_out((t - j * 0.5 / max(1, len(title))) / 0.35)
                d.text((x, ty + (1 - lt) * f_title.size * 0.6), ch, font=f_title, fill=(*INK, int(255 * lt)))
                x += probe.textlength(ch, font=f_title)
        else:  # fade-up
            d.text(((w - tw) / 2, ty + (1 - p) * f_title.size * 0.5), title, font=f_title, fill=(*INK, int(255 * p)))
        if subtitle:
            sp = _ease_out((t - 0.35) / 0.4)
            d.text(((w - sw) / 2, sy + (1 - sp) * 12), subtitle.upper(), font=f_sub, fill=(*ACCENT, int(255 * sp)))
            lw = int(sw * 0.4 * sp)
            d.line(((w - lw) / 2, sy - f_sub.size * 0.6, (w + lw) / 2, sy - f_sub.size * 0.6), fill=(*ACCENT, int(200 * sp)), width=max(2, h // 400))
        z = 1.0 + 0.04 * t  # the backdrop drifts in slightly
        bg = cv2.resize(base, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
        oy, ox = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
        bg = bg[oy : oy + h, ox : ox + w]
        a = np.asarray(layer, np.float32)
        alpha = a[..., 3:] / 255.0
        yield np.clip(bg * (1 - alpha) + a[..., :3] * alpha, 0, 255).astype(np.uint8)
