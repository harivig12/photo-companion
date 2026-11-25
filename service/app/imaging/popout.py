"""Pop-out effects built on a person matte. All CPU, Pillow + OpenCV + NumPy.

References:
- blenderskool/color-pop (code used: grey weights 0.3 / 0.59 / 0.11 and the +10 % saturation
  on the subject; see `splash`);
- LionZLH/popoutx, AllsVoid/popout-photo-fx: the subject drawn over the frame so it steps out
  of the picture (`frame`, written here from scratch);
- aeren108/color_splash, HareColor: colour splash; Gauravv97/CameraBlur: background blur.

Styles: splash, spotlight, blur, frame, sticker (+ `splash_frames` for a short reveal video).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .frames import INK, PAPER, _font, _paper, _place, _print
from .matting import Matte, smooth_silhouette

STYLES = ("splash", "spotlight", "blur", "frame", "sticker")
STYLE_LABELS = {
    "splash": "Colour splash", "spotlight": "Spotlight", "blur": "Portrait blur",
    "frame": "Pop out of frame", "sticker": "Sticker",
}


# ---- helpers ----------------------------------------------------------------------------


def _rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), np.float32) / 255.0


def _out(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _over(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return fg * a + bg * (1 - a)


def _to_linear(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0, 1), 2.2)


def _to_srgb(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0, 1), 1 / 2.2)


def _disk_kernel(r: float) -> np.ndarray:
    r = max(1.0, r)
    n = int(math.ceil(r))
    yy, xx = np.mgrid[-n : n + 1, -n : n + 1].astype(np.float32)
    k = np.clip(r + 0.5 - np.sqrt(xx**2 + yy**2), 0, 1)  # anti-aliased disk
    return k / k.sum()


def _lens_blur(lin: np.ndarray, r: float) -> np.ndarray:
    """Disk ("bokeh") blur, done at reduced size for big radii (it's smooth anyway)."""
    if r < 1:
        return lin
    scale = 1.0 if r <= 12 else 12 / r
    if scale < 1:
        small = cv2.resize(lin, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.filter2D(small, -1, _disk_kernel(r * scale), borderType=cv2.BORDER_REFLECT)
        return cv2.resize(b, (lin.shape[1], lin.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.filter2D(lin, -1, _disk_kernel(r), borderType=cv2.BORDER_REFLECT)


def clean_plate(rgb: np.ndarray, m: Matte) -> np.ndarray:
    """The photo with the people removed (filled from around them), for halo-free blurs."""
    h, w = m.alpha.shape
    s = min(1.0, 640 / max(w, h))
    small = cv2.resize((rgb * 255).astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hole = cv2.resize((m.alpha > 0.02).astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    filled = cv2.inpaint(small, hole, 6, cv2.INPAINT_TELEA)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    keep = cv2.GaussianBlur((m.alpha <= 0.02).astype(np.float32), (0, 0), 2)[..., None]
    return rgb * keep + big * (1 - keep)


def _subject_box(m: Matte) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m.solid)
    h, w = m.solid.shape
    if not xs.size:
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---- styles -----------------------------------------------------------------------------


def splash(img: Image.Image, m: Matte) -> Image.Image:
    """People in colour, the rest black and white. From color-pop: grey = 0.3 R + 0.59 G + 0.11 B,
    subject saturation x1.1. Added: soft matte edge and a touch of contrast on the grey."""
    rgb = _rgb(img)
    grey = rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11
    grey = np.clip((grey - 0.5) * 1.08 + 0.48, 0, 1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.1, 0, 1)
    colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return _out(_over(colour, np.repeat(grey[..., None], 3, -1), m.alpha))


def spotlight(img: Image.Image, m: Matte, depth: np.ndarray | None = None) -> Image.Image:
    """A stage light on the people: muted, softly blurred, darker background; a warm pool of light
    at their feet and a soft beam from above; rim light on their outline."""
    rgb = _rgb(img)
    h, w = m.alpha.shape
    fw = m.face_width
    x1, y1, x2, y2 = _subject_box(m)
    cx, top, bottom = (x1 + x2) / 2, y1, y2
    sw = max(x2 - x1, fw * 3)

    # Background: 25 % colour, softly blurred (people removed first, so no halo), darkened.
    plate = clean_plate(rgb, m)
    lin = _lens_blur(_to_linear(plate), max(2.0, fw / 10))
    bg = _to_srgb(lin)
    grey = bg.mean(-1, keepdims=True)
    bg = grey + (bg - grey) * 0.25
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Light: a beam widening from above the people down to an ellipse at their feet.
    t = np.clip((yy - (top - (bottom - top) * 0.6)) / max(1.0, bottom - top) / 1.6, 0, 1)
    half = sw * (0.32 + 0.38 * t)
    beam = np.clip(1 - np.abs(xx - cx) / np.maximum(half, 1), 0, 1) ** 1.6
    pool = np.exp(-(((xx - cx) / (sw * 0.6)) ** 2 + ((yy - bottom) / (fw * 1.2)) ** 2))
    light = np.clip(0.16 + 0.42 * beam + 0.5 * pool, 0, 1)
    vignette = 1 - 0.35 * np.clip(np.sqrt(((xx - cx) / w) ** 2 + ((yy - (top + bottom) / 2) / h) ** 2) * 1.6, 0, 1)
    warm = np.array([1.0, 0.93, 0.80], np.float32)
    bg = bg * (light * vignette)[..., None] * warm

    # People: a little warmer and punchier, plus rim light where their outline faces the beam.
    fg = m.foreground
    lab = cv2.cvtColor(fg, cv2.COLOR_RGB2LAB)
    L = lab[..., 0] / 100.0
    L = np.clip(L + (L - cv2.GaussianBlur(L, (0, 0), fw / 3)) * 0.35, 0, 1)  # local contrast
    lab[..., 0] = L * 100
    fg = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * np.array([1.04, 1.0, 0.95], np.float32), 0, 1)
    a = m.alpha
    edge = np.clip(a - cv2.erode(a, np.ones((3, 3), np.uint8), iterations=max(1, int(fw / 25))), 0, 1)
    up = np.clip(cv2.Sobel(cv2.GaussianBlur(a, (0, 0), 2), cv2.CV_32F, 0, 1) * 4, 0, 1)  # top-facing edges
    rim = cv2.GaussianBlur(edge * (0.45 + 0.55 * up), (0, 0), max(1.0, fw / 40))
    fg = np.clip(fg + rim[..., None] * np.array([1.0, 0.92, 0.75], np.float32) * 0.55, 0, 1)
    glow = cv2.GaussianBlur(a, (0, 0), fw / 2.0) * (1 - a)
    bg = np.clip(bg + glow[..., None] * np.array([0.16, 0.15, 0.13], np.float32), 0, 1)
    return _out(_over(fg, bg, a))


def portrait_blur(img: Image.Image, m: Matte, depth: np.ndarray | None = None, subject_depth: float | None = None, strength: float = 1.0) -> Image.Image:
    """Phone-style portrait mode. Lens (disk) blur in linear light, so lights bloom into round
    bokeh; computed on a plate with the people removed, so nothing smears around them; the amount
    grows smoothly with distance behind the people (depth map), and the ground they stand on
    stays sharp."""
    rgb = _rgb(img)
    h, w = m.alpha.shape
    fw = m.face_width
    rmax = max(8.0, w / 55) * strength
    lin = _to_linear(clean_plate(rgb, m))
    radii = [rmax * k for k in (0.12, 0.3, 0.55, 1.0)]
    stack = [lin] + [_lens_blur(lin, r) for r in radii]
    if depth is not None and subject_depth is not None:
        gap = (subject_depth - depth) / max(subject_depth, 1e-3)  # > 0 behind, < 0 in front
        behind = np.clip(gap * 1.8, 0, 1) ** 0.8
        front = np.clip(-gap * 1.2, 0, 1) * 0.6
        # Everything that isn't the people gets at least a little blur, or a scene where the
        # background sits close behind them (roots, a wall) would barely change.
        behind = np.maximum(np.maximum(behind, front), 0.38)
        # ...except the ground they stand on, which stays in focus near their feet.
        x1, y1, x2, y2 = _subject_box(m)
        yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
        feet = np.exp(-(((yy - y2) / max(fw * 1.2, 1)) ** 2)) * (np.abs(gap) < 0.12)
        behind = behind * (1 - 0.7 * feet)
    else:
        x1, y1, x2, y2 = _subject_box(m)
        yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
        behind = np.clip((y2 - yy) / max(1.0, y2 - y1) * 0.8 + 0.2, 0.2, 1)
    behind = cv2.GaussianBlur(behind.astype(np.float32), (0, 0), max(4.0, fw / 2))
    level = behind * (len(stack) - 1)
    out = np.zeros_like(lin)
    for i in range(len(stack)):
        wgt = np.clip(1 - np.abs(level - i), 0, 1)[..., None]
        out += stack[i] * wgt
    bg = _to_srgb(out)
    return _out(_over(m.foreground, bg, m.alpha))


def _cutout_rgba(m: Matte, alpha: np.ndarray | None = None) -> Image.Image:
    a = m.alpha if alpha is None else alpha
    rgba = np.dstack([np.clip(m.foreground, 0, 1), np.clip(a, 0, 1)])
    return Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA")


def pop_frame(img: Image.Image, m: Matte, faces=(), size: tuple[int, int] | None = None, tilt: float = -4.0, caption: str | None = None) -> Image.Image:
    """A tilted polaroid; the people stay at their real size and their heads rise out over the
    top border. Nothing is enlarged, so arms and hands line up with the photo (no doubles).
    The canvas follows the window: landscape for a row of people, portrait for one or two."""
    h, w = m.alpha.shape
    x1, y1, x2, y2 = _subject_box(m)
    fh = np.median([(f.y2 - f.y1) * h for f in faces]) if faces else (y2 - y1) / 6
    face_tops = sorted(f.y1 * h for f in faces) if faces else [y1]
    # Window top: about 60 % down the typical head, so heads rise out of the frame.
    wtop = int(np.median(face_tops) + fh * 0.6)
    pad_x = int((x2 - x1) * 0.07 + m.face_width * 0.4)
    wl, wr = max(0, x1 - pad_x), min(w, x2 + pad_x)
    wb = min(h, int(y2 + (y2 - wtop) * 0.04))
    if (wr - wl) / max(1, wb - wtop) > 1.6:  # very wide group: keep a pleasant landscape window
        extra = int((wr - wl) / 1.6 - (wb - wtop))
        wb = min(h, wb + extra)
    window = img.convert("RGB").crop((wl, wtop, wr, wb))
    landscape = window.width >= window.height * 1.15
    W, H = size or ((3000, 2400) if landscape else (2400, 3000))
    canvas = _paper((W, H), seed=3).convert("RGBA")
    win_w = int(W * (0.74 if landscape else 0.62))
    s = win_w / window.width
    win_h = int(window.height * s)
    pad = int(win_w * 0.05)
    bottom = int(win_w * 0.20) if caption else pad
    card = _print(window.resize((win_w, win_h), Image.Resampling.LANCZOS), (pad, pad, pad, bottom))

    # The pop layer: everything of the people above the window (and beyond its sides), at the
    # same scale and position as in the photo.
    up = int(max(0, wtop - y1) + fh * 0.3)
    src = (wl, max(0, wtop - up), wr, wb)
    cut = _cutout_rgba(m).crop(src).resize((win_w, int((src[3] - src[1]) * s)), Image.Resampling.LANCZOS)
    inside = Image.new("L", cut.size, 0)
    # Fade the layer out just inside the window edge, so the join is invisible.
    top_in = int((wtop - src[1]) * s)
    grad = np.zeros((cut.height,), np.float32)
    grad[:top_in] = 1
    fade = max(2, int(win_w * 0.006))
    grad[top_in : top_in + fade] = np.linspace(1, 0, min(fade, max(0, cut.height - top_in)))
    inside = Image.fromarray((np.repeat(grad[:, None], cut.width, 1) * 255).astype(np.uint8))
    a = np.asarray(cut.getchannel("A"), np.float32) * (np.asarray(inside, np.float32) / 255)
    cut.putalpha(Image.fromarray(a.astype(np.uint8)))

    margin = int(win_w * 0.25)
    base = Image.new("RGBA", (card.width + 2 * margin, card.height + 2 * margin), (0, 0, 0, 0))
    base.alpha_composite(card, (margin, margin))
    ox, oy = margin + pad, margin + pad - top_in
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sh = Image.new("RGBA", cut.size, (0, 0, 0, 0))
    sh.putalpha(cut.getchannel("A").point(lambda v: int(v * 0.45)))
    shadow.alpha_composite(sh, (ox + int(win_w * 0.012), max(0, oy + int(win_w * 0.014))))
    base.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(win_w * 0.010)))
    base.alpha_composite(cut, (ox, max(0, oy)))
    if caption:
        d = ImageDraw.Draw(base)
        f = _font(int(win_w * 0.065))
        d.text((margin + (card.width - d.textlength(caption, font=f)) / 2, margin + pad + win_h + int(bottom * 0.28)), caption, font=f, fill=INK)
    _place(canvas, base, W / 2, H / 2 + H * 0.02, tilt, shadow=0.0)
    return canvas.convert("RGB")


def sticker(img: Image.Image, m: Matte, size: tuple[int, int] | None = None, backdrop: str = "dots") -> Image.Image:
    """Die-cut sticker: a smooth white outline (gaps between arms and body closed) and a shadow on
    a patterned backdrop. People cut off by the photo's bottom edge sit on the canvas bottom.
    The canvas is square, or 3:2 landscape for a wide group."""
    h, w = m.alpha.shape
    bx1, by1, bx2, by2 = _subject_box(m)
    W, H = size or ((3000, 2000) if (bx2 - bx1) > (by2 - by1) * 1.3 else (2400, 2400))
    fw = m.face_width
    border = max(4, int(fw * 0.22))
    x1, y1, x2, y2 = _subject_box(m)
    pad = border * 3
    box = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
    shape = smooth_silhouette(m.solid, fw, border)
    cut = _cutout_rgba(m).crop(box)
    outline = shape[box[1] : box[3], box[0] : box[2]]
    touches_bottom = y2 >= h - 2
    scale = min(W * 0.84 / cut.width, H * (0.92 if touches_bottom else 0.82) / cut.height)
    size2 = (max(1, int(cut.width * scale)), max(1, int(cut.height * scale)))
    cut = cut.resize(size2, Image.Resampling.LANCZOS)
    o = Image.fromarray((outline * 255).astype(np.uint8)).resize(size2, Image.Resampling.LANCZOS)
    white = Image.new("RGBA", size2, (255, 255, 255, 255))
    white.putalpha(o)
    piece = Image.new("RGBA", size2, (0, 0, 0, 0))
    piece.alpha_composite(white)
    piece.alpha_composite(cut)

    bg = Image.new("RGBA", (W, H), (252, 211, 77, 255) if backdrop != "paper" else (*PAPER, 255))
    d = ImageDraw.Draw(bg)
    step = int(min(W, H) / 16)
    for yy in range(step // 2, H, step):
        for xx in range(step // 2 + (step // 2 if (yy // step) % 2 else 0), W, step):
            d.ellipse((xx - step // 7, yy - step // 7, xx + step // 7, yy + step // 7), fill=(245, 158, 11, 255))
    if touches_bottom:  # grounded: the flat cut edge sits on the canvas bottom, upright
        sh = Image.new("RGBA", size2, (0, 0, 0, 0))
        sh.putalpha(piece.getchannel("A").point(lambda v: int(v * 0.4)))
        bg.alpha_composite(sh.filter(ImageFilter.GaussianBlur(size2[0] * 0.012)), ((W - size2[0]) // 2 + int(W * 0.012), H - size2[1] + int(H * 0.004)))
        bg.alpha_composite(piece, ((W - size2[0]) // 2, H - size2[1]))
    else:
        _place(bg, piece, W / 2, H / 2, -3.0, shadow=0.45, lift=1.5)
    return bg.convert("RGB")


def splash_frames(img: Image.Image, m: Matte, seconds: float, fps: int, size: tuple[int, int]):
    """A short video: the photo in black and white, then the people fade into colour (nearest the
    centre first) while the camera pushes in slowly."""
    w, h = size
    still = np.asarray(splash(img, m), np.float32) / 255.0
    rgb = _rgb(img)
    grey = rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11
    grey = np.repeat(np.clip((grey - 0.5) * 1.08 + 0.48, 0, 1)[..., None], 3, -1)
    H0, W0 = m.alpha.shape
    x1, y1, x2, y2 = _subject_box(m)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(W0, H0)
    scale = max(w / W0, h / H0) * 1.08
    n = round(seconds * fps)
    for i in range(n):
        t = i / max(1, n - 1)
        e = t * t * (3 - 2 * t)
        reveal = np.clip((e - 0.15) / 0.45 - dist * 0.9, 0, 1)[..., None]
        frame = grey * (1 - reveal) + still * reveal
        z = scale * (1.0 + 0.06 * e)
        cw, ch = w / z, h / z
        x0 = float(np.clip(cx - cw / 2, 0, W0 - cw))
        y0 = float(np.clip(cy - ch / 2, 0, H0 - ch))
        M = np.array([[z, 0, -x0 * z], [0, z, -y0 * z]], np.float32)
        yield (cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * 255).astype(np.uint8)
