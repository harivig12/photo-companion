"""Image helpers shared by the pipelines (CPU only: Pillow, NumPy, OpenCV)."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

try:  # HEIC/HEIF originals from iPhones
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - optional
    pass

Image.MAX_IMAGE_PIXELS = 200_000_000  # large panoramas are fine; decompression bombs are not


def decode(data: bytes) -> Image.Image:
    """Decode image bytes, apply the EXIF orientation and return an RGB image."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def encode_jpeg(img: Image.Image, quality: int = 90, exif: Image.Exif | None = None) -> bytes:
    buf = io.BytesIO()
    kwargs = {"quality": quality, "optimize": True, "progressive": True, "subsampling": "4:2:0"}
    if exif is not None:
        kwargs["exif"] = exif
    img.save(buf, "JPEG", **kwargs)
    return buf.getvalue()
