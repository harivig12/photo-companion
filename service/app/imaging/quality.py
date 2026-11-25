"""Cheap quality signals: sharpness, near-duplicates, screenshots."""

from __future__ import annotations

import re
from statistics import median

import cv2
import numpy as np
from PIL import Image

SCREENSHOT_NAME = re.compile(r"screen\s?shot|screenshot|scrn|screen_recording|capture", re.IGNORECASE)


def sharpness(img: Image.Image, max_side: int = 512) -> float:
    """Variance of the Laplacian on a downscaled greyscale copy. Higher = sharper."""
    g = img.convert("L")
    g.thumbnail((max_side, max_side))
    arr = np.asarray(g, dtype=np.float32)
    return float(cv2.Laplacian(arr, cv2.CV_32F).var())


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash; near-identical photos differ in only a few bits."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def looks_like_screenshot(filename: str, mime: str, camera_make: str | None) -> bool:
    if SCREENSHOT_NAME.search(filename or ""):
        return True
    return mime == "image/png" and not camera_make


def blurry_flags(scores: list[float], absolute: float = 60.0, relative: float = 0.35) -> list[bool]:
    """A photo is blurry when it's soft in absolute terms *and* much softer than the rest of the set."""
    if not scores:
        return []
    mid = median(scores)
    return [s < absolute or (s < relative * mid and s < 3 * absolute) for s in scores]


def duplicate_groups(hashes: list[int], max_distance: int = 8) -> list[int]:
    """Group index for each hash; photos in the same group are near-duplicates."""
    groups = list(range(len(hashes)))
    for i in range(len(hashes)):
        for j in range(i):
            if hamming(hashes[i], hashes[j]) <= max_distance:
                groups[i] = groups[j]
                break
    return groups


def colour_signature(img: Image.Image) -> np.ndarray:
    """Normalised hue/saturation histogram (30x16 bins) for "same scene" checks."""
    small = img.copy()
    small.thumbnail((256, 256))
    hsv = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


# (max seconds apart, max dHash distance, min colour correlation). Closer in time = looser.
# Tuned on a real trip album: re-framed repeats 10-40 s apart scored dHash 22-23 and
# colour 0.64-0.95; different scenes were 20+ minutes apart with colour <= 0.55.
SCENE_TIERS = ((60.0, 28, 0.55), (300.0, 22, 0.80))


def scene_groups(times: list[float], hashes: list[int], signatures: list[np.ndarray]) -> list[int]:
    """Group photos of the same scene: taken close together *and* looking alike (loose dHash
    match or similar colours). Catches re-framed repeats of a pose, which the strict
    duplicate check misses. Input must be in time order."""
    window = max(t[0] for t in SCENE_TIERS)
    groups = list(range(len(times)))
    for i in range(len(times)):
        for j in range(i - 1, -1, -1):
            dt = times[i] - times[j]
            if dt > window:
                break
            ham = hamming(hashes[i], hashes[j])
            col = None
            for max_dt, max_ham, min_col in SCENE_TIERS:
                if dt > max_dt:
                    continue
                if ham <= max_ham:
                    break
                col = cv2.compareHist(signatures[i], signatures[j], cv2.HISTCMP_CORREL) if col is None else col
                if col >= min_col:
                    break
            else:
                continue
            groups[i] = groups[j]
            break
    return groups
