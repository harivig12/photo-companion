"""ffmpeg helpers (encode frames to H.264 MP4). ffmpeg runs as a separate process."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np


class FfmpegMissing(RuntimeError):
    pass


def find_ffmpeg(configured: str) -> str:
    path = shutil.which(configured) if not Path(configured).is_file() else configured
    if not path:
        raise FfmpegMissing(f"ffmpeg not found ({configured!r}). Install it or set PHOTOS_FFMPEG.")
    return path


def encode_mp4(
    frames: Iterable[np.ndarray],
    size: tuple[int, int],
    fps: int,
    out: Path,
    ffmpeg: str = "ffmpeg",
    crf: int = 20,
    on_frame=None,
) -> Path:
    """Pipe raw RGB frames into libx264 (yuv420p, faststart) and return the output path."""
    w, h = size
    cmd = [
        find_ffmpeg(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i, frame in enumerate(frames):
            proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
            if on_frame:
                on_frame(i)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()[:300]}")
    except BaseException:
        proc.kill()
        raise
    return out
