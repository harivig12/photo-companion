"""Pop-out photo pipeline: one photo with people, rendered in the selected styles.

Input: one asset ID, or an album / event span (the best photo of 1-6 people is picked).
Output: one JPEG per style (splash, spotlight, blur, frame, sticker) and, if enabled, a short
"colour splash" video. Each file is uploaded / saved as its own creation, stacked with the source.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .. import models
from ..immich import AssetInfo, ImmichError
from ..imaging import decode, encode_jpeg
from ..imaging import popout as fx
from ..imaging.depth import DepthModel, estimate
from ..imaging.matting import Matter
from ..imaging.quality import looks_like_screenshot, sharpness
from ..media import encode_mp4
from ..settings_store import POPOUT_STYLES
from . import JobInputError, OutputFile, PipelineContext, PipelineResult

log = logging.getLogger(__name__)
WORK_LONG_SIDE = 2400  # matte and effects are computed at this size
MAX_PEOPLE = 6


@lru_cache(maxsize=1)
def _matter(path: str) -> Matter:
    return Matter(Path(path))


@lru_cache(maxsize=1)
def _depth(path: str) -> DepthModel:
    return DepthModel(Path(path))


async def _pick(ctx: PipelineContext) -> AssetInfo:
    p = ctx.job.params
    if p.get("asset_ids"):
        if len(p["asset_ids"]) != 1:
            raise JobInputError("A pop-out photo uses exactly one photo.")
        return await ctx.gateway.asset(p["asset_ids"][0])
    if p.get("album_id"):
        pool = await ctx.gateway.album_assets(p["album_id"])
    elif p.get("event_from"):
        pool = await ctx.gateway.search(
            taken_after=datetime.fromisoformat(p["event_from"]) - timedelta(seconds=1),
            taken_before=datetime.fromisoformat(p["event_to"]) + timedelta(seconds=1),
        )
    else:
        raise JobInputError("Pick a photo, an album or an event.")
    cands = [
        a for a in pool
        if a.type == "IMAGE" and a.usable and a.people and len(a.people) <= MAX_PEOPLE
        and min(a.width or 0, a.height or 0) >= 1000 and not looks_like_screenshot(a.filename, a.mime, a.make)
    ]
    if not cands:
        raise JobInputError("No photo with 1-6 named people found there.")
    # Score a time-spread sample on sharpness; favourites and 2-4 people win close calls.
    step = max(1, len(cands) // 24)
    best, best_score = None, -1.0
    for a in cands[::step][:24]:
        thumb = await ctx.cpu(decode, await ctx.gateway.thumbnail(a.id, "thumbnail"))
        s = min(2.0, await ctx.cpu(sharpness, thumb) / 300) + 0.6 * a.is_favorite + 0.4 * (2 <= len(a.people) <= 4)
        if s > best_score:
            best, best_score = a, s
    return best


async def run_popout(ctx: PipelineContext) -> PipelineResult:
    cfg = ctx.settings.popout
    styles = [s for s in (ctx.job.params.get("styles") or cfg.styles()) if s in POPOUT_STYLES]
    want_video = bool(ctx.job.params.get("splash_video", cfg.splash_video))
    if not styles and not want_video:
        raise JobInputError("Pick at least one pop-out style.")
    ctx.progress("Finding the photo", 3)
    info = await _pick(ctx)
    if info.type != "IMAGE":
        raise JobInputError("That asset is a video; pick a photo.")

    ctx.progress("Preparing the models", 8)
    try:
        mpath = await ctx.cpu(models.ensure, models.MODNET, ctx.models_dir, ctx.model_base_url)
        dpath = await ctx.cpu(models.ensure, models.DEPTH_ANYTHING_V2_SMALL, ctx.models_dir, ctx.model_base_url) if "blur" in styles else None
    except models.ModelUnavailable as exc:
        raise JobInputError(str(exc)) from exc

    ctx.check_cancel()
    ctx.progress("Fetching the photo", 12)
    try:
        img = await ctx.cpu(decode, await ctx.gateway.original(info.id))
    except ImmichError:
        raise
    except Exception:  # noqa: BLE001 - RAW etc.: the preview is enough at this size
        img = await ctx.cpu(decode, await ctx.gateway.thumbnail(info.id, "preview"))
    if max(img.size) > WORK_LONG_SIDE:
        img.thumbnail((WORK_LONG_SIDE, WORK_LONG_SIDE))
    faces = await ctx.gateway.faces(info.id)

    ctx.progress("Cutting out the people", 22)
    matte = await ctx.cpu(lambda: _matter(str(mpath)).matte(img, faces))
    if not matte.solid.any() or matte.solid.mean() < 0.01:
        return PipelineResult("skipped", "Not eligible: no person could be cut out of this photo.", details={"reason": "no-subject"})

    caption = (ctx.job.params.get("caption") or "").strip() or _default_caption(info)
    taken_at = info.taken_at + timedelta(seconds=1)
    files: list[OutputFile] = []
    stem = Path(info.filename).stem
    depth = None
    for n, style in enumerate(styles, 1):
        ctx.check_cancel()
        ctx.progress(f"Rendering {fx.STYLE_LABELS[style].lower()}", 30 + int(55 * n / max(1, len(styles))))
        if style == "blur" and depth is None and dpath:
            depth = await ctx.cpu(estimate, _depth(str(dpath)), img, faces)
        out = await ctx.cpu(lambda s=style: _render(s, img, matte, faces, depth, caption))
        exif = Image.Exif()
        exif[0x0132] = taken_at.strftime("%Y:%m:%d %H:%M:%S")
        exif[0x0131] = "photos companion"
        files.append(OutputFile(f"popout-{ctx.job.id}-{style}.jpg", await ctx.cpu(encode_jpeg, out, 92, exif), taken_at))
    if want_video:
        ctx.progress("Rendering the colour-splash video", 90)
        size = (1920, 1080) if img.width >= img.height else (1080, 1920)
        mp4 = ctx.work_dir / f"popout-{ctx.job.id}-splash.mp4"
        await ctx.cpu(encode_mp4, fx.splash_frames(img, matte, 4.0, 30, size), size, 30, mp4, ctx.ffmpeg)
        files.append(OutputFile(mp4.name, mp4.read_bytes(), taken_at, duration_ms=4000))

    labels = ", ".join(fx.STYLE_LABELS[s].lower() for s in styles) + (" + splash video" if want_video else "")
    return PipelineResult(
        "done",
        f"{stem} · {len(faces)} face(s) · {labels}",
        files=files,
        source_ids=[info.id],
        year=info.taken_at.year,
        details={"styles": styles, "video": want_video, "caption": caption, "faces": len(faces), "source": info.filename},
    )


def _default_caption(info: AssetInfo) -> str:
    return f"{info.city} {info.taken_at.year}" if info.city else f"{info.taken_at:%B %Y}"


def _render(style, img, matte, faces, depth, caption):
    if style == "splash":
        return fx.splash(img, matte)
    if style == "spotlight":
        return fx.spotlight(img, matte)
    if style == "blur":
        return fx.portrait_blur(img, matte, depth.depth if depth else None, depth.subject if depth else None)
    if style == "frame":
        return fx.pop_frame(img, matte, faces, caption=caption)
    if style == "sticker":
        return fx.sticker(img, matte)
    raise ValueError(style)
