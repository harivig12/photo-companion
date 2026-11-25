"""Cinematic photo pipeline (phase 3). Rules: docs/research/immich-vs-google-photos.md section 4.

1. One photo; check it's eligible (not a screenshot, big enough, enough depth).
2. Depth Anything V2 Small on CPU -> normalised depth, edges cleaned, people pulled forward.
3. Pick the camera move with the least visible stretch (or the preset from Settings).
4. Render frames with OpenCV and encode an H.264 MP4 with ffmpeg.
5. Also return the depth map as a 16-bit PNG (Settings: "Upload PNG").
"""

from __future__ import annotations

import io
import logging
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import models
from ..imaging import decode
from ..imaging.depth import DepthModel, _body_mask, depth_png, estimate
from ..imaging.parallax import choose_motion, frames, output_size
from ..imaging.quality import looks_like_screenshot
from ..media import encode_mp4
from . import JobInputError, OutputFile, PipelineContext, PipelineResult

log = logging.getLogger(__name__)
MIN_SHORT_SIDE = 1000
PREVIEW_LONG_SIDE = 1440


@lru_cache(maxsize=2)
def _model(path: str) -> DepthModel:
    return DepthModel(Path(path))


async def run_cinematic(ctx: PipelineContext) -> PipelineResult:
    cfg = ctx.settings.cinematic
    ids = ctx.job.params.get("asset_ids") or []
    force = bool(ctx.job.params.get("force"))
    if len(ids) != 1:
        raise JobInputError("A cinematic photo uses exactly one photo.")
    ctx.progress("Loading photo", 2)
    info = await ctx.gateway.asset(ids[0])
    if info.type != "IMAGE":
        raise JobInputError("That asset is a video; pick a photo.")
    if not force:
        if looks_like_screenshot(info.filename, info.mime, info.make):
            return PipelineResult("skipped", "Not eligible: looks like a screenshot.", details={"reason": "screenshot"})
        if info.width and info.height and min(info.width, info.height) < MIN_SHORT_SIDE:
            return PipelineResult("skipped", f"Not eligible: too small ({info.width}×{info.height}).", details={"reason": "small"})

    ctx.progress("Preparing the depth model", 5)
    try:
        model_path = await ctx.cpu(models.ensure, models.DEPTH_ANYTHING_V2_SMALL, ctx.models_dir, ctx.model_base_url)
    except models.ModelUnavailable as exc:
        raise JobInputError(str(exc)) from exc
    model = await ctx.cpu(_model, str(model_path))

    ctx.check_cancel()
    ctx.progress("Fetching photo", 10)
    out_w, out_h = output_size(info.width or 4, info.height or 3, cfg.resolution)
    need = max(out_w, out_h) * 1.15
    img = None
    if need > PREVIEW_LONG_SIDE:
        try:
            img = await ctx.cpu(decode, await ctx.gateway.original(info.id))
        except Exception:  # noqa: BLE001 - RAW etc.: use the preview
            ctx.log("original not decodable, using the preview")
    if img is None:
        img = await ctx.cpu(decode, await ctx.gateway.thumbnail(info.id, "preview"))
    # Work at no more than ~1.3x the output size: sharper than needed costs CPU for nothing.
    limit = int(max(out_w, out_h) * 1.3)
    if max(img.size) > limit:
        img.thumbnail((limit, limit))
    out_w, out_h = output_size(*img.size, cfg.resolution)
    faces = await ctx.gateway.faces(info.id)

    ctx.check_cancel()
    ctx.progress("Estimating depth", 20)
    res = await ctx.cpu(estimate, model, img, faces)
    ctx.log(f"depth spread {res.spread:.2f}, subject depth {res.subject:.2f}, {len(faces)} face(s)")
    if not force and res.spread < cfg.min_depth_spread:
        return PipelineResult(
            "skipped",
            f"Not eligible: flat scene (depth spread {res.spread:.2f}, minimum {cfg.min_depth_spread:.2f}).",
            details={"reason": "flat", "spread": round(res.spread, 3)},
        )

    ctx.check_cancel()
    ctx.progress("Choosing the camera move", 35)
    subject_mask = _body_mask(res.depth.shape, faces) if faces else None
    motion, scores = await ctx.cpu(choose_motion, res.depth, (out_w, out_h), res.subject, cfg.motion, subject_mask)
    ctx.log(f"camera: {motion.preset}, amplitude {motion.amplitude:.3f} (stretch scores {scores})")

    n_frames = cfg.seconds * cfg.fps
    image = np.asarray(img, np.uint8)
    out = ctx.work_dir / f"cinematic-{ctx.job.id}.mp4"

    def render() -> None:
        def tick(i: int) -> None:
            if i % 10 == 0:
                ctx.progress(f"Rendering frame {i} of {n_frames}", 40 + int(50 * i / n_frames))
            if ctx.cancel_requested():
                raise KeyboardInterrupt  # stops ffmpeg; turned into a cancel below

        encode_mp4(frames(image, res.depth, (out_w, out_h), res.subject, motion, n_frames), (out_w, out_h), cfg.fps, out, ctx.ffmpeg, on_frame=tick)

    try:
        await ctx.cpu(render)
    except KeyboardInterrupt:
        ctx.check_cancel()
        raise
    taken_at = info.taken_at + timedelta(seconds=1)
    files = [OutputFile(out.name, out.read_bytes(), taken_at, duration_ms=cfg.seconds * 1000)]
    if cfg.upload_depth_map:
        buf = io.BytesIO()
        depth_png(res.depth).save(buf, "PNG", optimize=True)
        files.append(OutputFile(f"cinematic-{ctx.job.id}-depth.png", buf.getvalue(), taken_at, role="depth"))

    stem = Path(info.filename).stem
    return PipelineResult(
        "done",
        f"{stem} · {motion.preset} · {out_w}×{out_h} · {cfg.seconds} s at {cfg.fps} fps · depth spread {res.spread:.2f}",
        files=files,
        source_ids=[info.id],
        year=info.taken_at.year,
        details={"preset": motion.preset, "amplitude": motion.amplitude, "stretch": scores, "spread": round(res.spread, 3), "faces": len(faces)},
    )
