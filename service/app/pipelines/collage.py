"""Collage pipeline (phase 2). Rules: docs/research/immich-vs-google-photos.md section 3.

Steps:
1. Candidates: the album (auto-pick) or the given asset IDs (used as given).
2. Score on small thumbnails: sharpness, near-duplicates, screenshots, faces, favourites.
3. Pick up to N photos spread across the time range (album source only).
4. Plan the layout with the least cropping (rows / columns / hero).
5. Render at 3000 px on the long side with face-aware crops; JPEG with the shoot date in EXIF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median

from PIL import Image

from ..immich import AssetInfo, FaceBox, ImmichError
from ..imaging import decode, encode_jpeg, frames
from ..imaging.collage_render import canvas_size, render, slot_pixels
from ..imaging.layout import parse_ratio, plan
from ..imaging.quality import (
    blurry_flags,
    colour_signature,
    dhash,
    duplicate_groups,
    looks_like_screenshot,
    scene_groups,
    sharpness,
)
from . import JobInputError, OutputFile, PipelineContext, PipelineResult

log = logging.getLogger(__name__)

LONG_SIDE = 3000
SCORE_POOL = 80  # most photos scored per album job (thumbnails are ~250 px, cheap)
PREVIEW_LONG_SIDE = 1440  # Immich's default preview size
MIN_PHOTOS = 2
STYLE_MAX = {"grid": 12, "prints": 12, "scatter": 7, "fan": 5}  # what each style can show well
STYLE_MIN = {"grid": 2, "prints": 2, "scatter": 3, "fan": 3}


def date_range(dates: list) -> str:
    """'13-15 Sep 2026', '30 Aug - 2 Sep 2026' or '28 Dec 2025 - 2 Jan 2026' (with en dashes)."""
    dash = "\u2013"
    a, b = min(dates), max(dates)
    if a.date() == b.date():
        return f"{a.day} {a:%b %Y}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}{dash}{b.day} {b:%b %Y}"
    if a.year == b.year:
        return f"{a.day} {a:%b} {dash} {b.day} {b:%b %Y}"
    return f"{a.day} {a:%b %Y} {dash} {b.day} {b:%b %Y}"


@dataclass
class Candidate:
    info: AssetInfo
    thumb: Image.Image | None = None
    sharp: float = 0.0
    hash: int = 0
    faces: list[FaceBox] = field(default_factory=list)
    blurry: bool = False
    duplicate_of: str | None = None
    same_scene_as: str | None = None
    screenshot: bool = False
    signature: object = None

    @property
    def aspect(self) -> float:
        if self.thumb is not None:
            return self.thumb.width / self.thumb.height
        if self.info.width and self.info.height:
            return self.info.width / self.info.height
        return 1.0

    def score(self, mid_sharp: float) -> float:
        sharp = min(2.0, self.sharp / mid_sharp) if mid_sharp > 0 else 1.0
        return sharp + 0.6 * bool(self.faces) + 0.4 * self.info.is_favorite + 0.2 * bool(self.info.people)


def _spread(items: list, k: int, key) -> list:
    """Pick k items spread across the (time-ordered) list: the best of each of k equal buckets."""
    if len(items) <= k:
        return items
    out = []
    for b in range(k):
        bucket = items[b * len(items) // k : (b + 1) * len(items) // k]
        if bucket:
            out.append(max(bucket, key=key))
    return out


async def _load_candidates(ctx: PipelineContext) -> tuple[list[AssetInfo], bool]:
    params = ctx.job.params
    ids = params.get("asset_ids") or []
    if ids:
        infos = [await ctx.gateway.asset(i) for i in ids]
        return infos, True
    if params.get("album_id"):
        return await ctx.gateway.album_assets(params["album_id"]), False
    if params.get("event_from"):  # from a suggestion: the event's exact time span
        return await ctx.gateway.search(
            taken_after=datetime.fromisoformat(params["event_from"]) - timedelta(seconds=1),
            taken_before=datetime.fromisoformat(params["event_to"]) + timedelta(seconds=1),
        ), False
    raise JobInputError("Pick an album or give asset IDs.")



async def _caption(ctx: PipelineContext, picked: list[Candidate]) -> tuple[str, str]:
    """Title and date line for scatter / fan styles: job caption, else the album name."""
    params = ctx.job.params
    title = (params.get("caption") or "").strip()
    if not title and params.get("album_id"):
        try:
            title = next((a["name"] for a in await ctx.gateway.list_albums() if a["id"] == params["album_id"]), "")
        except ImmichError:
            title = ""
    return title[:40], date_range([c.info.taken_at for c in picked])


RANDOM_STYLES = ("prints", "scatter", "fan")


def pick_style(requested: str, job_id: int, photo_count: int, manual: bool) -> str:
    """Resolve 'random' to one of the tilted styles, seeded by the job so a retry looks the same.

    Only styles that suit the number of photos are drawn: a manual pick of 9 can't be a fan.
    """
    if requested != "random":
        return requested
    import random

    fits = [s for s in RANDOM_STYLES if photo_count >= STYLE_MIN[s] and (not manual or photo_count <= STYLE_MAX[s])]
    return random.Random(job_id).choice(fits or ["prints"])


async def run_collage(ctx: PipelineContext) -> PipelineResult:
    cfg = ctx.settings.collage
    ctx.progress("Loading photos", 2)
    infos, manual = await _load_candidates(ctx)
    images = sorted((a for a in infos if a.type == "IMAGE" and a.usable), key=lambda a: a.taken_at)
    dropped_hidden = len(infos) - len(images)
    style = pick_style(ctx.job.params.get("style") or cfg.style, ctx.job.id, len(images), manual)
    if style not in STYLE_MAX:
        raise JobInputError(f"Unknown collage style: {style}")
    limit = min(cfg.max_photos_manual if manual else cfg.max_photos_auto, STYLE_MAX[style])
    if ctx.job.params.get("edit_of"):  # an edited re-make keeps exactly the photos it had
        limit = max(limit, min(len(images), STYLE_MAX[style]))
    need = max(MIN_PHOTOS, STYLE_MIN[style])
    if manual and len(images) > limit:
        raise JobInputError(f"The {style} style takes up to {limit} photos; {len(images)} were given.")
    if len(images) < need:
        raise JobInputError(f"The {style} style needs at least {need} visible photos; found {len(images)}.")

    # Score a bounded, time-spread pool; favourites always make the pool.
    pool = images
    if not manual and len(images) > SCORE_POOL:
        favs = [a for a in images if a.is_favorite][: SCORE_POOL // 2]
        rest = _spread([a for a in images if not a.is_favorite], SCORE_POOL - len(favs), key=lambda a: 0)
        pool = sorted(favs + rest, key=lambda a: a.taken_at)
    ctx.log(f"{len(images)} visible photos, scoring {len(pool)}, style {style}")

    cands: list[Candidate] = []
    for n, info in enumerate(pool, 1):
        ctx.check_cancel()
        ctx.progress(f"Checking photo {n} of {len(pool)}", 5 + int(35 * n / len(pool)))
        c = Candidate(info, screenshot=looks_like_screenshot(info.filename, info.mime, info.make))
        try:
            c.thumb = await ctx.cpu(decode, await ctx.gateway.thumbnail(info.id, "thumbnail"))
        except ImmichError:
            raise
        except Exception:  # noqa: BLE001 - an undecodable photo is just skipped
            ctx.log(f"skip {info.filename}: thumbnail could not be decoded")
            continue
        c.sharp = await ctx.cpu(sharpness, c.thumb)
        c.hash = await ctx.cpu(dhash, c.thumb)
        c.signature = await ctx.cpu(colour_signature, c.thumb)
        cands.append(c)
    if len(cands) < need:
        raise JobInputError("Not enough photos could be read from Immich.")

    for c, flag in zip(cands, blurry_flags([c.sharp for c in cands])):
        c.blurry = flag
    _mark_groups(cands, duplicate_groups([c.hash for c in cands]), "duplicate_of")
    _mark_groups(
        cands,
        scene_groups([c.info.taken_at.timestamp() for c in cands], [c.hash for c in cands], [c.signature for c in cands]),
        "same_scene_as",
    )

    if manual:
        picked = cands  # the user chose these; only warn
        for c in picked:
            c.faces = await ctx.gateway.faces(c.info.id)
    else:
        usable = [
            c for c in cands
            if not (c.blurry and cfg.skip_blurry) and not c.duplicate_of and not c.same_scene_as and not c.screenshot
        ]
        if len(usable) < need:
            raise JobInputError(
                f"Only {len(usable)} usable photo(s) after removing blurry shots, repeats and screenshots."
            )
        # Faces only for the shortlist (one call per photo).
        shortlist = _spread(usable, min(len(usable), limit * 2), key=lambda c: c.sharp)
        for c in shortlist:
            c.faces = await ctx.gateway.faces(c.info.id)
        mid = median(c.sharp for c in shortlist)
        picked = _spread(shortlist, limit, key=lambda c: c.score(mid))

    ctx.check_cancel()
    ctx.progress("Choosing a layout", 45)
    ratio = 1.0 if style in ("scatter", "fan") else parse_ratio(cfg.canvas_ratio)
    mid = median(c.sharp for c in picked) or 1.0
    scores = [c.score(mid) for c in picked]
    layout = plan([c.aspect for c in picked], scores, ratio)
    size = canvas_size(ratio, LONG_SIDE)

    # Fetch each photo at the size its slot needs (previews are enough for small cells).
    slot_long = [max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in slot_pixels(layout, size, round(cfg.border_px * LONG_SIDE / 1000))]
    if style in ("scatter", "fan"):
        slot_long = [LONG_SIDE // 2] * len(picked)
    full: list[Image.Image] = []
    for n, (c, need_px) in enumerate(zip(picked, slot_long), 1):
        ctx.check_cancel()
        ctx.progress(f"Fetching photo {n} of {len(picked)}", 50 + int(30 * n / len(picked)))
        img = None
        if need_px > PREVIEW_LONG_SIDE:
            try:
                img = await ctx.cpu(decode, await ctx.gateway.original(c.info.id))
            except ImmichError:
                raise
            except Exception:  # noqa: BLE001 - e.g. RAW originals: fall back to the preview
                ctx.log(f"{c.info.filename}: original not decodable, using the preview")
        if img is None:
            img = await ctx.cpu(decode, await ctx.gateway.thumbnail(c.info.id, "preview"))
        full.append(img)

    ctx.progress("Rendering", 85)
    seed = ctx.job.id
    photos = [frames.Photo(img, c.faces, s) for img, c, s in zip(full, picked, scores)]
    if style == "grid":
        canvas = await ctx.cpu(lambda: render(
            full, [c.faces for c in picked], layout, ratio, LONG_SIDE,
            cfg.border_px, cfg.radius_px, cfg.border_colour, cfg.face_aware_crop,
        ))
        layout_name = layout.name
    elif style == "prints":
        canvas = await ctx.cpu(lambda: frames.prints(photos, size, ratio, seed=seed))
        layout_name = f"tilted prints, {layout.name}"
    else:
        title, dates = await _caption(ctx, picked)
        if style == "scatter":
            caption = f"{title} · {dates}" if title else dates
            canvas = await ctx.cpu(lambda: frames.scatter(photos, size, seed=seed, caption=caption))
            layout_name = "polaroid scatter"
        else:
            canvas = await ctx.cpu(lambda: frames.fan(photos, size, title=title or None, subtitle=dates))
            layout_name = "gallery fan"
    ctx.log(f"layout: {layout_name}")

    newest = max(c.info.taken_at for c in picked)
    taken_at = newest + timedelta(seconds=1)  # sits right after its photos on the timeline
    exif = Image.Exif()
    exif[0x0132] = taken_at.strftime("%Y:%m:%d %H:%M:%S")  # DateTime
    exif[0x0131] = "photos companion"  # Software
    data = await ctx.cpu(encode_jpeg, canvas, 90, exif)

    notes = []
    if manual:
        if b := sum(c.blurry for c in picked):
            notes.append(f"{b} look blurry")
        if d := sum(bool(c.duplicate_of or c.same_scene_as) for c in picked):
            notes.append(f"{d} repeat another shot")
    else:
        left_out = {
            "blurry": sum(c.blurry for c in cands),
            "duplicate": sum(bool(c.duplicate_of) for c in cands),
            "same scene": sum(bool(c.same_scene_as and not c.duplicate_of) for c in cands),
            "screenshot": sum(c.screenshot for c in cands),
        }
        if parts := [f"{v} {k}" for k, v in left_out.items() if v]:
            notes.append("left out " + ", ".join(parts))
    if dropped_hidden:
        notes.append(f"{dropped_hidden} hidden, archived, trashed or video item(s) ignored")
    message = f"{len(picked)} photos · {layout_name} · {canvas.width}×{canvas.height}" + (f" · {'; '.join(notes)}" if notes else "")

    years = sorted(c.info.taken_at.year for c in picked)
    return PipelineResult(
        status="done",
        message=message,
        files=[OutputFile(f"collage-{ctx.job.id}-{style}.jpg", data, taken_at)],
        source_ids=[c.info.id for c in picked],
        year=years[len(years) // 2],
        details={
            "style": style,
            "layout": layout_name,
            "photos": [
                {"id": c.info.id, "file": c.info.filename, "sharpness": round(c.sharp, 1), "faces": len(c.faces),
                 "favorite": c.info.is_favorite, "blurry": c.blurry}
                for c in picked
            ],
        },
    )


def _mark_groups(cands: list[Candidate], groups: list[int], attr: str) -> None:
    """Within each group keep the sharpest photo; point the others at it via `attr`."""
    best: dict[int, Candidate] = {}
    for c, g in zip(cands, groups):
        if g not in best or c.sharp > best[g].sharp:
            best[g] = c
    for c, g in zip(cands, groups):
        if best[g] is not c:
            setattr(c, attr, best[g].info.id)
