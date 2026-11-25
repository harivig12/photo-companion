"""Highlight video with music (phase 4). Rules: docs/research/immich-vs-google-photos.md section 5.

1. Items from an album, person(s), a date range or given IDs; photos and videos.
2. Drop blurry shots, screenshots, duplicates and repeats of the same scene.
3. Music from the mounted folder: prefer a track whose beat divides the photo cadence
   (immich-memories' track_tempo), then snap photo and clip lengths to whole beats.
4. Keep chronological order; trim to the length range, picking the best spread across time.
5. Photos get immich-memories' face-aware Ken Burns move (blurred fill when they don't fill
   the frame); videos give their middle section, blur-filled to the frame.
6. Crossfades between shots, fade in and out, music faded and trimmed to the film.

Not yet: the clips' own sound (and lowering music under speech); title cards.
"""

from __future__ import annotations

import logging
import random
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np

from ..immich import AssetInfo, ImmichError
from ..imaging import decode
from ..imaging.quality import (
    blurry_flags,
    colour_signature,
    dhash,
    duplicate_groups,
    looks_like_screenshot,
    scene_groups,
    sharpness,
)
from ..imaging.titles import pick_title_style, title_frames
from ..media import encode_mp4, find_ffmpeg
from ..vendor.immich_memories import track_tempo
from .collage import date_range
from ..vendor.immich_memories.framing_target import PictureFaces
from ..vendor.immich_memories.ken_burns_moves import FilmSubject, film_subject, planned_move
from ..vendor.immich_memories.renderer import ken_burns_frames, ken_burns_layers
from ..vendor.immich_memories.subject_framing import FaceBox as KBFace
from . import JobInputError, OutputFile, PipelineContext, PipelineResult

log = logging.getLogger(__name__)

AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}
MIN_ITEMS_AUTO = 8
MIN_ITEMS_MANUAL = 2
MIN_CLIP_MS = 1500
CLIP_SECONDS = 4.0
CROSSFADE_S = 0.4
FADE_IN_S, FADE_OUT_S = 0.6, 1.5
FPS = 30
TITLE_SECONDS = 2.6
SCORE_POOL = 150
SIZES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


@dataclass
class Item:
    info: AssetInfo
    sharp: float = 0.0
    hash: int = 0
    signature: object = None
    blurry: bool = False
    repeat_of: str | None = None
    screenshot: bool = False
    seconds: float = 0.0
    faces: list = field(default_factory=list)

    @property
    def is_video(self) -> bool:
        return self.info.type == "VIDEO"

    def score(self, mid: float) -> float:
        s = min(2.0, self.sharp / mid) if mid > 0 else 1.0
        return s + 0.5 * self.info.is_favorite + 0.3 * bool(self.info.people) + 0.3 * self.is_video


# ---- inputs ------------------------------------------------------------------------------


def _day(text: str, end: bool = False) -> datetime:
    d = datetime.strptime(text, "%Y-%m-%d").date()
    return datetime.combine(d, time.max if end else time.min, tzinfo=timezone.utc)


async def _load(ctx: PipelineContext) -> tuple[list[AssetInfo], bool]:
    p = ctx.job.params
    if p.get("asset_ids"):
        return [await ctx.gateway.asset(i) for i in p["asset_ids"]], True
    if p.get("album_id"):
        return await ctx.gateway.album_assets(p["album_id"]), False
    if p.get("event_from"):  # from a suggestion: the event's exact time span
        return await ctx.gateway.search(
            taken_after=datetime.fromisoformat(p["event_from"]) - timedelta(seconds=1),
            taken_before=datetime.fromisoformat(p["event_to"]) + timedelta(seconds=1),
        ), False
    if p.get("person_ids") or p.get("date_from") or p.get("date_to"):
        return await ctx.gateway.search(
            person_ids=p.get("person_ids"),
            taken_after=_day(p["date_from"]) if p.get("date_from") else None,
            taken_before=_day(p["date_to"], end=True) if p.get("date_to") else None,
        ), False
    raise JobInputError("Pick an album, a person, a date range or asset IDs.")


def music_tracks(folder: Path | None) -> list[Path]:
    if not folder or not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT and p.is_file())


def choose_music(tracks: list[Path], cadence: float, seed: int) -> tuple[Path | None, float | None]:
    """A track whose beat lands on the photo cadence if one does, else any; plus its tempo."""
    if not tracks:
        return None, None
    random.seed(seed)  # track_for_cadence draws with the module-level RNG
    track = track_tempo.track_for_cadence(tracks, cadence) or random.Random(seed).choice(tracks)
    bpm = track_tempo.remembered_fact("track-tempo", track, track_tempo.detect_bpm)
    return track, bpm


def on_beat(seconds: float, bpm: float | None, min_beats: int = 2) -> float:
    """`seconds` rounded to a whole number of beats (at least `min_beats`)."""
    if not bpm:
        return seconds
    beat = 60.0 / bpm
    return max(min_beats, round(seconds / beat)) * beat


def fit_length(items: list[Item], max_s: float, mid: float) -> list[Item]:
    """Drop the weakest items, spread over time, until the film fits `max_s`."""
    def total(xs: list[Item]) -> float:
        return sum(x.seconds for x in xs) - CROSSFADE_S * max(0, len(xs) - 1)

    keep = list(items)
    while len(keep) > 2 and total(keep) > max_s:
        # Remove the lowest-scoring item from the most crowded stretch of time.
        gaps = [
            (keep[i + 1].info.taken_at - keep[i - 1].info.taken_at).total_seconds() if 0 < i < len(keep) - 1 else float("inf")
            for i in range(len(keep))
        ]
        crowded = sorted(range(len(keep)), key=lambda i: gaps[i])[: max(3, len(keep) // 4)]
        keep.pop(min(crowded, key=lambda i: keep[i].score(mid)))
    return keep


# ---- rendering ---------------------------------------------------------------------------


def _kb_faces(faces) -> PictureFaces:
    boxes = tuple(
        KBFace(f.x1, f.y1, f.x2, f.y2, named=bool(f.person), person_id=f.person_id) for f in faces
    )
    return PictureFaces(boxes, frozenset())


def photo_frames(img, faces, size, seconds: float, seed_key: str, film: FilmSubject, subject_ids: frozenset[str]) -> Iterator[np.ndarray]:
    pf = _kb_faces(faces)
    pf = PictureFaces(pf.faces, subject_ids)
    params = planned_move(seed_key, pf, film, fps=FPS, duration=seconds, portrait=img.height > img.width)
    src = np.asarray(img, np.uint8)
    layers = ken_burns_layers(src, size[0], size[1], params, scale_mode="blur")
    return ken_burns_frames(layers, params)


def video_frames(path: Path, start: float, seconds: float, size, ffmpeg: str) -> Iterator[np.ndarray]:
    w, h = size
    vf = (
        f"[0:v]split[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:2[bg];"
        f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={FPS},format=rgb24"
    )
    cmd = [find_ffmpeg(ffmpeg), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(path),
           "-filter_complex", vf, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = w * h * 3
    n = round(seconds * FPS)
    last = None
    try:
        for _ in range(n):
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            last = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield last
            n -= 1
        # A clip a few frames short of the plan holds its last frame, so the cut stays on the beat.
        for _ in range(max(0, n)):
            if last is not None:
                yield last
    finally:
        proc.kill()
        proc.wait()


def assemble(segments: list[Iterator[np.ndarray]], lengths: list[int], fade: int) -> Iterator[np.ndarray]:
    """Crossfade consecutive segments by `fade` frames, and fade the whole film in and out."""
    total = sum(lengths) - fade * (len(lengths) - 1)
    fin, fout = round(FADE_IN_S * FPS), round(FADE_OUT_S * FPS)
    held: list[np.ndarray] = []
    index = 0

    def out(frame: np.ndarray) -> np.ndarray:
        nonlocal index
        k = 1.0
        if index < fin:
            k = index / fin
        elif index >= total - fout:
            k = max(0.0, (total - 1 - index) / fout)
        index += 1
        return frame if k >= 1.0 else (frame.astype(np.float32) * k).astype(np.uint8)

    for s, (seg, n) in enumerate(zip(segments, lengths)):
        last = s == len(segments) - 1
        tail_from = n if last else n - fade
        new_held: list[np.ndarray] = []
        prev = None
        for i in range(n):
            frame = next(seg, None)
            if frame is None:  # a segment a frame short: hold its last frame
                frame = prev if prev is not None else (held[-1] if held else np.zeros_like(held[0]))
            prev = frame
            if held and i < len(held):
                a = (i + 1) / (len(held) + 1)
                frame = (held[i].astype(np.float32) * (1 - a) + frame.astype(np.float32) * a).astype(np.uint8)
            if i >= tail_from:
                new_held.append(frame)
                continue
            yield out(frame)
        held = new_held
    for frame in held:
        yield out(frame)


def mux_music(video: Path, music: Path | None, seconds: float, out: Path, ffmpeg: str) -> Path:
    if music is None:
        video.replace(out)
        return out
    fade_st = max(0.0, seconds - 2.0)
    cmd = [find_ffmpeg(ffmpeg), "-v", "error", "-y", "-i", str(video), "-stream_loop", "-1", "-i", str(music),
           "-t", f"{seconds:.3f}", "-map", "0:v", "-map", "1:a", "-c:v", "copy",
           "-af", f"afade=t=in:d=1,afade=t=out:st={fade_st:.2f}:d=2,volume=0.9",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {proc.stderr.decode(errors='replace')[:300]}")
    return out


async def _default_title(ctx: PipelineContext, film: list[Item]) -> str:
    """Album name, else the most common city, else the month."""
    p = ctx.job.params
    if p.get("album_id"):
        try:
            for a in await ctx.gateway.list_albums():
                if a["id"] == p["album_id"]:
                    return a["name"][:40]
        except ImmichError:
            pass
    cities = [i.info.city for i in film if i.info.city]
    if cities:
        return max(set(cities), key=cities.count)
    return f"{film[0].info.taken_at:%B %Y}"


# ---- pipeline ----------------------------------------------------------------------------


async def run_highlight(ctx: PipelineContext) -> PipelineResult:
    cfg = ctx.settings.highlight
    track_tempo.FFMPEG = find_ffmpeg(ctx.ffmpeg)
    ctx.progress("Loading photos and videos", 2)
    infos, manual = await _load(ctx)
    usable = sorted(
        (a for a in infos if a.usable and (a.type == "IMAGE" or (a.type == "VIDEO" and (a.duration_ms or 0) >= MIN_CLIP_MS))),
        key=lambda a: a.taken_at,
    )
    need = MIN_ITEMS_MANUAL if manual else MIN_ITEMS_AUTO
    if len(usable) < need:
        raise JobInputError(f"A highlight video needs at least {need} photos or clips; found {len(usable)}.")

    pool = usable
    if len(pool) > SCORE_POOL:
        favs = [a for a in pool if a.is_favorite][: SCORE_POOL // 3]
        step = len(pool) / (SCORE_POOL - len(favs))
        rest = [pool[int(i * step)] for i in range(SCORE_POOL - len(favs))]
        pool = sorted({a.id: a for a in favs + rest}.values(), key=lambda a: a.taken_at)
    ctx.log(f"{len(usable)} usable items, scoring {len(pool)}")

    items: list[Item] = []
    for n, info in enumerate(pool, 1):
        ctx.check_cancel()
        ctx.progress(f"Checking item {n} of {len(pool)}", 3 + int(22 * n / len(pool)))
        it = Item(info, screenshot=info.type == "IMAGE" and looks_like_screenshot(info.filename, info.mime, info.make))
        try:
            thumb = await ctx.cpu(decode, await ctx.gateway.thumbnail(info.id, "thumbnail"))
        except ImmichError:
            raise
        except Exception:  # noqa: BLE001
            ctx.log(f"skip {info.filename}: thumbnail could not be decoded")
            continue
        it.sharp = await ctx.cpu(sharpness, thumb)
        it.hash = await ctx.cpu(dhash, thumb)
        it.signature = await ctx.cpu(colour_signature, thumb)
        items.append(it)

    photos = [i for i in items if not i.is_video]
    for it, flag in zip(photos, blurry_flags([i.sharp for i in photos])):
        it.blurry = flag
    for groups in (
        duplicate_groups([i.hash for i in items]),
        scene_groups([i.info.taken_at.timestamp() for i in items], [i.hash for i in items], [i.signature for i in items]),
    ):
        best: dict[int, Item] = {}
        for it, g in zip(items, groups):
            if g not in best or it.score(1) > best[g].score(1):
                best[g] = it
        for it, g in zip(items, groups):
            if best[g] is not it and not it.repeat_of:
                it.repeat_of = best[g].info.id
    kept = items if manual else [i for i in items if not i.blurry and not i.repeat_of and not i.screenshot]
    if len(kept) < need:
        raise JobInputError(f"Only {len(kept)} usable item(s) after removing blurry shots, repeats and screenshots (needs {need}).")

    # Music first: its tempo sets the cut lengths.
    ctx.progress("Choosing music", 27)
    tracks = music_tracks(ctx.music_dir)
    wanted = ctx.job.params.get("music")
    if wanted and any(t.name == wanted for t in tracks):
        tracks = [t for t in tracks if t.name == wanted]  # an edited film keeps its track
    track, bpm = await ctx.cpu(choose_music, tracks, cfg.seconds_per_photo, ctx.job.id)
    beats = cfg.cut_on_beat and bool(bpm)
    photo_s = on_beat(cfg.seconds_per_photo, bpm if beats else None)
    for it in kept:
        if it.is_video:
            it.seconds = min(on_beat(CLIP_SECONDS, bpm if beats else None), (it.info.duration_ms or 0) / 1000)
        else:
            it.seconds = photo_s
        it.seconds += CROSSFADE_S  # the overlap is shared with the next shot
    ctx.log(f"music: {track.name if track else 'none'} ({bpm or '?'} bpm); photo {photo_s:.2f} s")

    from statistics import median

    mid = median(i.sharp for i in kept) or 1.0
    # The title card counts towards the film's length.
    budget = cfg.max_seconds - (on_beat(TITLE_SECONDS, bpm if beats else None) if cfg.title_card else 0)
    film = fit_length(kept, max(budget, 6.0), mid)
    total_s = sum(i.seconds for i in film) - CROSSFADE_S * (len(film) - 1)
    ctx.log(f"{len(film)} shots, {total_s:.1f} s")

    # Who the film is about (person search): Ken Burns pushes in on them.
    subject_ids = frozenset(ctx.job.params.get("person_ids") or [])
    names = [await ctx.gateway.person_name(p) for p in subject_ids] if subject_ids else []
    film_kind = film_subject(names) if names else FilmSubject.OTHER

    size = SIZES[cfg.aspect]
    fade = round(CROSSFADE_S * FPS)
    segments, lengths = [], []
    title_photo = None
    ffmpeg = find_ffmpeg(ctx.ffmpeg)
    for n, it in enumerate(film, 1):
        ctx.check_cancel()
        ctx.progress(f"Fetching shot {n} of {len(film)}", 30 + int(20 * n / len(film)))
        frames_n = round(it.seconds * FPS)
        if it.is_video:
            path = ctx.work_dir / f"clip-{n}.mp4"
            path.write_bytes(await ctx.gateway.video(it.info.id))
            dur = (it.info.duration_ms or 0) / 1000
            start = max(0.0, min(dur * 0.25, dur - it.seconds))
            segments.append(video_frames(path, start, it.seconds, size, ffmpeg))
        else:
            img = None
            try:
                img = await ctx.cpu(decode, await ctx.gateway.original(it.info.id))
            except ImmichError:
                raise
            except Exception:  # noqa: BLE001
                ctx.log(f"{it.info.filename}: original not decodable, using the preview")
            if img is None:
                img = await ctx.cpu(decode, await ctx.gateway.thumbnail(it.info.id, "preview"))
            limit = int(max(size) * 1.6)  # Ken Burns zooms up to 1.5x
            if max(img.size) > limit:
                img.thumbnail((limit, limit))
            faces = await ctx.gateway.faces(it.info.id)
            # +0.5 frame so the renderer's int(fps * duration) never comes out one short.
            segments.append(photo_frames(img, faces, size, (frames_n + 0.5) / FPS, it.info.id, film_kind, subject_ids))
            if title_photo is None:
                title_photo = img
        lengths.append(frames_n)

    # Title card (editable text): the job's caption, else the album / event name; date line below.
    title_style = None
    title_text = (ctx.job.params.get("caption") or "").strip() or await _default_title(ctx, film)
    subtitle = (ctx.job.params.get("subtitle") or "").strip() or date_range([i.info.taken_at for i in film])
    if cfg.title_card and title_text:
        title_style = pick_title_style(ctx.job.params.get("title_style") or cfg.title_style, ctx.job.id)
        if title_photo is None:
            title_photo = await ctx.cpu(decode, await ctx.gateway.thumbnail(film[0].info.id, "preview"))
        card_n = round((on_beat(TITLE_SECONDS, bpm if beats else None) + CROSSFADE_S) * FPS)
        segments.insert(0, title_frames(title_photo, title_text, subtitle, size, card_n / FPS, FPS, title_style))
        lengths.insert(0, card_n)
        ctx.log(f"title card: '{title_text}' / '{subtitle}' ({title_style})")

    silent = ctx.work_dir / "film-video.mp4"
    total_frames = sum(lengths) - fade * (len(lengths) - 1)

    def render() -> None:
        def tick(i: int) -> None:
            if i % 15 == 0:
                ctx.progress(f"Rendering frame {i} of {total_frames}", 50 + int(45 * i / total_frames))
            if ctx.cancel_requested():
                raise KeyboardInterrupt

        encode_mp4(assemble(segments, lengths, fade), size, FPS, silent, ffmpeg, on_frame=tick)

    try:
        await ctx.cpu(render)
    except KeyboardInterrupt:
        ctx.check_cancel()
        raise
    seconds = total_frames / FPS
    final = await ctx.cpu(mux_music, silent, track, seconds, ctx.work_dir / f"highlight-{ctx.job.id}.mp4", ffmpeg)

    taken_at = max(i.info.taken_at for i in film) + timedelta(seconds=1)
    photos_n = sum(not i.is_video for i in film)
    notes = [f"music {track.name}" + (f" at {bpm:.0f} bpm, cuts on the beat" if beats else "") if track else "no music (folder empty)"]
    left = {
        "blurry": sum(i.blurry for i in items),
        "repeat": sum(bool(i.repeat_of) for i in items),
        "screenshot": sum(i.screenshot for i in items),
    }
    if not manual and (parts := [f"{v} {k}" for k, v in left.items() if v]):
        notes.append("left out " + ", ".join(parts))
    if len(film) < len(kept):
        notes.append(f"{len(kept) - len(film)} trimmed to fit {cfg.max_seconds} s")
    notes.append("clip sound not mixed yet")
    years = sorted(i.info.taken_at.year for i in film)
    return PipelineResult(
        "done",
        f"{len(film)} shots ({photos_n} photos, {len(film) - photos_n} clips) · {seconds:.0f} s · {size[0]}×{size[1]} · " + "; ".join(notes),
        files=[OutputFile(final.name, final.read_bytes(), taken_at, duration_ms=round(seconds * 1000))],
        source_ids=[i.info.id for i in film],
        year=years[len(years) // 2],
        details={
            "music": track.name if track else None, "bpm": bpm, "photo_seconds": round(photo_s, 3),
            "title": title_text if title_style else None, "subtitle": subtitle if title_style else None,
            "title_style": title_style,
            "shots": [{"id": i.info.id, "file": i.info.filename, "type": i.info.type, "seconds": round(i.seconds - CROSSFADE_S, 2)} for i in film],
        },
    )
