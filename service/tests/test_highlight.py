import asyncio
import io
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.immich import AssetInfo, ImmichGateway
from app.jobs import JobStore
from app.pipelines.highlight import FPS, Item, assemble, choose_music, fit_length, music_tracks, on_beat
from app.settings_store import SettingsStore
from app.vendor.immich_memories import beat_grid, track_tempo
from app.worker import Worker
from fake_immich import ALBUM_ID, FakeImmich, asset, with_fake

FFMPEG = os.environ.get("PHOTOS_FFMPEG") or shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def info(i: int, minutes: int = 60, video: bool = False) -> AssetInfo:
    return AssetInfo(
        id=f"id{i}", type="VIDEO" if video else "IMAGE", filename=f"f{i}", mime="image/jpeg", width=4, height=3,
        taken_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes * i),
        is_favorite=False, visibility="timeline", is_trashed=False, duration_ms=10_000 if video else None,
    )


def test_on_beat_rounds_to_whole_beats():
    assert on_beat(3.0, None) == 3.0
    assert on_beat(3.0, 100) == pytest.approx(3.0)  # 5 beats of 0.6 s
    assert on_beat(3.0, 120) == pytest.approx(3.0)  # 6 beats of 0.5 s
    assert on_beat(3.1, 90) == pytest.approx(60 / 90 * 5)
    assert on_beat(0.2, 120) == pytest.approx(1.0)  # at least 2 beats


def test_vendored_beat_grid_behaves_as_upstream():
    assert beat_grid.beat_aligned_bpm(100, 3.0, (60, 140)) == 100
    assert beat_grid.beat_aligned_bpm(95, 3.0, (60, 140)) == 100  # nudged onto 5 beats


def test_fit_length_keeps_order_and_fits():
    items = [Item(info(i), sharp=100 + i, seconds=3.4) for i in range(30)]
    kept = fit_length(items, 45, 100)
    total = sum(i.seconds for i in kept) - 0.4 * (len(kept) - 1)
    assert total <= 45 and len(kept) >= 10
    assert [i.info.id for i in kept] == sorted((i.info.id for i in kept), key=lambda s: int(s[2:]))


def test_assemble_frame_count_and_fades():
    seg = lambda v, n: (np.full((4, 4, 3), v, np.uint8) for _ in range(n))  # noqa: E731
    lengths, fade = [90, 90, 90], 12  # 3 s shots, 0.4 s crossfades
    frames = list(assemble([seg(200, 90), seg(100, 90), seg(50, 90)], lengths, fade))
    assert len(frames) == sum(lengths) - fade * 2
    assert frames[0].max() == 0  # fade in from black
    assert frames[40].max() == 200  # shot 1 at full brightness
    assert 200 > frames[78 + 6].max() > 100  # inside the 1 -> 2 crossfade
    assert frames[120].max() == 100  # shot 2
    assert frames[-1].max() == 0  # fade out to black


def test_assemble_tolerates_a_short_segment():
    seg = lambda v, n: (np.full((2, 2, 3), v, np.uint8) for _ in range(n))  # noqa: E731
    frames = list(assemble([seg(200, 25), seg(100, 30)], [30, 30], 6))
    assert len(frames) == 54


@needs_ffmpeg
def _make_track(path: Path, bpm: float, seconds: int = 20) -> Path:
    period = 60 / bpm
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"aevalsrc='0.5*sin(2*PI*660*t)*exp(-25*mod(t,{period}))':s=22050:d={seconds}",
                    str(path)], check=True)
    return path


@needs_ffmpeg
def test_choose_music_prefers_track_on_the_cadence(tmp_path):
    track_tempo.FFMPEG = FFMPEG
    _make_track(tmp_path / "a-100.wav", 100)  # 3.0 s = 5 beats
    _make_track(tmp_path / "b-111.wav", 111)  # 3.0 s = 5.55 beats: off the grid
    track, bpm = choose_music(music_tracks(tmp_path), 3.0, seed=1)
    assert track.name == "a-100.wav" and abs(bpm - 100) < 4


def test_no_music_folder():
    assert music_tracks(None) == [] and choose_music([], 3.0, 1) == (None, None)


def _jpeg(seed):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@needs_ffmpeg
def test_highlight_pipeline_end_to_end(config, db, tmp_path):
    from app.pipelines.highlight import run_highlight

    music = tmp_path / "music"
    music.mkdir()
    _make_track(music / "beat.wav", 100, 40)
    clip = tmp_path / "clip.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=6",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    start = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)
    assets, media = [], {}
    for i in range(10):
        aid = f"{i:08d}-0000-4000-8000-000000000000"
        taken = (start + timedelta(hours=i)).isoformat().replace("+00:00", ".000Z")
        if i == 4:
            assets.append(asset(aid, taken, "VID_4.mp4", 320, 240, type="VIDEO", duration=6000, originalMimeType="video/mp4"))
            media[aid] = clip.read_bytes()
        else:
            assets.append(asset(aid, taken, f"IMG_{i}.jpg", 400, 300))
            media[aid] = _jpeg(i)
    # The video's thumbnail is a still.
    thumbs = dict(media, **{assets[4]["id"]: _jpeg(99)})
    fake = FakeImmich().library(assets, media)
    fake.on_match("GET", lambda r: r.path.endswith("/thumbnail"), lambda r: __import__("fake_immich").Reply(200, thumbs[r.path.split("/")[3]]))
    fake.patterns.insert(0, fake.patterns.pop())  # thumbnails first
    cfg = config.model_copy(update={
        "photos_output_mode": "folder", "photos_output_dir": tmp_path / "out", "photos_ffmpeg": FFMPEG, "photos_music_dir": music,
    })
    store = SettingsStore(db)
    s = store.get()
    s.highlight.aspect, s.highlight.min_seconds, s.highlight.max_seconds = "1:1", 10, 20
    store.save(s)
    job = JobStore(db).enqueue("highlight", "Trip", {"album_id": ALBUM_ID, "asset_ids": []})

    async def go(c):
        async with ImmichGateway(c) as gw:
            await Worker(JobStore(db), store, gw, c, registry={"highlight": run_highlight}).run_once()

    asyncio.run(with_fake(fake, cfg, go))
    done = JobStore(db).get(job.id, with_log=True)
    assert done.status == "done", (done.message, done.log)
    out = tmp_path / "out" / f"{job.id:05d}-highlight" / f"highlight-{job.id}.mp4"
    ffprobe = Path(FFMPEG).with_name(Path(FFMPEG).name.replace("ffmpeg", "ffprobe"))
    probe = subprocess.run([str(ffprobe) if ffprobe.exists() else "ffprobe", "-v", "error",
                            "-show_entries", "stream=codec_type,width,height", "-show_entries", "format=duration",
                            "-of", "compact", str(out)], capture_output=True, text=True).stdout
    assert "codec_type=video|width=1080|height=1080" in probe and "codec_type=audio" in probe
    duration = float(probe.split("duration=")[1].split()[0])
    assert 10 <= duration <= 20.5
    types = {sh["type"] for sh in done.result["details"]["shots"]}
    assert types == {"IMAGE", "VIDEO"}
    assert done.result["details"]["bpm"] and abs(done.result["details"]["bpm"] - 100) < 4
