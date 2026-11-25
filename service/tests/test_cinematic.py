import asyncio
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.imaging import depth as depth_mod
from app.imaging.depth import depth_png, estimate
from app.imaging.parallax import Motion, choose_motion, frames, output_size, stretch_score
from app.immich import FaceBox, ImmichGateway
from app.jobs import JobStore
from app.settings_store import SettingsStore
from app.worker import Worker
from fake_immich import FakeImmich, asset, face, with_fake

FFMPEG = os.environ.get("PHOTOS_FFMPEG") or shutil.which("ffmpeg")


class FakeDepthModel:
    """Returns a fixed 'inverse depth' map: a near blob in the middle of a far background."""

    def __init__(self, flat: bool = False):
        self.flat = flat

    def raw(self, img):
        h, w = 280, 378
        if self.flat:
            return np.full((h, w), 5.0, np.float32) + np.random.default_rng(0).normal(0, 0.01, (h, w)).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = ((xx - w / 2) ** 2 / (w / 5) ** 2 + (yy - h / 2) ** 2 / (h / 3) ** 2) < 1
        return np.where(blob, 10.0, 2.0).astype(np.float32) + yy.astype(np.float32) / h


def test_output_size_follows_orientation():
    assert output_size(4000, 3000, "1080p") == (1920, 1080)
    assert output_size(3000, 4000, "1080p") == (1080, 1920)
    assert output_size(3000, 3000, "720p") == (720, 720)


def test_estimate_normalises_and_reports_spread():
    img = Image.new("RGB", (400, 300))
    res = estimate(FakeDepthModel(), img)
    assert res.depth.shape == (300, 400)
    assert 0 <= res.depth.min() and res.depth.max() <= 1
    assert res.depth[150, 200] > 0.9 and res.depth[10, 10] < 0.2  # near blob vs background
    assert res.spread > 0.5
    flat = estimate(FakeDepthModel(flat=True), img)
    assert flat.spread < 0.05


def test_faces_pull_people_forward_without_boxes():
    img = Image.new("RGB", (400, 300))
    f = FaceBox(0.05, 0.10, 0.12, 0.20)  # a face out in the "background"
    plain = estimate(FakeDepthModel(), img).depth
    pulled = estimate(FakeDepthModel(), img, [f]).depth
    # Background far from the face is unchanged.
    assert abs(float(pulled[280, 390]) - float(plain[280, 390])) < 0.02


def test_depth_png_is_16_bit():
    png = depth_png(np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10))
    buf = io.BytesIO()
    png.save(buf, "PNG")
    assert Image.open(io.BytesIO(buf.getvalue())).mode in ("I;16", "I")


def test_flat_depth_has_no_stretch_and_steps_do():
    # Realistic size: a 1440-wide photo rendered to 1280x720.
    flat = np.full((1080, 1440), 0.5, np.float32)
    step = flat.copy()
    step[:, 720:] = 1.0
    m = Motion("dolly", 0.04, 1.13)
    assert stretch_score(flat, (1280, 720), 0.5, m, None) == 0
    assert stretch_score(step, (1280, 720), 0.5, m, None) > 0


def test_auto_picks_a_preset_and_keeps_amplitude_in_range():
    d = estimate(FakeDepthModel(), Image.new("RGB", (400, 300))).depth
    m, scores = choose_motion(d, (480, 270), 0.8, "auto", None)
    assert m.preset in scores and set(scores) == {"zoom", "dolly", "circle"}
    assert 0.03 <= m.amplitude <= 0.065


def test_frames_have_output_size_and_move():
    img = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    d = estimate(FakeDepthModel(), Image.fromarray(img)).depth
    out = list(frames(img, d, (192, 108), 0.8, Motion("dolly", 0.04, 1.13), 5))
    assert len(out) == 5 and out[0].shape == (108, 192, 3)
    assert np.abs(out[0].astype(int) - out[-1].astype(int)).mean() > 1  # the camera moved


def _run(config, db, tmp_path, monkeypatch, flat=False, params=None):
    from app.pipelines import cinematic

    monkeypatch.setattr(cinematic, "_model", lambda path: FakeDepthModel(flat))
    monkeypatch.setattr(cinematic.models, "ensure", lambda *a, **k: tmp_path / "model.onnx")
    cfg = config.model_copy(update={
        "photos_output_mode": "folder", "photos_output_dir": tmp_path / "out", "photos_ffmpeg": FFMPEG or "ffmpeg",
    })
    aid = "00000001-0000-4000-8000-000000000000"
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(2).integers(0, 255, (1080, 1440, 3), dtype=np.uint8)).save(buf, "JPEG")
    fake = FakeImmich().library([asset(aid, "2026-09-15T11:03:27.000Z", "IMG_1.jpg", 1440, 1080)], {aid: buf.getvalue()},
                                {aid: [face(1440, 1080, (650, 300, 790, 470))]})
    store = SettingsStore(db)
    s = store.get()
    s.cinematic.seconds, s.cinematic.resolution = 2, "720p"
    store.save(s)
    job = JobStore(db).enqueue("cinematic", "IMG_1", params or {"asset_ids": [aid]})

    async def go(c):
        async with ImmichGateway(c) as gw:
            await Worker(JobStore(db), store, gw, c, registry={"cinematic": cinematic.run_cinematic}).run_once()

    asyncio.run(with_fake(fake, cfg, go))
    return JobStore(db).get(job.id, with_log=True), tmp_path / "out" / f"{job.id:05d}-cinematic"


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")
def test_cinematic_pipeline_makes_mp4_and_depth_png(config, db, tmp_path, monkeypatch):
    done, folder = _run(config, db, tmp_path, monkeypatch)
    assert done.status == "done", (done.message, done.log)
    mp4 = folder / f"cinematic-{done.id}.mp4"
    assert mp4.stat().st_size > 1000 and mp4.read_bytes()[4:8] == b"ftyp"
    png = Image.open(folder / f"cinematic-{done.id}-depth.png")
    assert png.size == (1440, 1080)
    assert done.result["details"]["preset"] in ("zoom", "dolly", "circle")


def test_flat_photo_is_skipped(config, db, tmp_path, monkeypatch):
    done, folder = _run(config, db, tmp_path, monkeypatch, flat=True)
    assert done.status == "skipped" and "flat scene" in done.message
    assert not folder.exists()


def test_two_photos_rejected(config, db, tmp_path, monkeypatch):
    done, _ = _run(config, db, tmp_path, monkeypatch, params={"asset_ids": ["a", "b"]})
    assert done.status == "failed" and "exactly one photo" in done.message
