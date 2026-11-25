import asyncio
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.db import Database
from app.imaging import popout as fx
from app.imaging.matting import clean, fill_holes, keep_people, smooth_silhouette, trim_above_heads
from app.imaging.titles import pick_title_style, title_frames
from app.immich import FaceBox, ImmichGateway
from app.jobs import JobStore
from app.settings_store import SettingsStore
from app.worker import Worker
from fake_immich import FakeImmich, asset, face, with_fake

W, H = 400, 300
FACE = FaceBox(0.45, 0.20, 0.55, 0.36)


def person_scene():
    """A green 'forest' with a person-like blob: head circle + body rectangle, plus a pole above
    the head and a stray leaf far away (both should be cleaned out)."""
    img = Image.new("RGB", (W, H), (40, 110, 50))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 25):
        d.rectangle((x, 0, x + 6, H), fill=(90, 70, 40))  # tree trunks
    raw = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    head = ((xx - 200) ** 2 + (yy - 85) ** 2) < 26**2
    body = (abs(xx - 200) < 45) & (yy > 105) & (yy < 300)
    hole = (abs(xx - 200) < 6) & (abs(yy - 200) < 6)  # a gap the model missed
    pole = (abs(xx - 200) < 3) & (yy < 60)
    leaf = ((xx - 40) ** 2 + (yy - 40) ** 2) < 12**2
    raw[head | body | pole | leaf] = 1.0
    raw[hole] = 0.0
    d.ellipse((174, 59, 226, 111), fill=(200, 160, 130))
    d.rectangle((155, 105, 245, 300), fill=(180, 40, 60))
    return img, raw


def test_clean_keeps_the_person_and_drops_junk():
    img, raw = person_scene()
    m = clean(raw, img, [FACE])
    assert m.solid[200, 200]  # hole filled
    assert not m.solid[40, 40]  # stray leaf dropped (no face)
    assert not m.solid[20, 200]  # pole above the head trimmed
    assert m.solid[90, 200] and m.solid[250, 170]  # head and body kept
    assert m.alpha.shape == (H, W) and 0 <= m.alpha.min() and m.alpha.max() <= 1
    assert m.foreground.shape == (H, W, 3)


def test_helpers():
    ring = np.zeros((20, 20), bool)
    ring[5:15, 5:15] = True
    ring[8:12, 8:12] = False
    assert fill_holes(ring)[10, 10]
    two = np.zeros((20, 40), bool)
    two[5:15, 2:10] = True
    two[5:15, 30:38] = True
    kept = keep_people(two, [FaceBox(0.05, 0.25, 0.25, 0.5)])
    assert kept[10, 5] and not kept[10, 33]
    tall = np.ones((100, 100), bool)
    trimmed = trim_above_heads(tall, [FaceBox(0.4, 0.4, 0.6, 0.6)], 100, 100)
    assert not trimmed[0, 50] and trimmed[50, 50] and trimmed[0, 5]  # only above the head's columns
    # a pole standing alone above the head goes; a shoulder that carries on sideways stays
    pole = np.zeros((100, 100), bool)
    pole[40:100, 40:60] = True  # the person
    pole[0:40, 48:52] = True  # pole out of the head
    assert not trim_above_heads(pole, [FaceBox(0.4, 0.4, 0.6, 0.6)], 100, 100)[5, 50]
    shoulder = np.zeros((100, 100), bool)
    shoulder[60:100, 40:60] = True  # a child, face at rows 60-70
    shoulder[30:100, 0:100] = True  # an adult's shoulders behind, wider than the child's band
    shoulder[8:30, 10:20] = True  # the adult's head
    faces = [FaceBox(0.45, 0.6, 0.55, 0.7), FaceBox(0.1, 0.1, 0.2, 0.2)]
    out = trim_above_heads(shoulder, faces, 100, 100)
    assert out[35, 50] and out[35, 5] and out[15, 15]
    s = smooth_silhouette(ring, 8, 3)
    assert s.max() == pytest.approx(1.0, abs=0.01) and s[10, 10] > 0.9


@pytest.mark.parametrize("style", ["splash", "spotlight", "blur", "frame", "sticker"])
def test_every_style_renders(style):
    img, raw = person_scene()
    m = clean(raw, img, [FACE])
    depth = np.tile(np.linspace(0.2, 0.9, H, dtype=np.float32)[:, None], (1, W))
    out = {
        "splash": lambda: fx.splash(img, m),
        "spotlight": lambda: fx.spotlight(img, m),
        "blur": lambda: fx.portrait_blur(img, m, depth, 0.8),
        "frame": lambda: fx.pop_frame(img, m, [FACE], caption="Goa 2026"),
        "sticker": lambda: fx.sticker(img, m),
    }[style]()
    assert out.mode == "RGB" and min(out.size) >= 300
    a = np.asarray(out, np.float32)
    assert a.std() > 5  # not a flat image


def test_splash_keeps_person_colour_and_greys_background():
    img, raw = person_scene()
    m = clean(raw, img, [FACE])
    out = np.asarray(fx.splash(img, m)).astype(int)
    bg, person = out[150, 30], out[250, 200]
    assert abs(bg[0] - bg[1]) <= 2 and abs(bg[1] - bg[2]) <= 2  # grey
    assert person[0] - person[1] > 80  # still red


def test_blur_leaves_person_sharp_and_blurs_background():
    img, raw = person_scene()
    m = clean(raw, img, [FACE])
    depth = np.full((H, W), 0.2, np.float32)
    depth[m.solid] = 0.9
    out = np.asarray(fx.portrait_blur(img, m, depth, 0.9), np.float32)
    src = np.asarray(img, np.float32)
    # Trunk edges in the background get soft; the person's middle is unchanged.
    edge = lambda a: np.abs(np.diff(a[150, :120, 0])).max()  # noqa: E731
    assert edge(out) < edge(src) * 0.6
    assert np.abs(out[220, 190] - src[220, 190]).max() < 6


def test_splash_video_frames():
    img, raw = person_scene()
    m = clean(raw, img, [FACE])
    frames = list(fx.splash_frames(img, m, 1.0, 10, (320, 180)))
    assert len(frames) == 10 and frames[0].shape == (180, 320, 3)
    sat = lambda f: (f.max(-1).astype(int) - f.min(-1)).mean()  # noqa: E731
    assert sat(frames[-1]) > sat(frames[0]) + 3  # colour arrives


def test_title_card_frames():
    img, _ = person_scene()
    for style in ("fade-up", "typewriter", "letters"):
        frames = list(title_frames(img, "Yercaud trip", "13\u201315 Sep 2026", (320, 180), 1.0, 10, style))
        assert len(frames) == 10 and frames[0].shape == (180, 320, 3)
        assert frames[-1].max() > frames[0].max() or frames[-1].mean() > frames[0].mean()  # text appears
    assert pick_title_style("random", 7) in ("fade-up", "typewriter", "letters")
    assert pick_title_style("random", 7) == pick_title_style("random", 7)


class FakeMatter:
    def matte(self, img, faces):
        raw = np.zeros((img.height, img.width), np.float32)
        raw[img.height // 4 :, img.width // 3 : 2 * img.width // 3] = 1
        return clean(raw, img, faces)


def test_popout_pipeline_end_to_end(config, db, tmp_path, monkeypatch):
    from app.pipelines import popout as pipe

    monkeypatch.setattr(pipe, "_matter", lambda path: FakeMatter())
    monkeypatch.setattr(pipe.models, "ensure", lambda *a, **k: tmp_path / "m.onnx")
    cfg = config.model_copy(update={"photos_output_mode": "folder", "photos_output_dir": tmp_path / "out"})
    aid = "00000001-0000-4000-8000-000000000000"
    img, _ = person_scene()
    buf = io.BytesIO()
    img.resize((1200, 900)).save(buf, "JPEG")
    fake = FakeImmich().library([asset(aid, "2026-09-15T11:03:27.000Z", "IMG_1.jpg", 1200, 900, people=[])], {aid: buf.getvalue()},
                                {aid: [face(1200, 900, (540, 180, 660, 324))]})
    job = JobStore(db).enqueue("popout", "IMG_1", {"asset_ids": [aid], "styles": ["splash", "frame", "sticker"], "caption": "Goa"})

    async def go(c):
        async with ImmichGateway(c) as gw:
            await Worker(JobStore(db), SettingsStore(db), gw, c, registry={"popout": pipe.run_popout}).run_once()

    asyncio.run(with_fake(fake, cfg, go))
    done = JobStore(db).get(job.id, with_log=True)
    assert done.status == "done", (done.message, done.log)
    folder = tmp_path / "out" / f"{job.id:05d}-popout"
    names = sorted(p.name for p in folder.glob("*.jpg"))
    assert names == [f"popout-{job.id}-frame.jpg", f"popout-{job.id}-splash.jpg", f"popout-{job.id}-sticker.jpg"]
    assert json.loads((folder / "job.json").read_text(encoding="utf-8"))["details"]["caption"] == "Goa"


def test_housekeeping_removes_old_jobs_and_their_folders(config, db, tmp_path):
    cfg = config.model_copy(update={"photos_output_mode": "folder", "photos_output_dir": tmp_path / "out"})
    store = JobStore(db)
    old = store.enqueue("collage", "old", {})
    store.claim_next()
    folder = tmp_path / "out" / "00001-collage"
    folder.mkdir(parents=True)
    (folder / "a.jpg").write_bytes(b"x")
    store.finish(old.id, "done", result={"folder": str(folder)})
    store.log(old.id, "line")
    keep = store.enqueue("collage", "waiting", {})
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    db.execute("UPDATE jobs SET finished_at = ? WHERE id = ?", ((datetime.now(timezone.utc) - timedelta(days=40)).isoformat(), old.id))
    worker = Worker(store, SettingsStore(db), None, cfg)
    res = worker.housekeep()
    assert res["jobs"] == 1 and res["files"] == 1
    assert store.get(old.id) is None and not folder.exists()
    assert store.get(keep.id) is not None  # queued jobs are never removed
    assert outside.exists()
    # retention 0 = keep everything
    s = SettingsStore(db).get()
    s.housekeeping.retention_days = 0
    SettingsStore(db).save(s)
    assert worker.housekeep()["jobs"] == 0


def test_migration_from_an_old_database(tmp_path):
    p = tmp_path / "old.sqlite3"
    c = sqlite3.connect(p)
    c.executescript(
        """
        CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT,
          kind TEXT NOT NULL CHECK (kind IN ('collage', 'cinematic', 'highlight')),
          status TEXT NOT NULL, origin TEXT NOT NULL, title TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}',
          position REAL NOT NULL, progress INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL DEFAULT '',
          message TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '{}', cancel_requested INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
        CREATE TABLE job_log (id INTEGER PRIMARY KEY AUTOINCREMENT,
          job_id INTEGER NOT NULL REFERENCES jobs (id) ON DELETE CASCADE, at TEXT NOT NULL, line TEXT NOT NULL);
        INSERT INTO jobs (kind, status, origin, title, position, created_at) VALUES ('collage', 'done', 'manual', 'old one', 1, '2026-11-01T00:00:00+00:00');
        INSERT INTO job_log (job_id, at, line) VALUES (1, '2026-11-01T00:00:00+00:00', 'old line');
        """
    )
    c.close()
    db = Database(p)
    store = JobStore(db)
    assert store.get(1).title == "old one" and store.get(1).superseded_by is None
    new = store.enqueue("popout", "new kind", {})
    assert new.kind == "popout"
    store.log(new.id, "logging still works after the rebuild")
    assert [entry for entry in store.get(1, with_log=True).log if "old line" in str(entry)]
    assert "jobs_old" not in db.one("SELECT sql FROM sqlite_master WHERE name = 'job_log'")[0]
    db.close()


def test_repairs_a_job_log_left_pointing_at_jobs_old(tmp_path):
    """An early migration renamed `jobs` away, so SQLite re-pointed job_log at the dropped table."""
    p = tmp_path / "broken.sqlite3"
    db = Database(p)
    store = JobStore(db)
    job = store.enqueue("collage", "kept", {})
    store.log(job.id, "kept line")
    db.close()
    c = sqlite3.connect(p)
    c.executescript(
        """
        PRAGMA foreign_keys = OFF;
        ALTER TABLE job_log RENAME TO job_log_tmp;
        CREATE TABLE job_log (id INTEGER PRIMARY KEY AUTOINCREMENT,
          job_id INTEGER NOT NULL REFERENCES "jobs_old" (id) ON DELETE CASCADE, at TEXT NOT NULL, line TEXT NOT NULL);
        INSERT INTO job_log SELECT * FROM job_log_tmp;
        DROP TABLE job_log_tmp;
        """
    )
    c.close()
    db = Database(p)
    store = JobStore(db)
    store.log(job.id, "works again")
    lines = [str(entry) for entry in store.get(job.id, with_log=True).log]
    assert any("kept line" in line for line in lines) and any("works again" in line for line in lines)
    db.close()
