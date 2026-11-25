import asyncio
import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.imaging.crop import crop_box
from app.imaging.frames import Photo, fan, prints, scatter
from app.imaging.layout import plan
from app.imaging.quality import blurry_flags, colour_signature, dhash, duplicate_groups, scene_groups, sharpness
from app.immich import FaceBox, ImmichGateway
from app.jobs import JobStore
from app.pipelines.collage import date_range
from app.settings_store import SettingsStore
from app.worker import Worker
from fake_immich import ALBUM_ID, FakeImmich, asset, face, with_fake


def photo(seed: int, size=(400, 300), blur: float = 0) -> Image.Image:
    """A busy synthetic photo: random rectangles in a seeded palette."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, tuple(int(c) for c in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        d.rectangle((x, y, x + rng.integers(10, 80), y + rng.integers(10, 80)), fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


def jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ---- layout ------------------------------------------------------------------------------


def _covers_canvas(layout, ratio):
    area = sum(s.w * s.h for s in layout.slots)
    assert area == pytest.approx(1 / ratio, rel=1e-6)
    for s in layout.slots:
        assert s.x >= -1e-9 and s.y >= -1e-9 and s.x + s.w <= 1 + 1e-9 and s.y + s.h <= 1 / ratio + 1e-9


def test_two_portraits_side_by_side_two_landscapes_stacked():
    side = plan([0.75, 0.75], [1, 1], 1.0)
    assert side.slots[0].y == side.slots[1].y  # same row
    stacked = plan([1.5, 1.5], [1, 1], 1.0)
    assert stacked.slots[0].x == stacked.slots[1].x  # same column


@pytest.mark.parametrize("n", range(1, 13))
@pytest.mark.parametrize("ratio", [1.0, 0.8, 16 / 9])
def test_every_count_fills_the_canvas_without_overlap(n, ratio):
    rng = np.random.default_rng(n)
    aspects = list(rng.choice([0.75, 1.0, 1.33, 1.78], n))
    layout = plan(aspects, list(range(n)), ratio)
    assert len(layout.slots) == n
    _covers_canvas(layout, ratio)


def test_hero_goes_to_best_photo():
    layout = plan([1.33, 1.33, 1.33], [0.1, 5.0, 0.2], 1.0)
    if layout.hero is not None:
        assert layout.hero == 1
        assert layout.slots[1].w * layout.slots[1].h == max(s.w * s.h for s in layout.slots)


# ---- crop and quality --------------------------------------------------------------------


def test_crop_keeps_face_inside():
    face_box = FaceBox(0.85, 0.40, 0.95, 0.55)  # far right of a landscape photo
    left, top, right, bottom = crop_box(4000, 3000, 0.75, [face_box])
    assert right - left == pytest.approx(3000 * 0.75, abs=2)
    assert left <= 0.85 * 4000 and right >= 0.95 * 4000


def test_crop_without_faces_is_centred():
    left, _, right, _ = crop_box(4000, 3000, 1.0)
    assert left == 500 and right == 3500


def test_sharpness_and_blur_flags():
    sharp = [sharpness(photo(i)) for i in range(6)]
    soft = sharpness(photo(9, blur=6))
    assert soft < min(sharp) / 3
    flags = blurry_flags(sharp + [soft])
    assert flags[-1] and not any(flags[:-1])


def test_duplicates_and_scenes():
    a = photo(1)
    a2 = a.resize((380, 285)).resize((400, 300))  # near-identical
    b = photo(2)
    h = [dhash(x) for x in (a, a2, b)]
    assert duplicate_groups(h) == [0, 0, 2]
    sigs = [colour_signature(x) for x in (a, a2, b)]
    assert scene_groups([0, 20, 40], h, sigs)[:2] == [0, 0]
    # Same pictures an hour apart are separate scenes.
    assert scene_groups([0, 3600, 7200], h, sigs) == [0, 1, 2]


def test_date_range_formats():
    d = lambda s: datetime.fromisoformat(s)  # noqa: E731
    assert date_range([d("2026-09-13"), d("2026-09-15")]) == "13\u201315 Sep 2026"
    assert date_range([d("2026-08-30"), d("2026-09-02")]) == "30 Aug \u2013 2 Sep 2026"
    assert date_range([d("2026-09-13"), d("2026-09-13T20:00")]) == "13 Sep 2026"


# ---- frame styles ------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 5, 7])
def test_frame_styles_render_at_size(n):
    photos = [Photo(photo(i, (300, 400) if i % 2 else (400, 300)), score=i) for i in range(n)]
    assert scatter(photos, (900, 900), seed=1, caption="Trip \u00b7 13\u201315 Sep").size == (900, 900)
    assert fan(photos[:5], (900, 900), title="Trip", subtitle="13\u201315 Sep").size == (900, 900)
    assert prints(photos, (720, 900), 0.8, seed=1).size == (720, 900)


def test_fan_keeps_outer_cards_on_canvas():
    photos = [Photo(Image.new("RGB", (300, 400), (255, 0, 0)), score=i) for i in range(5)]
    img = np.asarray(fan(photos, (900, 900)))
    red = (img[..., 0] > 200) & (img[..., 1] < 60)
    cols = np.where(red.any(axis=0))[0]
    assert cols.min() > 5 and cols.max() < 894  # no card touches the left/right edge


def test_scatter_is_deterministic():
    photos = [Photo(photo(i), score=i) for i in range(5)]
    a = np.asarray(scatter(photos, (600, 600), seed=7))
    b = np.asarray(scatter(photos, (600, 600), seed=7))
    assert (a == b).all()


# ---- pipeline end to end (fake Immich, folder output) ------------------------------------


def _library(n=10):
    start = datetime(2026, 9, 13, 9, 0, tzinfo=timezone.utc)
    assets, images, faces = [], {}, {}
    for i in range(n):
        aid = f"{i:08d}-0000-4000-8000-000000000000"
        taken = (start + timedelta(hours=2 * i)).isoformat().replace("+00:00", ".000Z")
        assets.append(asset(aid, taken, f"IMG_{i}.jpg"))
        images[aid] = jpeg(photo(i, (400, 300)))
        faces[aid] = [face(400, 300, (150, 60, 210, 130))] if i % 3 == 0 else []
    if n < 7:
        return assets, images, faces
    # One blurry, one screenshot, one near-duplicate of photo 0 taken a minute later.
    images[assets[4]["id"]] = jpeg(photo(4, blur=6))
    assets[5]["originalFileName"] = "Screenshot_2026.png"
    assets[6]["fileCreatedAt"] = assets[6]["localDateTime"] = assets[6]["exifInfo"]["dateTimeOriginal"] = assets[0]["exifInfo"]["dateTimeOriginal"].replace("09:00", "09:01")
    images[assets[6]["id"]] = images[assets[0]["id"]]
    return assets, images, faces


@pytest.mark.parametrize("style", ["grid", "prints", "scatter", "fan"])
def test_collage_pipeline_from_album(config, db, tmp_path, style):
    from app.pipelines.collage import run_collage

    cfg = config.model_copy(update={"photos_output_mode": "folder", "photos_output_dir": tmp_path / "out"})
    assets, images, faces = _library()
    fake = FakeImmich().library(assets, images, faces)
    job = JobStore(db).enqueue("collage", "Trip", {"album_id": ALBUM_ID, "asset_ids": [], "style": style})

    async def go(c):
        async with ImmichGateway(c) as gw:
            await Worker(JobStore(db), SettingsStore(db), gw, c, registry={"collage": run_collage}).run_once()

    asyncio.run(with_fake(fake, cfg, go))
    done = JobStore(db).get(job.id, with_log=True)
    assert done.status == "done", (done.message, done.log)
    assert "left out 1 blurry" in done.message and "1 screenshot" in done.message
    assert "duplicate" in done.message
    used = set(done.result["sources"])
    assert assets[4]["id"] not in used and assets[5]["id"] not in used
    assert not {assets[0]["id"], assets[6]["id"]} <= used  # never both copies
    out = tmp_path / "out" / f"{job.id:05d}-collage" / f"collage-{job.id}-{style}.jpg"
    img = Image.open(out)
    assert max(img.size) == 3000
    summary = json.loads((out.parent / "job.json").read_text(encoding="utf-8"))
    assert summary["details"]["style"] == style
    assert fake.calls("POST", "/assets") == []  # folder mode: nothing uploaded


def test_manual_collage_uses_exactly_the_given_photos(config, db, tmp_path):
    from app.pipelines.collage import run_collage

    cfg = config.model_copy(update={"photos_output_mode": "folder", "photos_output_dir": tmp_path / "out"})
    assets, images, faces = _library(4)
    fake = FakeImmich().library(assets, images, faces)
    ids = [a["id"] for a in assets[:3]]
    job = JobStore(db).enqueue("collage", "Pick", {"asset_ids": ids, "style": "grid"})

    async def go(c):
        async with ImmichGateway(c) as gw:
            await Worker(JobStore(db), SettingsStore(db), gw, c, registry={"collage": run_collage}).run_once()

    asyncio.run(with_fake(fake, cfg, go))
    done = JobStore(db).get(job.id)
    assert done.status == "done", done.message
    assert done.result["sources"] == ids
