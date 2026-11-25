import asyncio
import json
from datetime import datetime, timezone

from app.immich import ImmichGateway
from app.jobs import JobStore
from app.pipelines import JobInputError, OutputFile, PipelineResult
from app.settings_store import SettingsStore
from app.worker import Worker
from fake_immich import ALBUM_ID, DEPTH_ID, SOURCE_ID, UPLOAD_ID, FakeImmich, with_fake

TAKEN = datetime(2023, 5, 1, 10, 0, tzinfo=timezone.utc)


async def fake_cinematic(ctx):
    """Stands in for a pipeline: returns an MP4 and a depth PNG."""
    ctx.progress("Render", 50)
    ctx.log("render 60/120 frames")
    ctx.check_cancel()
    return PipelineResult(
        status="done",
        files=[OutputFile("c.mp4", b"mp4", TAKEN, duration_ms=4000), OutputFile("c-depth.png", b"png", TAKEN, role="depth")],
        source_ids=[SOURCE_ID],
        year=2023,
        details={"preset": "dolly"},
    )


async def flat_photo(ctx):
    return PipelineResult(status="skipped", message="Not eligible: flat scene (depth spread 0.08, minimum 0.20).")


async def boom(ctx):
    raise ZeroDivisionError


async def bad_input(ctx):
    raise JobInputError("A collage needs at least 2 visible photos; found 1.")


def run_one(config, db, registry, fake=None):
    fake = fake or FakeImmich()

    async def go(cfg):
        async with ImmichGateway(cfg) as gw:
            worker = Worker(JobStore(db), SettingsStore(db), gw, cfg, registry=registry)
            return await worker.run_once()

    return asyncio.run(with_fake(fake, config, go)), fake


def delivering_fake():
    """Two uploads (main + depth) and the full write-back."""
    fake = FakeImmich().write_back()
    ids = iter([UPLOAD_ID, DEPTH_ID])
    from fake_immich import Reply, TS

    fake.on_match("POST", lambda r: r.path == "/api/assets", lambda r: Reply(201, {"id": next(ids), "status": "created"}))
    fake.on_match("PUT", lambda r: r.path.endswith("/metadata"), Reply(200, [{"key": "photos.companion", "updatedAt": TS, "value": {}}]))
    return fake


PERSON = "99999999-0000-4000-8000-000000000000"
BIRTHDAY = {"title": "Atharv's 9th birthday", "kind": "birthday", "person_id": PERSON, "person_name": "Atharv"}


def test_done_job_uploads_then_writes_back(config, db):
    job = JobStore(db).enqueue("cinematic", "IMG_2041", {"asset_ids": [SOURCE_ID], "event": BIRTHDAY})
    ran, fake = run_one(config, db, {"cinematic": fake_cinematic}, delivering_fake())
    assert ran
    done = JobStore(db).get(job.id, with_log=True)
    assert done.status == "done", done.message
    assert [o["role"] for o in done.result["outputs"]] == ["main", "depth"]
    assert done.result["album_id"] == ALBUM_ID and done.result["memory_id"]
    uploads = fake.calls("POST", "/assets")
    assert [u.form["filename"] for u in uploads] == ["c.mp4", "c-depth.png"]
    assert uploads[0].form["duration"] == "4000"
    assert fake.calls("PUT", f"/albums/{ALBUM_ID}/assets")[0].json["ids"] == [UPLOAD_ID, DEPTH_ID]
    assert fake.calls("PUT", "/tags")[0].json["tags"] == ["creation:cinematic"]
    assert fake.calls("POST", "/stacks")[0].json["assetIds"] == [SOURCE_ID, UPLOAD_ID, DEPTH_ID]
    memory = fake.calls("POST", "/memories")[0].json
    assert memory["assetIds"] == [UPLOAD_ID]  # the depth map stays out of the Memories strip
    assert memory["type"] == "birthday" and memory["data"]["personName"] == "Atharv"
    assert memory["data"]["year"] == 2023
    # The event also gets its own named album, and the creation a description.
    assert any(c.json and c.json.get("albumName") == "Atharv's 9th birthday" for c in fake.calls("POST", "/albums")) or \
        fake.calls("GET", "/albums")[-1].query.get("name") == "Atharv's 9th birthday"
    assert fake.calls("PUT", f"/assets/{UPLOAD_ID}")[0].json == {"description": "Atharv's 9th birthday"}
    assert "render 60/120 frames" in done.log


def test_non_birthday_events_get_a_named_album_not_a_memory(config, db):
    event = {"title": "Deepavali 2026", "kind": "festival"}
    JobStore(db).enqueue("cinematic", "IMG", {"asset_ids": [SOURCE_ID], "event": event})
    _, fake = run_one(config, db, {"cinematic": fake_cinematic}, delivering_fake())
    assert fake.calls("POST", "/memories") == []
    assert [c.query.get("name") for c in fake.calls("GET", "/albums")] == ["Creations", "Deepavali 2026"]


def test_manual_job_without_event_gets_no_memory(config, db):
    JobStore(db).enqueue("cinematic", "IMG", {"asset_ids": [SOURCE_ID]})
    _, fake = run_one(config, db, {"cinematic": fake_cinematic}, delivering_fake())
    assert fake.calls("POST", "/memories") == [] and len(fake.calls("GET", "/albums")) == 1


def test_folder_mode_never_calls_immich(config, db, tmp_path):
    cfg = config.model_copy(update={"photos_output_mode": "folder", "photos_output_dir": tmp_path / "out"})
    job = JobStore(db).enqueue("cinematic", "IMG", {"asset_ids": [SOURCE_ID]})
    _, fake = run_one(cfg, db, {"cinematic": fake_cinematic})
    done = JobStore(db).get(job.id)
    assert done.status == "done", done.message
    assert fake.requests == []
    folder = tmp_path / "out" / f"{job.id:05d}-cinematic"
    assert (folder / "c.mp4").read_bytes() == b"mp4"
    assert (folder / "c-depth.png").read_bytes() == b"png"
    summary = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    assert summary["sources"] == [SOURCE_ID] and summary["details"] == {"preset": "dolly"}
    assert done.result["folder"] == str(folder)


def test_settings_can_turn_off_memories_and_stack(config, db):
    store = SettingsStore(db)
    s = store.get()
    s.output.add_to_memories = False
    s.output.stack_with_source = False
    store.save(s)
    JobStore(db).enqueue("cinematic", "IMG", {"asset_ids": [SOURCE_ID]})
    _, fake = run_one(config, db, {"cinematic": fake_cinematic}, delivering_fake())
    assert fake.calls("POST", "/stacks") == [] and fake.calls("POST", "/memories") == []
    assert fake.calls("PUT", f"/albums/{ALBUM_ID}/assets")  # album still used


def test_skipped_job_uploads_nothing(config, db):
    job = JobStore(db).enqueue("cinematic", "flat", {"asset_ids": [SOURCE_ID]})
    _, fake = run_one(config, db, {"cinematic": flat_photo})
    got = JobStore(db).get(job.id)
    assert got.status == "skipped" and "flat scene" in got.message
    assert fake.requests == []


def test_input_error_message_is_shown(config, db):
    job = JobStore(db).enqueue("collage", "A", {})
    run_one(config, db, {"collage": bad_input})
    got = JobStore(db).get(job.id)
    assert got.status == "failed" and got.message.startswith("A collage needs at least 2")


def test_unexpected_error_does_not_leak_details(config, db):
    job = JobStore(db).enqueue("highlight", "A", {})
    run_one(config, db, {"highlight": boom})
    got = JobStore(db).get(job.id)
    assert got.status == "failed" and got.message == "Unexpected error: ZeroDivisionError. See the service log."


def test_cancel_requested_while_running(config, db):
    store = JobStore(db)
    job = store.enqueue("cinematic", "IMG", {})

    async def cancels_itself(ctx):
        store.cancel(ctx.job.id)  # the user presses Cancel mid-run
        ctx.check_cancel()
        raise AssertionError("should have stopped")

    run_one(config, db, {"cinematic": cancels_itself})
    assert store.get(job.id).status == "cancelled"


def test_missing_permission_during_write_back_fails_job(config, db):
    job = JobStore(db).enqueue("cinematic", "IMG", {"asset_ids": [SOURCE_ID]})
    fake = delivering_fake()
    fake.on("GET", "/albums", {"message": "Missing required permission: album.read"}, status=403)
    run_one(config, db, {"cinematic": fake_cinematic}, fake)
    got = JobStore(db).get(job.id)
    assert got.status == "failed" and "lacks a permission" in got.message


def test_empty_queue_returns_false(config, db):
    ran, _ = run_one(config, db, None)
    assert ran is False
