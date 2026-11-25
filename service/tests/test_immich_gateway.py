import asyncio
import json
from datetime import datetime, timezone

import pytest

from app.immich import ImmichError, ImmichGateway
from app.permissions import ALL_REQUIRED
from fake_immich import ALBUM_ID, SOURCE_ID, UPLOAD_ID, FakeImmich, album, api_key, version, with_fake


def run_gw(fake, config, fn):
    async def go(cfg):
        async with ImmichGateway(cfg) as gw:
            return await fn(gw)

    return asyncio.run(with_fake(fake, config, go))


def test_status_reports_version(config):
    fake = FakeImmich().on("GET", "/server/ping", {"res": "pong"}).on("GET", "/server/version", version())
    status = run_gw(fake, config, lambda gw: gw.status())
    assert status.reachable and status.version == "v3.4.2"


def test_api_key_sent_on_authenticated_calls(config):
    fake = FakeImmich().on("GET", "/api-keys/me", api_key(["all"]))
    run_gw(fake, config, lambda gw: gw.permission_report())
    assert fake.calls("GET", "/api-keys/me")[0].headers["x-api-key"] == "test-key"


def test_status_when_unreachable(config):
    cfg = config.model_copy(update={"immich_url": "http://127.0.0.1:9", "photos_request_timeout": 2.0})

    async def go():
        async with ImmichGateway(cfg) as gw:
            return await gw.status()

    status = asyncio.run(go())
    assert not status.reachable
    assert "Can't reach Immich" in status.error


def test_permission_report_lists_missing(config):
    granted = [p for p in ALL_REQUIRED if p != "memoryAsset.create"]
    fake = FakeImmich().on("GET", "/api-keys/me", api_key(granted))
    report = run_gw(fake, config, lambda gw: gw.permission_report())
    assert report.key_name == "photos"
    assert report.missing == ("memoryAsset.create",)


def test_permission_report_all(config):
    fake = FakeImmich().on("GET", "/api-keys/me", api_key(["all"]))
    assert run_gw(fake, config, lambda gw: gw.permission_report()).missing == ()


def test_unauthorised_key_gives_readable_error(config):
    fake = FakeImmich().on("GET", "/api-keys/me", {"message": "Invalid API key"}, status=401)
    with pytest.raises(ImmichError, match="rejected the API key"):
        run_gw(fake, config, lambda gw: gw.permission_report())


def test_ensure_album_reuses_existing(config):
    fake = FakeImmich().on("GET", "/albums", [album()])
    assert run_gw(fake, config, lambda gw: gw.ensure_album("Creations")) == ALBUM_ID
    assert fake.calls("GET", "/albums")[0].query == {"name": "Creations"}
    assert fake.calls("POST", "/albums") == []


def test_ensure_album_creates_when_missing(config):
    fake = FakeImmich().on("GET", "/albums", []).on("POST", "/albums", album(), status=201)
    assert run_gw(fake, config, lambda gw: gw.ensure_album("Creations")) == ALBUM_ID
    assert fake.calls("POST", "/albums")[0].json["albumName"] == "Creations"


def test_upload_sends_file_dates_and_metadata(config):
    async def go(gw):
        return await gw.upload(
            b"\xff\xd8fakejpeg", "collage.jpg",
            taken_at=datetime(2026, 11, 1, tzinfo=timezone.utc), job_id=7, kind="collage", source_ids=[SOURCE_ID],
        )

    fake = FakeImmich().uploads()
    result = run_gw(fake, config, go)
    assert result.asset_id == UPLOAD_ID and not result.duplicate
    form = fake.calls("POST", "/assets")[0].form
    assert form["assetData"] == "<file:collage.jpg>"
    assert form["fileCreatedAt"].startswith("2026-11-01")
    assert form["filename"] == "collage.jpg"
    assert "metadata" not in form  # sent as JSON below, not in the multipart body
    meta = fake.calls("PUT", f"/assets/{UPLOAD_ID}/metadata")[0].json
    assert meta == {"items": [{"key": "photos.companion", "value": {"job": 7, "kind": "collage", "sources": [SOURCE_ID]}}]}


def test_upload_duplicate_skips_metadata(config):
    async def go(gw):
        return await gw.upload(b"x", "a.jpg", taken_at=datetime.now(timezone.utc), job_id=1, kind="collage", source_ids=[])

    fake = FakeImmich().uploads(status="duplicate")
    assert run_gw(fake, config, go).duplicate
    assert fake.calls("PUT", f"/assets/{UPLOAD_ID}/metadata") == []


def test_memory_is_a_birthday_memory_with_the_person(config):
    pid = "99999999-0000-4000-8000-000000000000"
    fake = FakeImmich().write_back()
    run_gw(fake, config, lambda gw: gw.add_memory([UPLOAD_ID], year=2024, person_id=pid, person_name="Atharv"))
    body = fake.calls("POST", "/memories")[0].json
    assert body["type"] == "birthday"
    assert body["data"] == {"year": 2024, "personId": pid, "personName": "Atharv"}
    assert body["assetIds"] == [UPLOAD_ID]
    assert body["hideAt"] > body["showAt"]


def test_stack_puts_source_first(config):
    fake = FakeImmich().write_back()
    run_gw(fake, config, lambda gw: gw.stack(SOURCE_ID, [UPLOAD_ID]))
    assert fake.calls("POST", "/stacks")[0].json["assetIds"] == [SOURCE_ID, UPLOAD_ID]


def test_tag_upserts_then_tags_assets(config):
    fake = FakeImmich().write_back()
    run_gw(fake, config, lambda gw: gw.tag("creation:cinematic", [UPLOAD_ID]))
    assert fake.calls("PUT", "/tags")[0].json == {"tags": ["creation:cinematic"]}
    assert fake.calls("PUT", "/tags/assets")[0].json["assetIds"] == [UPLOAD_ID]
