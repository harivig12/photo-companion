import pytest
from fastapi.testclient import TestClient

from app.immich import ImmichError, ImmichGateway, PermissionReport, ServerStatus
from app.main import create_app
from app.permissions import ALL_REQUIRED

ORIGIN = "http://testserver"
PASSWORD = "correct horse battery"
U1 = "00000001-0000-4000-8000-000000000000"
U2 = "00000002-0000-4000-8000-000000000000"


class FakeGateway(ImmichGateway):
    """No network: canned answers for the calls the UI makes."""

    reachable = True
    missing: tuple[str, ...] = ()

    async def open(self):
        pass

    async def close(self):
        pass

    async def status(self):
        if self.reachable:
            return ServerStatus(True, "v3.4.2")
        return ServerStatus(False, error="Can't reach Immich. Check IMMICH_URL and that the server is running.")

    async def list_albums(self):
        if not self.reachable:
            raise ImmichError("Can't reach Immich.")
        return [{"id": "11111111-1111-4111-8111-111111111111", "name": "Goa trip", "count": 142}]

    async def permission_report(self):
        granted = tuple(p for p in ALL_REQUIRED if p not in self.missing)
        return PermissionReport("photos", granted, self.missing)

    async def list_people(self):
        return [{"id": "00000009-0000-4000-8000-000000000000", "name": "Asha"}]

    async def thumbnail(self, asset_id, size="preview"):
        return b"\xff\xd8\xff\xe0fake"

    async def search(self, **kw):
        return []

    async def people_with_birthdays(self):
        return []

    async def ocr_hits(self, text, a, b):
        return 0


@pytest.fixture
def client(config):
    FakeGateway.reachable = True
    FakeGateway.missing = ()
    app = create_app(config, gateway_factory=FakeGateway, start_worker=False)
    with TestClient(app, headers={"Origin": ORIGIN}) as c:
        yield c


def login(client):
    r = client.post("/login", data={"password": PASSWORD}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return r


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_pages_require_login(client):
    for path in ("/jobs", "/create", "/suggestions", "/settings"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/api/jobs").status_code == 401


def test_wrong_password(client):
    r = client.post("/login", data={"password": "nope"})
    assert r.status_code == 401 and "Wrong password." in r.text


def test_login_sets_strict_httponly_cookie(client):
    cookie = login(client).headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_cross_site_post_blocked(client):
    r = client.post("/login", data={"password": PASSWORD}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    login(client)
    r = client.post("/create", data={"kind": "collage", "asset_ids": f"{U1}\n{U2}"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_security_headers(client):
    login(client)
    r = client.get("/jobs")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"


def test_create_job_and_see_it_in_queue(client):
    login(client)
    r = client.post("/create", data={"kind": "collage", "album_id": "11111111-1111-4111-8111-111111111111", "title": "Goa collage"}, follow_redirects=False)
    assert r.status_code == 303
    page = client.get("/jobs").text
    assert "Goa collage" in page and "Queued" in page
    data = client.get("/api/jobs").json()
    assert data["counts"]["queued"] == 1 and data["running"] is None
    assert data["queued"][0]["title"] == "Goa collage"


def test_create_validation(client):
    login(client)
    r = client.post("/create", data={"kind": "collage"})
    assert r.status_code == 422 and "Pick an album, a person or dates, or paste asset IDs." in r.text
    r = client.post("/create", data={"kind": "cinematic", "asset_ids": f"{U1}\n{U2}"})
    assert r.status_code == 422 and "exactly one photo" in r.text


def test_queue_actions(client):
    login(client)
    for t in ("A", "B"):
        client.post("/create", data={"kind": "collage", "asset_ids": f"{U1}\n{U2}", "title": t})
    ids = [j["id"] for j in client.get("/api/jobs").json()["queued"]]
    client.post(f"/jobs/{ids[1]}/up")
    assert [j["title"] for j in client.get("/api/jobs").json()["queued"]] == ["B", "A"]
    client.post(f"/jobs/{ids[0]}/cancel")
    assert [j["title"] for j in client.get("/api/jobs").json()["queued"]] == ["B"]


def test_settings_save_and_errors(client):
    login(client)
    page = client.get("/settings").text
    assert "Suggest, I approve" in page
    form = {
        "suggestions.mode": "auto", "suggestions.scan_at": "04:30", "suggestions.event_gap_hours": "4",
        "suggestions.min_good_photos": "6", "suggestions.max_per_day": "3",
        "output.concurrency": "1", "output.album_name": "My creations", "output.tag_prefix": "creation:",
        "output.add_to_memories": "on",
        "collage.canvas_ratio": "4:5", "collage.max_photos_manual": "9", "collage.max_photos_auto": "12",
        "collage.border_px": "6", "collage.radius_px": "10", "collage.border_colour": "#112233", "collage.skip_blurry": "on",
        "cinematic.motion": "dolly", "cinematic.seconds": "5", "cinematic.fps": "24", "cinematic.resolution": "720p",
        "cinematic.min_depth_spread": "0.25",
        "highlight.min_seconds": "30", "highlight.max_seconds": "60", "highlight.aspect": "9:16", "highlight.seconds_per_photo": "2.5",
    }
    r = client.post("/settings", data=form)
    assert r.status_code == 200 and "Saved." in r.text
    assert 'value="My creations"' in r.text
    # Unchecked boxes are saved as off.
    assert 'name="output.stack_with_source" checked' not in r.text

    bad = dict(form, **{"collage.border_colour": "white"})
    r = client.post("/settings", data=bad)
    assert r.status_code == 422 and "Not saved." in r.text and "Use a colour like #FFFFFF" in r.text
    assert 'value="My creations"' in client.get("/settings").text  # previous values kept


def test_permission_check_shows_missing(client):
    login(client)
    FakeGateway.missing = ("memoryAsset.create",)
    r = client.post("/settings/permissions")
    assert "memoryAsset.create missing" in r.text and "memories strip" in r.text


def test_offline_banner_and_create_page(client):
    FakeGateway.reachable = False
    login(client)
    r = client.get("/create")
    assert r.status_code == 200
    assert "Can't reach Immich" in r.text and "Albums unavailable" in r.text


def test_password_change_signs_out(client):
    login(client)
    r = client.post("/settings/password", data={"new_password": "short"})
    assert r.status_code == 422 and "at least 12" in r.text
    r = client.post("/settings/password", data={"new_password": "an even longer password"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/jobs", follow_redirects=False).status_code == 303
    assert client.post("/login", data={"password": "an even longer password"}, follow_redirects=False).status_code == 303


def test_jobs_page_shows_live_job_and_by_type(client):
    login(client)
    state = client.app.state.s
    job = state.jobs.enqueue("cinematic", "IMG_2041", {"asset_ids": ["x"]})
    state.jobs.claim_next()
    state.jobs.progress(job.id, "Render", 62)
    state.jobs.log(job.id, "render 74/120 frames")
    page = client.get("/jobs").text
    assert "Live: Cinematic photo · IMG_2041" in page
    assert 'value="62"' in page and "render 74/120 frames" in page
    assert "Worker busy: 1 running" in page
    data = client.get("/api/jobs").json()
    assert data["running"]["progress"] == 62 and data["running"]["stage"] == "Render"


def test_suggestions_page_empty_state(client):
    login(client)
    page = client.get("/suggestions").text
    assert "No ideas yet" in page and "Suggest, I approve" in page


def test_filter_by_kind(client):
    login(client)
    client.post("/create", data={"kind": "collage", "asset_ids": f"{U1}\n{U2}", "title": "C1"})
    client.post("/create", data={"kind": "highlight", "album_id": "11111111-1111-4111-8111-111111111111", "title": "H1"})
    page = client.get("/jobs?kind=highlight").text
    assert "H1" in page and "C1" not in page


def test_templates_escape_user_text(client):
    login(client)
    client.post("/create", data={"kind": "collage", "asset_ids": f"{U1}\n{U2}", "title": "<script>alert(1)</script>"})
    page = client.get("/jobs").text
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_create_highlight_by_person_and_dates(client):
    login(client)
    page = client.get("/create").text
    assert "Asha" in page  # people picker
    r = client.post("/create", data={"kind": "highlight", "person_id": "00000009-0000-4000-8000-000000000000",
                                     "date_from": "2026-01-01", "date_to": "2026-06-30"}, follow_redirects=False)
    assert r.status_code == 303
    job = client.app.state.s.jobs.queued()[0]
    assert job.params["person_ids"] == ["00000009-0000-4000-8000-000000000000"]
    assert job.params["date_from"] == "2026-01-01" and job.params["date_to"] == "2026-06-30"


def test_create_rejects_bad_ids_and_misplaced_filters(client):
    login(client)
    r = client.post("/create", data={"kind": "collage", "asset_ids": "not-an-id\nx"})
    assert r.status_code == 422 and "8-4-4-4-12" in r.text
    r = client.post("/create", data={"kind": "collage", "date_from": "2026-01-01"})
    assert r.status_code == 422 and "for highlight videos" in r.text


def test_collage_style_from_create(client):
    login(client)
    client.post("/create", data={"kind": "collage", "asset_ids": f"{U1}\n{U2}", "style": "fan"})
    assert client.app.state.s.jobs.queued()[0].params["style"] == "fan"


def test_thumb_proxy_validates_id(client):
    login(client)
    assert client.get("/thumb/../../etc").status_code == 404
    r = client.get(f"/thumb/{U1}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_suggestions_page_approve_and_dismiss(client):
    from app.suggestions import Idea

    login(client)
    state = client.app.state.s
    state.suggestions.add(Idea("collage:e1", "collage", "Goa · 14–17 Nov 2026", "An event with 38 photos.", {"asset_ids": []}, [U1]))
    state.suggestions.add(Idea("cinematic:e1", "cinematic", "IMG_1 · Goa", "A sharp photo.", {"asset_ids": [U1]}, [U1]))
    page = client.get("/suggestions").text
    assert "2 ideas waiting" in page and f"/thumb/{U1}" in page and "An event with 38 photos." in page
    ids = [s["id"] for s in state.suggestions.list()]
    client.post(f"/suggestions/{ids[1]}/approve")
    client.post(f"/suggestions/{ids[0]}/dismiss")
    assert state.suggestions.list() == []
    job = state.jobs.queued()[0]
    assert job.kind == "collage" and job.origin == "suggestion"


def test_scan_now_reports(client):
    login(client)
    r = client.post("/suggestions/scan")
    assert r.status_code == 200 and "Checked 0 photos in 0 events" in r.text


def _finish(state, job, details=None, sources=None):
    state.jobs.claim_next()
    state.jobs.finish(job.id, "done", result={"sources": sources or [U1, U2], "details": details or {}})


def test_edit_text_remakes_with_same_photos_and_hides_old(client):
    login(client)
    state = client.app.state.s
    job = state.jobs.enqueue("collage", "Goa", {"album_id": "11111111-1111-4111-8111-111111111111", "asset_ids": []})
    _finish(state, job, details={"style": "fan"})
    page = client.get("/jobs").text
    assert f"/jobs/{job.id}/edit" in page and "Save as preset" in page
    form = client.get(f"/jobs/{job.id}/edit").text
    assert 'name="caption"' in form and "Gallery fan" in form
    r = client.post(f"/jobs/{job.id}/edit", data={"title": "Goa 2026", "caption": "Goa with Asha"}, follow_redirects=False)
    assert r.status_code == 303
    new = state.jobs.queued()[0]
    assert new.title == "Goa 2026"
    assert new.params["caption"] == "Goa with Asha"
    assert new.params["style"] == "fan"  # the look it got is kept
    assert new.params["asset_ids"] == [U1, U2] and "album_id" not in new.params  # same photos
    assert new.params["edit_of"] == job.id
    assert state.jobs.get(job.id).superseded_by == new.id
    assert f"/jobs/{job.id}/edit" not in client.get("/jobs").text  # old version hidden


def test_edit_only_changes_allowed_keys(client):
    login(client)
    state = client.app.state.s
    job = state.jobs.enqueue("highlight", "Trip", {"album_id": "11111111-1111-4111-8111-111111111111", "asset_ids": []})
    _finish(state, job, details={"music": "song.mp3", "title_style": "typewriter"})
    client.post(f"/jobs/{job.id}/edit", data={"caption": "Our trip", "subtitle": "Sep 2026", "style": "fan"})
    new = state.jobs.queued()[0]
    assert new.params["caption"] == "Our trip" and new.params["subtitle"] == "Sep 2026"
    assert new.params["music"] == "song.mp3" and new.params["title_style"] == "typewriter"
    assert "style" not in new.params  # collage-only key ignored for a video


def test_cinematic_has_no_editable_text(client):
    login(client)
    state = client.app.state.s
    job = state.jobs.enqueue("cinematic", "IMG", {"asset_ids": [U1]})
    _finish(state, job, sources=[U1])
    assert f"/jobs/{job.id}/edit" not in client.get("/jobs").text
    assert client.get(f"/jobs/{job.id}/edit", follow_redirects=False).status_code == 303


def test_preset_from_job_and_reuse_on_create(client):
    login(client)
    state = client.app.state.s
    job = state.jobs.enqueue("collage", "Goa", {"asset_ids": [U1, U2], "style": "random", "caption": "Goa"})
    _finish(state, job, details={"style": "scatter"})
    r = client.post(f"/jobs/{job.id}/preset", data={"name": "Scatter with caption"}, follow_redirects=False)
    assert "Saved preset" in client.get(r.headers["location"]).text
    p = state.presets.list()[0]
    assert p["params"] == {"style": "scatter", "caption": "Goa"}  # "random" becomes what it landed on
    assert "asset_ids" not in p["params"]  # never the photos
    page = client.get(f"/create?preset={p['id']}").text
    assert "Scatter with caption" in page and 'value="Goa"' in page
    client.post("/create", data={"kind": "collage", "album_id": "11111111-1111-4111-8111-111111111111", "preset_id": str(p["id"])})
    queued = state.jobs.queued()[0]
    assert queued.params["caption"] == "Goa"
    # Same name twice is refused with a message.
    r = client.post(f"/jobs/{job.id}/preset", data={"name": "Scatter with caption"}, follow_redirects=False)
    assert r.status_code == 303, (r.status_code, r.text[:300])
    assert "already exists" in client.get(r.headers["location"]).text
    client.post(f"/presets/{p['id']}/delete")
    assert state.presets.list() == []


def test_create_popout_job(client):
    login(client)
    r = client.post("/create", data={"kind": "popout", "asset_ids": U1, "styles": ["splash", "frame"], "caption": "Goa"}, follow_redirects=False)
    assert r.status_code == 303
    job = client.app.state.s.jobs.queued()[0]
    assert job.kind == "popout" and job.params["styles"] == ["splash", "frame"] and job.params["caption"] == "Goa"
    r = client.post("/create", data={"kind": "popout", "asset_ids": U1})
    assert r.status_code == 422 and "at least one style" in r.text


def test_settings_page_has_new_sections(client):
    login(client)
    page = client.get("/settings").text
    assert "Housekeeping" in page and "Pop-out photo" in page and "Birthday memories" in page and "Album per event" in page
