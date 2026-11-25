"""FastAPI app: password-protected UI (layout C, "command deck") plus a small JSON API for polling."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from . import __version__
from .auth import Auth
from .config import Config, get_config
from .db import Database
from .immich import ImmichError, ImmichGateway, PermissionReport, ServerStatus
from .imaging.popout import STYLE_LABELS as POPOUT_LABELS
from .jobs import EDITABLE, KIND_LABELS, KINDS, JobStore, PresetStore
from .settings_store import POPOUT_STYLES
from .permissions import ALL_REQUIRED, affected_features
from .settings_store import AppSettings, SettingsStore
from .suggestions import SuggestionStore, due, scan
from .worker import Worker

UUID_RE = __import__("re").compile(r"^[0-9a-fA-F-]{36}$")

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
SESSION_COOKIE = "photos_session"
STATUS_TTL_SECONDS = 30
PUBLIC_PATHS = ("/login", "/health", "/static/")

CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)


class AppState:
    def __init__(self, config: Config, gateway: ImmichGateway, db: Database) -> None:
        self.config = config
        self.gateway = gateway
        self.db = db
        self.jobs = JobStore(db)
        self.settings = SettingsStore(db)
        self.auth = Auth(db, session_hours=config.photos_session_hours)
        self.suggestions = SuggestionStore(db)
        self.presets = PresetStore(db)
        self.worker: Worker | None = None
        self.scan_lock = asyncio.Lock()
        self._status: ServerStatus | None = None
        self._status_at = 0.0

    async def immich_status(self, refresh: bool = False) -> ServerStatus:
        if refresh or self._status is None or time.monotonic() - self._status_at > STATUS_TTL_SECONDS:
            self._status = await self.gateway.status()
            self._status_at = time.monotonic()
        return self._status


def create_app(
    config: Config | None = None,
    gateway_factory: Callable[[Config], ImmichGateway] | None = None,
    start_worker: bool = True,
) -> FastAPI:
    config = config or get_config()
    gateway_factory = gateway_factory or ImmichGateway

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(config.db_path)
        gateway = gateway_factory(config)
        await gateway.open()
        state = AppState(config, gateway, db)
        generated = state.auth.ensure_password(
            config.photos_password.get_secret_value() if config.photos_password else None
        )
        if generated:
            # Printed once, on first start only. Change it in Settings > Account.
            log.warning("No PHOTOS_PASSWORD set. Generated UI password (shown once): %s", generated)
        app.state.s = state
        task = scheduler = None
        if start_worker:
            state.worker = Worker(state.jobs, state.settings, gateway, config)
            task = asyncio.create_task(state.worker.run_forever())
            scheduler = asyncio.create_task(_daily_scans(state))
        try:
            yield
        finally:
            if scheduler:
                scheduler.cancel()
            if state.worker and task:
                state.worker.stop()
                await task
            await gateway.close()
            db.close()

    app = FastAPI(title="Photos companion", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals["kind_labels"] = KIND_LABELS

    # ---- middleware: security headers, same-origin POSTs, login ------------------------

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
            return Response("Cross-site request blocked.", status_code=403)
        if not path.startswith(PUBLIC_PATHS):
            state: AppState = request.app.state.s
            if not state.auth.session_valid(request.cookies.get(SESSION_COOKIE)):
                if path.startswith("/api/"):
                    return JSONResponse({"error": "sign in required"}, status_code=401)
                return RedirectResponse("/login", status_code=303)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if not path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def st(request: Request) -> AppState:
        return request.app.state.s

    async def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        state = st(request)
        counts = state.jobs.counts()
        ctx.setdefault("status_code", 200)
        status_code = ctx.pop("status_code")
        return templates.TemplateResponse(
            request,
            name,
            {
                "immich": await state.immich_status(), "counts": counts, "version": __version__,
                "folder_mode": config.folder_mode, "output_dir": str(config.output_dir), **ctx,
            },
            status_code=status_code,
        )

    # ---- public --------------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "login.html", {"error": None, "immich_url": config.immich_url})

    @app.post("/login")
    async def login(request: Request, password: str = Form("")) -> Response:
        result = st(request).auth.login(password)
        if not result.ok:
            return templates.TemplateResponse(
                request, "login.html", {"error": result.error, "immich_url": config.immich_url}, status_code=401
            )
        resp = RedirectResponse("/jobs", status_code=303)
        resp.set_cookie(
            SESSION_COOKIE,
            result.token or "",
            max_age=config.photos_session_hours * 3600,
            httponly=True,
            samesite="strict",
            secure=config.photos_secure_cookie,
        )
        return resp

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        st(request).auth.logout(request.cookies.get(SESSION_COOKIE))
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    # ---- jobs ----------------------------------------------------------------------------

    @app.get("/")
    async def root() -> Response:
        return RedirectResponse("/jobs", status_code=303)

    @app.get("/jobs", response_class=HTMLResponse)
    async def jobs_page(request: Request, kind: str | None = None, note: str | None = None) -> HTMLResponse:
        state = st(request)
        kind = kind if kind in KINDS else None
        running = state.jobs.running()
        live = state.jobs.get(running[0].id, with_log=True) if running else None
        return await page(
            request,
            "jobs.html",
            nav="jobs",
            kind=kind,
            jobs=state.jobs.list(kind=kind),
            live=live,
            by_kind=state.jobs.counts_by_kind_today(),
            note=note,
            retention_days=state.settings.get().housekeeping.retention_days,
        )

    @app.get("/api/jobs")
    async def jobs_api(request: Request) -> dict[str, Any]:
        state = st(request)
        running = state.jobs.running()
        live = state.jobs.get(running[0].id, with_log=True) if running else None
        return {
            "counts": state.jobs.counts(),
            "running": None
            if live is None
            else {"id": live.id, "title": live.title, "kind": live.kind, "stage": live.stage, "progress": live.progress, "log": live.log[-8:]},
            "queued": [{"id": j.id, "title": j.title, "kind": j.kind} for j in state.jobs.queued()],
        }

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(request: Request, job_id: int) -> Response:
        st(request).jobs.cancel(job_id)
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs/{job_id}/up")
    async def move_up(request: Request, job_id: int) -> Response:
        st(request).jobs.move_up(job_id)
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs/{job_id}/retry")
    async def retry(request: Request, job_id: int) -> Response:
        st(request).jobs.retry(job_id)
        return RedirectResponse("/jobs", status_code=303)

    async def create_page(request: Request, errors: dict[str, str] | None = None, values: dict | None = None, status_code: int = 200) -> HTMLResponse:
        state = st(request)
        albums, people, album_error = [], [], None
        try:
            albums = await state.gateway.list_albums()
            people = await state.gateway.list_people()
        except ImmichError as exc:
            album_error = str(exc)
        values = values or {"kind": "collage"}
        preset = None
        if values.get("preset_id"):
            preset = state.presets.get(int(values["preset_id"]))
        return await page(
            request, "create.html", nav="create", albums=albums, people=people, album_error=album_error,
            errors=errors or {}, values=values, presets=state.presets.list(), preset=preset,
            popout_styles=POPOUT_STYLES, popout_labels=POPOUT_LABELS, popout_default=state.settings.get().popout.styles(),
            status_code=status_code,
        )

    @app.get("/create", response_class=HTMLResponse)
    async def create_form(request: Request, preset: int | None = None) -> HTMLResponse:
        state = st(request)
        values: dict[str, Any] = {"kind": "collage"}
        if preset and (p := state.presets.get(preset)):
            values = {"kind": p["kind"], "preset_id": str(p["id"]), **{k: v for k, v in p["params"].items() if isinstance(v, str)}}
            if isinstance(p["params"].get("styles"), list):
                values["styles"] = p["params"]["styles"]
        return await create_page(request, values=values)

    @app.post("/create")
    async def create_job(request: Request) -> Response:
        form = await request.form()
        get = lambda k: str(form.get(k) or "").strip()  # noqa: E731
        kind, album_id, title, caption = get("kind"), get("album_id"), get("title"), get("caption")
        person_id, date_from, date_to, style = get("person_id"), get("date_from"), get("date_to"), get("style")
        asset_ids = get("asset_ids")
        styles = [s for s in form.getlist("styles") if s in POPOUT_STYLES]
        splash_video = form.get("splash_video") is not None
        preset_id = get("preset_id")
        values = {"kind": kind, "album_id": album_id, "asset_ids": asset_ids, "title": title, "caption": caption,
                  "person_id": person_id, "date_from": date_from, "date_to": date_to, "style": style,
                  "styles": styles, "preset_id": preset_id}
        errors: dict[str, str] = {}
        if kind not in KINDS:
            errors["kind"] = "Pick what to make."
        ids = [a.strip() for a in asset_ids.replace(",", "\n").splitlines() if a.strip()]
        if any(not UUID_RE.match(i) for i in ids):
            errors["source"] = "Asset IDs look like 8-4-4-4-12 hex characters."
        if (person_id or date_from or date_to) and kind != "highlight":
            errors["source"] = "People and dates are for highlight videos; for a collage pick an album."
        if style and style not in ("random", "grid", "prints", "scatter", "fan"):
            errors["style"] = "Unknown style."
        if not album_id and not ids and not (kind == "highlight" and (person_id or date_from)):
            errors.setdefault("source", "Pick an album, a person or dates, or paste asset IDs.")
        if kind in ("cinematic", "popout") and len(ids) > 1:
            errors["source"] = f"A {KIND_LABELS[kind].lower()} uses exactly one photo."
        if kind == "cinematic" and not ids:
            errors["source"] = "Paste the asset ID of one photo."
        if kind == "popout" and not styles and not splash_video:
            errors["styles"] = "Pick at least one style."
        if len(caption) > 60:
            errors["caption"] = "Keep the text to 60 characters."
        if errors:
            return await create_page(request, errors=errors, values=values, status_code=422)
        source = "album" if album_id else ("person" if person_id else ("dates" if date_from else f"{len(ids)} photo(s)"))
        label = title or caption or f"{KIND_LABELS[kind]} · {source}"
        params: dict[str, Any] = {"album_id": album_id or None, "asset_ids": ids}
        if person_id and UUID_RE.match(person_id):
            params["person_ids"] = [person_id]
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if style and kind == "collage":
            params["style"] = style
        if caption:
            params["caption"] = caption
        if kind == "popout":
            params["styles"] = styles
            params["splash_video"] = splash_video
        state = st(request)
        if preset_id.isdigit() and (p := state.presets.get(int(preset_id))) and p["kind"] == kind:
            for k, v in p["params"].items():
                params.setdefault(k, v)  # what the form says wins over the preset
            state.presets.use(p["id"])
        state.jobs.enqueue(kind, label[:120], params, origin="manual")
        return RedirectResponse("/jobs", status_code=303)

    # ---- edit a finished job / save its look as a preset ---------------------------------

    @app.get("/jobs/{job_id}/edit", response_class=HTMLResponse)
    async def edit_form(request: Request, job_id: int) -> Response:
        job = st(request).jobs.get(job_id)
        if job is None or not job.editable:
            return RedirectResponse("/jobs", status_code=303)
        details = job.result.get("details") or {}
        values = {
            "title": job.title,
            "caption": job.params.get("caption") or details.get("caption") or details.get("title") or "",
            "subtitle": job.params.get("subtitle") or details.get("subtitle") or "",
            "style": job.params.get("style") or details.get("style") or "",
            "title_style": job.params.get("title_style") or details.get("title_style") or "",
            "styles": job.params.get("styles") or details.get("styles") or [],
        }
        return await page(request, "edit.html", nav="jobs", job=job, values=values, errors={},
                          editable=EDITABLE[job.kind], popout_styles=POPOUT_STYLES, popout_labels=POPOUT_LABELS)

    @app.post("/jobs/{job_id}/edit")
    async def edit_job(request: Request, job_id: int) -> Response:
        state = st(request)
        job = state.jobs.get(job_id)
        if job is None or not job.editable:
            return RedirectResponse("/jobs", status_code=303)
        form = await request.form()
        changes: dict[str, Any] = {}
        for key in EDITABLE[job.kind]:
            if key == "styles":
                changes["styles"] = [s for s in form.getlist("styles") if s in POPOUT_STYLES]
            elif key in form:
                changes[key] = str(form.get(key) or "").strip()[:60]
        if job.kind == "popout" and not changes.get("styles"):
            values = {**{k: form.get(k, "") for k in ("title", "caption")}, "styles": []}
            return await page(request, "edit.html", nav="jobs", job=job, values=values,
                              errors={"styles": "Pick at least one style."}, editable=EDITABLE[job.kind],
                              popout_styles=POPOUT_STYLES, popout_labels=POPOUT_LABELS, status_code=422)
        state.jobs.edit(job_id, changes, title=str(form.get("title") or "").strip() or None)
        return RedirectResponse("/jobs", status_code=303)

    @app.post("/jobs/{job_id}/preset")
    async def save_preset(request: Request, job_id: int, name: str = Form("")) -> Response:
        state = st(request)
        job = state.jobs.get(job_id)
        from urllib.parse import quote

        if job is None or job.status != "done":
            return RedirectResponse("/jobs", status_code=303)
        try:
            state.presets.save_from_job(job, name or job.title)
            note = f"Saved preset '{(name or job.title)[:60]}'."
        except ValueError as exc:
            note = str(exc)
        return RedirectResponse(f"/jobs?note={quote(note)}", status_code=303)

    @app.post("/presets/{preset_id}/delete")
    async def delete_preset(request: Request, preset_id: int) -> Response:
        st(request).presets.delete(preset_id)
        return RedirectResponse("/settings#presets", status_code=303)

        return RedirectResponse("/jobs", status_code=303)

    # ---- suggestions ---------------------------------------------------------------------

    @app.get("/suggestions", response_class=HTMLResponse)
    async def suggestions(request: Request, scanned: str | None = None) -> HTMLResponse:
        state = st(request)
        return await page(
            request, "suggestions.html", nav="suggestions", settings=state.settings.get(),
            ideas=state.suggestions.list("new"), last_scan=state.suggestions.last_scan(), scanned=scanned,
        )

    @app.post("/suggestions/scan")
    async def scan_now(request: Request) -> Response:
        state = st(request)
        try:
            async with state.scan_lock:
                res = await scan(state.gateway, state.suggestions, state.jobs, state.settings.get())
            note = f"Checked {res['checked']} photos in {res['events']} events: {res['added']} new idea(s)."
        except ImmichError as exc:
            note = f"Scan failed: {exc}"
        from urllib.parse import quote

        return RedirectResponse(f"/suggestions?scanned={quote(note)}", status_code=303)

    @app.post("/suggestions/{sid}/approve")
    async def approve(request: Request, sid: int) -> Response:
        st(request).suggestions.approve(sid, st(request).jobs)
        return RedirectResponse("/suggestions", status_code=303)

    @app.post("/suggestions/{sid}/dismiss")
    async def dismiss(request: Request, sid: int) -> Response:
        st(request).suggestions.dismiss(sid)
        return RedirectResponse("/suggestions", status_code=303)

    @app.post("/suggestions/approve-all")
    async def approve_all(request: Request) -> Response:
        state = st(request)
        for s in state.suggestions.list("new"):
            state.suggestions.approve(s["id"], state.jobs)
        return RedirectResponse("/jobs", status_code=303)

    @app.get("/thumb/{asset_id}")
    async def thumb(request: Request, asset_id: str) -> Response:
        """Thumbnails proxied from Immich, so the browser never talks to Immich (CSP stays 'self')."""
        if not UUID_RE.match(asset_id):
            return Response(status_code=404)
        try:
            data = await st(request).gateway.thumbnail(asset_id, "thumbnail")
        except ImmichError:
            return Response(status_code=404)
        ctype = "image/webp" if data[:4] == b"RIFF" else "image/jpeg"
        return Response(data, media_type=ctype, headers={"Cache-Control": "private, max-age=3600"})

    # ---- settings ------------------------------------------------------------------------

    async def settings_page(
        request: Request,
        settings: AppSettings | None = None,
        errors: dict[str, str] | None = None,
        saved: bool = False,
        report: PermissionReport | None = None,
        report_error: str | None = None,
        password_message: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        return await page(
            request,
            "settings.html",
            nav="settings",
            s=settings or st(request).settings.get(),
            errors=errors or {},
            saved=saved,
            report=report,
            report_error=report_error,
            required=ALL_REQUIRED,
            affected=affected_features(report.missing) if report else [],
            password_message=password_message,
            immich_url=config.immich_url,
            music_dir=str(config.photos_music_dir),
            presets=st(request).presets.list(),
            popout_styles=POPOUT_STYLES,
            popout_labels=POPOUT_LABELS,
            status_code=status_code,
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_form(request: Request) -> HTMLResponse:
        return await settings_page(request)

    @app.post("/settings")
    async def save_settings(request: Request) -> Response:
        form = await request.form()
        data = _settings_from_form(dict(form))
        try:
            settings = AppSettings.model_validate(data)
        except ValidationError as exc:
            errors = {".".join(str(p) for p in e["loc"]): e["msg"].removeprefix("Value error, ") for e in exc.errors()}
            return await settings_page(request, settings=st(request).settings.get(), errors=errors, status_code=422)
        st(request).settings.save(settings)
        return await settings_page(request, settings=settings, saved=True)

    @app.post("/settings/permissions")
    async def check_permissions(request: Request) -> Response:
        state = st(request)
        try:
            report = await state.gateway.permission_report()
            await state.immich_status(refresh=True)
            return await settings_page(request, report=report)
        except ImmichError as exc:
            return await settings_page(request, report_error=str(exc))

    @app.post("/settings/password")
    async def change_password(request: Request, new_password: str = Form("")) -> Response:
        try:
            st(request).auth.set_password(new_password)
        except ValueError as exc:
            return await settings_page(request, password_message=str(exc), status_code=422)
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    return app


async def _daily_scans(state: AppState, every_s: float = 60.0) -> None:
    """Run the suggestion scan once a day at the configured time."""
    from datetime import datetime, timezone

    while True:
        await asyncio.sleep(every_s)
        try:
            settings = state.settings.get()
            if due(settings, state.suggestions.last_scan(), datetime.now(timezone.utc)):
                async with state.scan_lock:
                    await scan(state.gateway, state.suggestions, state.jobs, settings)
        except Exception:  # noqa: BLE001 - try again tomorrow; never kill the app
            log.exception("Daily suggestion scan failed")


def _same_origin(request: Request) -> bool:
    """POSTs must carry an Origin (or Referer) that matches the Host header."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return False
    host = request.headers.get("host", "")
    return urlsplit(source).netloc == host


def _settings_from_form(form: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Turn `section.field` form keys into nested settings; unchecked checkboxes become False."""
    defaults = AppSettings().model_dump()
    data: dict[str, dict[str, Any]] = {}
    for section, fields in defaults.items():
        data[section] = {}
        for name, default in fields.items():
            key = f"{section}.{name}"
            if isinstance(default, bool):
                data[section][name] = key in form
            elif key in form:
                data[section][name] = form[key]
    return data


def run() -> None:  # pragma: no cover - used by the container entrypoint
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=8765, proxy_headers=False)
