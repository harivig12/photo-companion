"""Job queue on SQLite: enqueue, claim, progress, finish, cancel, reorder, recover."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import Database

KINDS = ("collage", "cinematic", "highlight", "popout")
KIND_LABELS = {"collage": "Collage", "cinematic": "Cinematic photo", "highlight": "Highlight video", "popout": "Pop-out photo"}
# Params a user may change when editing a finished job (the text on the creation and its look).
EDITABLE = {
    "collage": ("caption", "style"),
    "highlight": ("caption", "subtitle", "title_style"),
    "popout": ("caption", "styles"),
    "cinematic": (),
}
# Params kept in a preset: how it looks, never which photos.
PRESET_KEYS = ("style", "styles", "caption", "subtitle", "title_style", "ratio")
ORIGINS = ("manual", "suggestion")
FINISHED = ("done", "failed", "skipped", "cancelled")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: int
    kind: str
    status: str
    origin: str
    title: str
    params: dict[str, Any]
    position: float
    progress: int
    stage: str
    message: str
    result: dict[str, Any]
    cancel_requested: bool
    created_at: str
    started_at: str | None
    finished_at: str | None
    superseded_by: int | None = None
    log: list[str] = field(default_factory=list)

    @property
    def kind_label(self) -> str:
        return KIND_LABELS[self.kind]

    @property
    def editable(self) -> bool:
        return self.status == "done" and bool(EDITABLE.get(self.kind)) and self.superseded_by is None


def _row_to_job(row: Any) -> Job:
    return Job(
        id=row["id"],
        kind=row["kind"],
        status=row["status"],
        origin=row["origin"],
        title=row["title"],
        params=json.loads(row["params"]),
        position=row["position"],
        progress=row["progress"],
        stage=row["stage"],
        message=row["message"],
        result=json.loads(row["result"]),
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        superseded_by=row["superseded_by"] if "superseded_by" in row.keys() else None,
    )


class JobStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- create and read ---------------------------------------------------------------

    def enqueue(self, kind: str, title: str, params: dict[str, Any], origin: str = "manual") -> Job:
        if kind not in KINDS:
            raise ValueError(f"unknown job kind: {kind}")
        if origin not in ORIGINS:
            raise ValueError(f"unknown origin: {origin}")
        with self.db.transaction() as conn:
            top = conn.execute("SELECT COALESCE(MAX(position), 0) FROM jobs").fetchone()[0]
            cur = conn.execute(
                "INSERT INTO jobs (kind, status, origin, title, params, position, created_at)"
                " VALUES (?, 'queued', ?, ?, ?, ?, ?)",
                (kind, origin, title, json.dumps(params), top + 1, now_iso()),
            )
            job_id = cur.lastrowid
        job = self.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: int, with_log: bool = False) -> Job | None:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            return None
        job = _row_to_job(row)
        if with_log:
            job.log = [r["line"] for r in self.db.query("SELECT line FROM job_log WHERE job_id = ? ORDER BY id", (job_id,))]
        return job

    def list(self, *, kind: str | None = None, limit: int = 100) -> list[Job]:
        """Running first, then queued in queue order, then finished newest first."""
        # Jobs replaced by an edited version are hidden; the new version shows instead.
        sql = (
            "SELECT * FROM jobs WHERE superseded_by IS NULL {where} ORDER BY"
            " CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,"
            " CASE WHEN status = 'queued' THEN position END ASC,"
            " COALESCE(finished_at, created_at) DESC LIMIT ?"
        )
        if kind:
            return [_row_to_job(r) for r in self.db.query(sql.format(where="AND kind = ?"), (kind, limit))]
        return [_row_to_job(r) for r in self.db.query(sql.format(where=""), (limit,))]

    def queued(self) -> list[Job]:
        return [_row_to_job(r) for r in self.db.query("SELECT * FROM jobs WHERE status = 'queued' ORDER BY position")]

    def running(self) -> list[Job]:
        return [_row_to_job(r) for r in self.db.query("SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at")]

    def counts(self, today_prefix: str | None = None) -> dict[str, int]:
        today_prefix = today_prefix or now_iso()[:10]
        counts = {s: 0 for s in ("queued", "running", "done_today", "failed")}
        for row in self.db.query("SELECT status, COUNT(*) AS n FROM jobs WHERE status IN ('queued','running','failed') GROUP BY status"):
            counts[row["status"]] = row["n"]
        row = self.db.one(
            "SELECT COUNT(*) AS n FROM jobs WHERE status = 'done' AND finished_at LIKE ?", (today_prefix + "%",)
        )
        counts["done_today"] = row["n"] if row else 0
        return counts

    def counts_by_kind_today(self, today_prefix: str | None = None) -> dict[str, dict[str, int]]:
        """Done vs planned (done today + queued + running) per kind, for the command-deck dots."""
        today_prefix = today_prefix or now_iso()[:10]
        out = {k: {"done": 0, "planned": 0} for k in KINDS}
        for row in self.db.query(
            "SELECT kind, status, COUNT(*) AS n FROM jobs"
            " WHERE status IN ('queued','running') OR (status = 'done' AND finished_at LIKE ?)"
            " GROUP BY kind, status",
            (today_prefix + "%",),
        ):
            out[row["kind"]]["planned"] += row["n"]
            if row["status"] == "done":
                out[row["kind"]]["done"] += row["n"]
        return out

    # ---- worker side -------------------------------------------------------------------

    def claim_next(self) -> Job | None:
        """Atomically move the first queued job to running."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY position LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, progress = 0, stage = '', message = ''"
                " WHERE id = ?",
                (now_iso(), row["id"]),
            )
            job_id = row["id"]
        return self.get(job_id)

    def progress(self, job_id: int, stage: str, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        self.db.execute("UPDATE jobs SET stage = ?, progress = ? WHERE id = ? AND status = 'running'", (stage, percent, job_id))

    def log(self, job_id: int, line: str) -> None:
        self.db.execute("INSERT INTO job_log (job_id, at, line) VALUES (?, ?, ?)", (job_id, now_iso(), line[:500]))

    def finish(self, job_id: int, status: str, message: str = "", result: dict[str, Any] | None = None) -> None:
        if status not in FINISHED:
            raise ValueError(f"not a finished status: {status}")
        self.db.execute(
            "UPDATE jobs SET status = ?, message = ?, result = ?, finished_at = ?,"
            " progress = CASE WHEN ? = 'done' THEN 100 ELSE progress END WHERE id = ?",
            (status, message[:1000], json.dumps(result or {}), now_iso(), status, job_id),
        )

    def is_cancel_requested(self, job_id: int) -> bool:
        row = self.db.one("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,))
        return bool(row and row["cancel_requested"])

    def recover_interrupted(self) -> int:
        """Jobs left running by a crash or restart go back to the front of the queue."""
        cur = self.db.execute(
            "UPDATE jobs SET status = 'queued', position = position - 1000000, progress = 0, stage = '',"
            " message = 'Restarted after the service stopped.' WHERE status = 'running'"
        )
        return cur.rowcount

    # ---- UI actions --------------------------------------------------------------------

    def cancel(self, job_id: int) -> bool:
        """Queued jobs are cancelled at once; a running job is asked to stop at its next step."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return False
            if row["status"] == "queued":
                conn.execute(
                    "UPDATE jobs SET status = 'cancelled', finished_at = ?, message = 'Removed from the queue.' WHERE id = ?",
                    (now_iso(), job_id),
                )
                return True
            if row["status"] == "running":
                conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                return True
            return False

    def move_up(self, job_id: int) -> bool:
        """Swap a queued job with the one ahead of it."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT position FROM jobs WHERE id = ? AND status = 'queued'", (job_id,)).fetchone()
            if row is None:
                return False
            ahead = conn.execute(
                "SELECT id, position FROM jobs WHERE status = 'queued' AND position < ? ORDER BY position DESC LIMIT 1",
                (row["position"],),
            ).fetchone()
            if ahead is None:
                return False
            conn.execute("UPDATE jobs SET position = ? WHERE id = ?", (ahead["position"], job_id))
            conn.execute("UPDATE jobs SET position = ? WHERE id = ?", (row["position"], ahead["id"]))
            return True

    def retry(self, job_id: int) -> Job | None:
        """Queue a fresh copy of a failed, skipped or cancelled job."""
        old = self.get(job_id)
        if old is None or old.status not in ("failed", "skipped", "cancelled"):
            return None
        params = dict(old.params)
        if old.status == "skipped":
            params["force"] = True  # "Run anyway" skips the eligibility check
        return self.enqueue(old.kind, old.title, params, origin=old.origin)

    # ---- editing a finished job --------------------------------------------------------

    def edit(self, job_id: int, changes: dict[str, Any], title: str | None = None) -> Job | None:
        """Re-make a finished job with new text / look. The old job is hidden (superseded) once
        the new one is queued; its files are left alone until housekeeping removes them."""
        old = self.get(job_id)
        if old is None or not old.editable:
            return None
        allowed = EDITABLE[old.kind]
        params = dict(old.params)
        for key, value in changes.items():
            if key in allowed:
                if value in (None, "", []):
                    params.pop(key, None)
                else:
                    params[key] = value
        params["edit_of"] = old.id
        # Keep the look it actually got (a "random" style must not re-roll) unless changed.
        details = old.result.get("details") or {}
        if old.kind == "collage" and "style" not in changes and details.get("style"):
            params["style"] = details["style"]
        if old.kind == "highlight" and "title_style" not in changes and details.get("title_style"):
            params["title_style"] = details["title_style"]
        if old.kind == "highlight" and details.get("music"):
            params["music"] = details["music"]  # same track, so only the text changes
        if old.kind == "popout" and "styles" not in changes and details.get("styles"):
            params["styles"] = details["styles"]
        # Re-use the exact photos of the first run, so only the text / look changes.
        if old.result.get("sources"):
            params["asset_ids"] = list(old.result["sources"])
            params.pop("album_id", None)
            params.pop("event_from", None)
            params.pop("event_to", None)
            params.pop("person_ids", None)
        new = self.enqueue(old.kind, (title or old.title)[:120], params, origin=old.origin)
        self.db.execute("UPDATE jobs SET superseded_by = ? WHERE id = ?", (new.id, old.id))
        return new

    # ---- housekeeping ------------------------------------------------------------------

    def purge(self, older_than_days: int, now: datetime | None = None) -> list[Job]:
        """Delete finished jobs (and their logs) older than N days. Returns what was removed so
        the caller can delete folder-mode output. Queued and running jobs are never touched."""
        if older_than_days <= 0:
            return []
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        rows = self.db.query(
            "SELECT * FROM jobs WHERE status IN ('done','failed','skipped','cancelled') AND COALESCE(finished_at, created_at) < ?",
            (cutoff,),
        )
        gone = [_row_to_job(r) for r in rows]
        if gone:
            ids = [j.id for j in gone]
            marks = ",".join("?" * len(ids))
            with self.db.transaction() as conn:
                conn.execute(f"DELETE FROM job_log WHERE job_id IN ({marks})", ids)
                conn.execute(f"DELETE FROM jobs WHERE id IN ({marks})", ids)
                conn.execute(f"UPDATE presets SET from_job = NULL WHERE from_job IN ({marks})", ids)
        return gone


class PresetStore:
    """Saved looks ("presets") made from a finished job: reuse a style without its photos."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save_from_job(self, job: Job, name: str) -> int:
        name = name.strip()[:60]
        if not name:
            raise ValueError("Give the preset a name.")
        params = {k: v for k, v in job.params.items() if k in PRESET_KEYS}
        details = job.result.get("details") or {}
        if job.kind == "collage" and params.get("style") in (None, "random") and details.get("style"):
            params["style"] = details["style"]  # a "random" pick becomes the style it landed on
        if job.kind == "highlight" and params.get("title_style") in (None, "random") and details.get("title_style"):
            params["title_style"] = details["title_style"]
        if job.kind == "popout" and not params.get("styles") and details.get("styles"):
            params["styles"] = details["styles"]
        try:
            cur = self.db.execute(
                "INSERT INTO presets (name, kind, params, from_job, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, job.kind, json.dumps(params), job.id, now_iso()),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A preset called '{name}' already exists.") from exc
        return int(cur.lastrowid)

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM presets" + (" WHERE kind = ?" if kind else "") + " ORDER BY COALESCE(last_used, created_at) DESC",
            (kind,) if kind else (),
        )
        return [dict(r, params=json.loads(r["params"])) for r in rows]

    def get(self, preset_id: int) -> dict[str, Any] | None:
        r = self.db.one("SELECT * FROM presets WHERE id = ?", (preset_id,))
        return dict(r, params=json.loads(r["params"])) if r else None

    def use(self, preset_id: int) -> None:
        self.db.execute("UPDATE presets SET last_used = ? WHERE id = ?", (now_iso(), preset_id))

    def delete(self, preset_id: int) -> bool:
        return self.db.execute("DELETE FROM presets WHERE id = ?", (preset_id,)).rowcount == 1
