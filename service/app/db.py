"""SQLite storage. One file in the data volume; WAL mode so the UI can read while the worker writes."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL CHECK (kind IN ('collage', 'cinematic', 'highlight', 'popout')),
    status       TEXT    NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'skipped', 'cancelled')),
    origin       TEXT    NOT NULL CHECK (origin IN ('manual', 'suggestion')),
    title        TEXT    NOT NULL,
    params       TEXT    NOT NULL DEFAULT '{}',
    position     REAL    NOT NULL,
    progress     INTEGER NOT NULL DEFAULT 0,
    stage        TEXT    NOT NULL DEFAULT '',
    message      TEXT    NOT NULL DEFAULT '',
    result       TEXT    NOT NULL DEFAULT '{}',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    superseded_by INTEGER  -- set when the job was edited and re-made as a new job
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, position);

CREATE TABLE IF NOT EXISTS presets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    kind        TEXT    NOT NULL,
    params      TEXT    NOT NULL,  -- style / ratio / caption / styles; never the photos
    from_job    INTEGER,
    created_at  TEXT    NOT NULL,
    last_used   TEXT
);

CREATE TABLE IF NOT EXISTS job_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  INTEGER NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    at      TEXT    NOT NULL,
    line    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS job_log_job ON job_log (job_id, id);

CREATE TABLE IF NOT EXISTS settings (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    data  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    password_hash  TEXT    NOT NULL,
    failed_count   INTEGER NOT NULL DEFAULT 0,
    locked_until   TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    expires_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS suggestions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT    NOT NULL UNIQUE,  -- kind + event; never suggested twice
    kind        TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    reason      TEXT    NOT NULL,
    params      TEXT    NOT NULL,
    preview_ids TEXT    NOT NULL DEFAULT '[]',
    status      TEXT    NOT NULL CHECK (status IN ('new', 'approved', 'dismissed')),
    job_id      INTEGER,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


class Database:
    """A single shared connection guarded by a lock (the app runs one process with one worker)."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._migrate()
            self._conn.executescript(SCHEMA)

    def _migrate(self) -> None:
        """Bring a database made by an older version up to date (SQLite can't alter a CHECK).

        Follows SQLite's recipe: build `jobs_new`, copy, drop `jobs`, rename `jobs_new` to `jobs`.
        Renaming the *old* table instead would make SQLite rewrite `job_log`'s foreign key to point
        at the dropped table, which an early version of this migration did; `_repair_job_log` fixes
        databases left in that state.
        """
        self._repair_job_log()
        row = self._conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'jobs'").fetchone()
        if row is None or "'popout'" in row[0]:
            return
        create_jobs = SCHEMA.split("CREATE INDEX IF NOT EXISTS jobs_queue")[0]
        create_new = create_jobs.replace("CREATE TABLE IF NOT EXISTS jobs (", "CREATE TABLE jobs_new (", 1)
        self._conn.execute("PRAGMA foreign_keys = OFF")
        self._conn.executescript(
            "BEGIN;\n"
            + create_new
            + """;
            INSERT INTO jobs_new (id, kind, status, origin, title, params, position, progress, stage, message,
                                  result, cancel_requested, created_at, started_at, finished_at)
                SELECT id, kind, status, origin, title, params, position, progress, stage, message,
                       result, cancel_requested, created_at, started_at, finished_at FROM jobs;
            DROP TABLE jobs;
            ALTER TABLE jobs_new RENAME TO jobs;
            COMMIT;
            """
        )
        self._conn.execute("PRAGMA foreign_keys = ON")

    def _repair_job_log(self) -> None:
        """Point `job_log` back at `jobs` if an earlier migration left it referencing `jobs_old`."""
        row = self._conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'job_log'").fetchone()
        if row is None or "jobs_old" not in row[0]:
            return
        columns = SCHEMA.split("CREATE TABLE IF NOT EXISTS job_log")[1].split(";")[0]  # " (\n id ... )"
        self._conn.execute("PRAGMA foreign_keys = OFF")
        self._conn.executescript(
            "BEGIN;\nCREATE TABLE job_log_new"
            + columns
            + """;
            INSERT INTO job_log_new (id, job_id, at, line)
                SELECT id, job_id, at, line FROM job_log WHERE job_id IN (SELECT id FROM jobs);
            DROP TABLE job_log;
            ALTER TABLE job_log_new RENAME TO job_log;
            CREATE INDEX IF NOT EXISTS job_log_job ON job_log (job_id, id);
            COMMIT;
            """
        )
        self._conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self.lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self.lock:
            return self._conn.execute(sql, params).fetchone()

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def close(self) -> None:
        with self.lock:
            self._conn.close()


class _Transaction:
    def __init__(self, db: Database) -> None:
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        self.db.lock.acquire()
        self.db._conn.execute("BEGIN IMMEDIATE")
        return self.db._conn

    def __exit__(self, exc_type: object, *_: object) -> None:
        try:
            self.db._conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.db.lock.release()
