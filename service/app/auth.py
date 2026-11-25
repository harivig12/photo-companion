"""Single-user password login: scrypt hash, server-side sessions, lockout after failed tries."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone

from .db import Database

log = logging.getLogger(__name__)

MAX_FAILURES = 5
LOCKOUT = timedelta(minutes=5)
MIN_PASSWORD_LENGTH = 12
_SCRYPT = {"n": 2**15, "r": 8, "p": 1, "maxmem": 64 * 1024 * 1024}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    salt, expected = base64.b64decode(salt_b64), base64.b64decode(digest_b64)
    actual = hashlib.scrypt(password.encode(), salt=salt, dklen=len(expected), **_SCRYPT)
    return hmac.compare_digest(actual, expected)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class LoginResult:
    def __init__(self, ok: bool, token: str | None = None, error: str | None = None) -> None:
        self.ok, self.token, self.error = ok, token, error


class Auth:
    def __init__(self, db: Database, session_hours: int = 12) -> None:
        self.db = db
        self.session_ttl = timedelta(hours=session_hours)

    def ensure_password(self, initial: str | None) -> str | None:
        """Set the first password. Returns a generated password (to log once) when none was given."""
        if self.db.one("SELECT 1 FROM auth WHERE id = 1"):
            return None
        generated = None
        if not initial:
            generated = secrets.token_urlsafe(15)
            initial = generated
        self.db.execute("INSERT INTO auth (id, password_hash) VALUES (1, ?)", (hash_password(initial),))
        return generated

    def set_password(self, new_password: str) -> None:
        if len(new_password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Use at least {MIN_PASSWORD_LENGTH} characters.")
        self.db.execute("UPDATE auth SET password_hash = ?, failed_count = 0, locked_until = NULL WHERE id = 1", (hash_password(new_password),))
        self.db.execute("DELETE FROM sessions")  # sign out everywhere

    def login(self, password: str) -> LoginResult:
        row = self.db.one("SELECT password_hash, failed_count, locked_until FROM auth WHERE id = 1")
        if row is None:
            return LoginResult(False, error="No password is set yet. Restart the service.")
        now = _now()
        if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > now:
            return LoginResult(False, error="Too many failed tries. Try again in a few minutes.")
        if not verify_password(password, row["password_hash"]):
            failures = row["failed_count"] + 1
            locked = (now + LOCKOUT).isoformat() if failures >= MAX_FAILURES else None
            self.db.execute(
                "UPDATE auth SET failed_count = ?, locked_until = ? WHERE id = 1",
                (0 if locked else failures, locked),
            )
            if locked:
                log.warning("UI sign-in locked for %s after %d failed tries", LOCKOUT, MAX_FAILURES)
                return LoginResult(False, error="Too many failed tries. Try again in a few minutes.")
            return LoginResult(False, error="Wrong password.")
        self.db.execute("UPDATE auth SET failed_count = 0, locked_until = NULL WHERE id = 1")
        token = secrets.token_urlsafe(32)
        self.db.execute("DELETE FROM sessions WHERE expires_at < ?", (now.isoformat(),))
        self.db.execute(
            "INSERT INTO sessions (token_hash, expires_at) VALUES (?, ?)",
            (_token_hash(token), (now + self.session_ttl).isoformat()),
        )
        return LoginResult(True, token=token)

    def session_valid(self, token: str | None) -> bool:
        if not token:
            return False
        row = self.db.one("SELECT expires_at FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        return bool(row and datetime.fromisoformat(row["expires_at"]) > _now())

    def logout(self, token: str | None) -> None:
        if token:
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
