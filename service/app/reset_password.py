"""Reset the UI password from inside the container:

    docker compose exec photos python -m app.reset_password

Prompts for the new password (not echoed) and signs out every session.
"""

from __future__ import annotations

import getpass
import sys

from .auth import MIN_PASSWORD_LENGTH, Auth
from .config import get_config
from .db import Database


def main() -> int:
    config = get_config()
    db = Database(config.db_path)
    auth = Auth(db)
    first = getpass.getpass("New password: ")
    if len(first) < MIN_PASSWORD_LENGTH:
        print(f"Use at least {MIN_PASSWORD_LENGTH} characters.", file=sys.stderr)
        return 1
    if getpass.getpass("Again: ") != first:
        print("The two entries don't match.", file=sys.stderr)
        return 1
    auth.ensure_password(first)  # no-op when a password already exists
    auth.set_password(first)
    print("Password changed. All sessions were signed out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
