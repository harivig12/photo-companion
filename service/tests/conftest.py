from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Config  # noqa: E402
from app.db import Database  # noqa: E402

IMMICH = "http://immich.test:2283"
API = IMMICH + "/api"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Tests must not pick up a developer's real Immich settings or test-mode flags."""
    import os

    for key in list(os.environ):
        if (key.startswith("PHOTOS_") or key.startswith("IMMICH_")) and key != "PHOTOS_FFMPEG":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        immich_url=IMMICH,
        immich_api_key="test-key",
        photos_data_dir=tmp_path,
        photos_music_dir=tmp_path / "music",
        photos_password="correct horse battery",
        _env_file=None,
    )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.sqlite3")
    yield database
    database.close()
