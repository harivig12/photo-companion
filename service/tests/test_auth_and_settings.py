import pytest
from pydantic import ValidationError

from app.auth import MAX_FAILURES, Auth, hash_password, verify_password
from app.settings_store import AppSettings, SettingsStore


def test_hash_roundtrip_and_salt():
    h = hash_password("correct horse battery")
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)
    assert hash_password("correct horse battery") != h  # random salt
    assert not verify_password("x", "garbage")


def test_generated_password_when_none_given(db):
    auth = Auth(db)
    generated = auth.ensure_password(None)
    assert generated and len(generated) >= 12
    assert auth.ensure_password("another password here") is None  # only on first start
    assert auth.login(generated).ok


def test_lockout_after_failures(db):
    auth = Auth(db)
    auth.ensure_password("correct horse battery")
    for _ in range(MAX_FAILURES - 1):
        assert auth.login("nope").error == "Wrong password."
    assert "Too many" in auth.login("nope").error
    assert not auth.login("correct horse battery").ok  # still locked


def test_sessions_and_password_change(db):
    auth = Auth(db)
    auth.ensure_password("correct horse battery")
    token = auth.login("correct horse battery").token
    assert auth.session_valid(token)
    assert not auth.session_valid("forged")
    with pytest.raises(ValueError):
        auth.set_password("short")
    auth.set_password("a brand new password")
    assert not auth.session_valid(token)  # signed out everywhere
    assert auth.login("a brand new password").ok


def test_settings_defaults_save_and_merge(db):
    store = SettingsStore(db)
    s = store.get()
    assert s.suggestions.mode == "suggest"
    assert s.output.album_name == "Creations"
    s.collage.border_colour = "#abcdef"
    store.save(AppSettings.model_validate(s.model_dump()))
    assert store.get().collage.border_colour == "#ABCDEF"

    # A stored row with an unknown key (older/newer version) still loads.
    db.execute("UPDATE settings SET data = ? WHERE id = 1", ('{"collage": {"old_key": 1, "border_px": 4}}',))
    loaded = store.get()
    assert loaded.collage.border_px == 4 and loaded.output.album_name == "Creations"


def test_settings_validation():
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"collage": {"border_colour": "white"}})
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"highlight": {"min_seconds": 120, "max_seconds": 60}})
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"cinematic": {"fps": "25"}})
    assert AppSettings.model_validate({"cinematic": {"fps": "24"}}).cinematic.fps == 24


def test_save_rejects_invalid_settings_set_by_code(db):
    store = SettingsStore(db)
    s = store.get()
    s.highlight.min_seconds, s.highlight.max_seconds = 120, 60  # assignment skips validators
    with pytest.raises(ValidationError):
        store.save(s)
    assert store.get().highlight.max_seconds == 90  # nothing bad was stored
