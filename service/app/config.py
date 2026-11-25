"""Environment configuration. Secrets come only from the environment (.env), never from the UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    immich_url: str = Field(..., description="Base URL of the Immich server, e.g. http://192.168.1.10:2283")
    immich_api_key: SecretStr = Field(..., description="Immich API key with the permissions in docs/plan.md")

    photos_data_dir: Path = Field(Path("/data"), description="SQLite database and work files")
    photos_music_dir: Path = Field(Path("/music"), description="Read-only folder of music tracks")
    photos_password: SecretStr | None = Field(None, description="Initial UI password (12+ chars); generated if unset")
    photos_session_hours: int = Field(12, ge=1, le=24 * 30)
    photos_secure_cookie: bool = Field(False, description="Set true when served over HTTPS")
    photos_request_timeout: float = Field(30.0, gt=0)
    photos_output_mode: Literal["immich", "folder"] = Field(
        "immich", description="'folder' saves results to PHOTOS_OUTPUT_DIR and never writes to Immich (test mode)"
    )
    photos_output_dir: Path | None = Field(None, description="Defaults to <PHOTOS_DATA_DIR>/output")
    photos_ffmpeg: str = Field("ffmpeg", description="ffmpeg executable (name on PATH or full path)")
    photos_model_base_url: str = Field(
        "https://huggingface.co", description="Where model files are downloaded from on first use"
    )

    @field_validator("immich_url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("IMMICH_URL must start with http:// or https://")
        return value

    @field_validator("photos_password", mode="before")
    @classmethod
    def _empty_password_is_unset(cls, value: object) -> object:
        # `PHOTOS_PASSWORD=` (empty, as in .env.example) means "generate one on first start".
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("photos_password")
    @classmethod
    def _password_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 12:
            raise ValueError("PHOTOS_PASSWORD must be at least 12 characters")
        return value

    @property
    def db_path(self) -> Path:
        return self.photos_data_dir / "photos.sqlite3"

    @property
    def output_dir(self) -> Path:
        return self.photos_output_dir or self.photos_data_dir / "output"

    @property
    def models_dir(self) -> Path:
        return self.photos_data_dir / "models"

    @property
    def folder_mode(self) -> bool:
        return self.photos_output_mode == "folder"


@lru_cache
def get_config() -> Config:
    return Config()  # type: ignore[call-arg]  # values come from the environment
