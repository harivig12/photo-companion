"""Settings the user changes in the UI. Stored as one JSON row; validated with pydantic."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .db import Database

HEX_COLOUR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class SuggestionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["suggest", "auto", "off"] = "suggest"
    scan_at: str = Field("03:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    event_gap_hours: int = Field(3, ge=1, le=48)
    min_good_photos: int = Field(6, ge=2, le=50)
    max_per_day: int = Field(3, ge=0, le=20)


class OutputSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concurrency: int = Field(1, ge=1, le=4)
    album_name: str = Field("Creations", min_length=1, max_length=100)
    tag_prefix: str = Field("creation:", min_length=1, max_length=40)
    # Birthday events become Immich "birthday" memories (the person's name, not "N years ago").
    add_to_memories: bool = True
    # Every other event gets its own album named after it ("Deepavali 2026", "Yercaud trip ...").
    named_albums: bool = True
    stack_with_source: bool = True


class CollageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # random = one of prints / scatter / fan per job; grid = flat cells; prints = tilted
    # bordered prints; scatter = polaroids; fan = gallery fan.
    style: Literal["random", "grid", "prints", "scatter", "fan"] = "random"
    canvas_ratio: Literal["1:1", "4:5", "9:16", "16:9"] = "1:1"
    max_photos_manual: int = Field(9, ge=2, le=9)
    max_photos_auto: int = Field(12, ge=2, le=12)
    border_px: int = Field(8, ge=0, le=64)
    radius_px: int = Field(12, ge=0, le=64)
    border_colour: str = "#FFFFFF"
    skip_blurry: bool = True
    face_aware_crop: bool = True

    @field_validator("border_colour")
    @classmethod
    def _colour(cls, value: str) -> str:
        if not HEX_COLOUR.match(value):
            raise ValueError("Use a colour like #FFFFFF")
        return value.upper()


class CinematicSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motion: Literal["auto", "zoom", "dolly", "circle"] = "auto"
    seconds: int = Field(4, ge=2, le=10)
    fps: int = 30
    resolution: Literal["1080p", "720p"] = "1080p"
    min_depth_spread: float = Field(0.20, ge=0.0, le=1.0)
    upload_depth_map: bool = True

    @field_validator("fps")
    @classmethod
    def _fps(cls, value: int) -> int:
        # An int field (not Literal) so form strings like "30" coerce cleanly.
        if value not in (24, 30):
            raise ValueError("Frame rate must be 24 or 30")
        return value


class HighlightSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_seconds: int = Field(30, ge=10, le=600)
    max_seconds: int = Field(90, ge=10, le=600)
    aspect: Literal["16:9", "9:16", "1:1"] = "16:9"
    seconds_per_photo: float = Field(3.0, ge=1.0, le=10.0)
    cut_on_beat: bool = True
    duck_under_speech: bool = True
    title_card: bool = True
    title_style: Literal["fade-up", "typewriter", "letters", "random"] = "random"

    @model_validator(mode="after")
    def _range(self) -> HighlightSettings:
        if self.min_seconds > self.max_seconds:
            raise ValueError("Minimum length can't be more than the maximum")
        return self


POPOUT_STYLES = ("splash", "spotlight", "blur", "frame", "sticker")


class PopoutSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    splash: bool = True
    spotlight: bool = True
    blur: bool = True
    frame: bool = True
    sticker: bool = True
    splash_video: bool = False

    def styles(self) -> list[str]:
        return [s for s in POPOUT_STYLES if getattr(self, s)]


class HousekeepingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Finished jobs, their logs, folder-mode output and handled suggestions older than this are
    # removed. 0 = keep everything. Creations already in Immich are never touched.
    retention_days: int = Field(30, ge=0, le=3650)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suggestions: SuggestionSettings = SuggestionSettings()
    output: OutputSettings = OutputSettings()
    collage: CollageSettings = CollageSettings()
    cinematic: CinematicSettings = CinematicSettings()
    highlight: HighlightSettings = HighlightSettings()
    popout: PopoutSettings = PopoutSettings()
    housekeeping: HousekeepingSettings = HousekeepingSettings()


class SettingsStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self) -> AppSettings:
        row = self.db.one("SELECT data FROM settings WHERE id = 1")
        if row is None:
            return AppSettings()
        # Unknown keys from an older version are dropped rather than failing startup.
        stored = json.loads(row["data"])
        merged = AppSettings().model_dump()
        for section, values in stored.items():
            if section in merged and isinstance(values, dict):
                merged[section].update({k: v for k, v in values.items() if k in merged[section]})
        return AppSettings.model_validate(merged)

    def save(self, settings: AppSettings) -> None:
        # Re-validate: attribute assignment on a model skips validators, and an invalid row
        # would stop every later load (and the worker) until fixed by hand.
        settings = AppSettings.model_validate(settings.model_dump())
        self.db.execute(
            "INSERT INTO settings (id, data) VALUES (1, ?) ON CONFLICT (id) DO UPDATE SET data = excluded.data",
            (settings.model_dump_json(),),
        )
