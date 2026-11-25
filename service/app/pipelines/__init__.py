"""Creation pipelines and the shared context they run with.

A pipeline reads from Immich through `ctx.gateway`, reports progress, checks for cancel
between steps and returns the files it made. It never writes to Immich itself: the worker
either uploads the files and does the album / tag / stack / Memories write-back, or (test
mode) saves them to a folder.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from ..immich import ImmichGateway
from ..jobs import Job
from ..settings_store import AppSettings

T = TypeVar("T")


class JobCancelled(Exception):
    """Raised by `ctx.check_cancel()` when the user pressed Cancel."""


class NotBuiltYet(Exception):
    """The pipeline for this kind is a placeholder in this phase."""


class JobInputError(Exception):
    """The job can't run with the photos it was given. The message is shown to the user."""


@dataclass
class OutputFile:
    filename: str
    data: bytes
    taken_at: datetime  # where the creation sits on the Immich timeline
    role: Literal["main", "depth"] = "main"
    duration_ms: int | None = None


@dataclass
class PipelineResult:
    status: Literal["done", "skipped"]
    message: str = ""
    files: list[OutputFile] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    year: int | None = None  # year of the source photos, for the Memories entry
    details: dict[str, Any] = field(default_factory=dict)  # saved with the job (layout, scores, ...)


@dataclass
class PipelineContext:
    job: Job
    gateway: ImmichGateway
    settings: AppSettings
    progress: Callable[[str, int], None]
    log: Callable[[str], None]
    cancel_requested: Callable[[], bool]
    work_dir: Path | None = None  # scratch space, removed after the job
    models_dir: Path | None = None
    music_dir: Path | None = None
    ffmpeg: str = "ffmpeg"
    model_base_url: str = "https://huggingface.co"

    def check_cancel(self) -> None:
        if self.cancel_requested():
            raise JobCancelled()

    async def cpu(self, fn: Callable[..., T], *args: Any) -> T:
        """Run CPU-heavy work in a thread so the UI stays responsive."""
        return await asyncio.to_thread(fn, *args)


class Pipeline(Protocol):
    def __call__(self, ctx: PipelineContext) -> Awaitable[PipelineResult]: ...


def _placeholder(kind: str, phase: int) -> Pipeline:
    async def run(ctx: PipelineContext) -> PipelineResult:
        ctx.progress("Not built yet", 0)
        raise NotBuiltYet(f"The {kind} pipeline isn't built yet (phase {phase} of docs/plan.md).")

    return run


def default_registry() -> dict[str, Pipeline]:
    registry: dict[str, Pipeline] = {
        "collage": _placeholder("collage", 2),
        "cinematic": _placeholder("cinematic photo", 3),
        "highlight": _placeholder("highlight video", 4),
    }
    import importlib

    registry["popout"] = _placeholder("pop-out photo", 7)
    for kind, module, attr in (
        ("collage", "collage", "run_collage"),
        ("cinematic", "cinematic", "run_cinematic"),
        ("highlight", "highlight", "run_highlight"),
        ("popout", "popout", "run_popout"),
    ):
        name = f"{__name__}.{module}"
        try:
            registry[kind] = getattr(importlib.import_module(name), attr)
        except ModuleNotFoundError as exc:
            if exc.name != name:  # a real missing dependency: fail loudly
                raise
    return registry
