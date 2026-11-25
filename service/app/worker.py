"""Background worker: takes one queued job at a time, runs its pipeline, delivers the files.

Delivery is either Immich (upload, then album / tag / stack / Memories) or, in test mode
(PHOTOS_OUTPUT_MODE=folder), a folder per job under PHOTOS_OUTPUT_DIR. Nothing is written
to Immich in test mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .immich import ImmichError, ImmichGateway
from .jobs import Job, JobStore
from .pipelines import (
    JobCancelled,
    JobInputError,
    NotBuiltYet,
    Pipeline,
    PipelineContext,
    PipelineResult,
    default_registry,
)
from .settings_store import AppSettings, SettingsStore

log = logging.getLogger(__name__)
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Worker:
    def __init__(
        self,
        jobs: JobStore,
        settings: SettingsStore,
        gateway: ImmichGateway,
        config: Config,
        registry: dict[str, Pipeline] | None = None,
        poll_seconds: float = 2.0,
    ) -> None:
        self.jobs = jobs
        self.settings = settings
        self.gateway = gateway
        self.config = config
        self.registry = registry or default_registry()
        self.poll_seconds = poll_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def housekeep(self, now: datetime | None = None) -> dict:
        """Remove finished jobs, their logs and folder-mode output past the retention period.
        Creations already in Immich are never touched; presets keep working (they copy params)."""
        from .suggestions import SuggestionStore

        days = self.settings.get().housekeeping.retention_days
        gone = self.jobs.purge(days, now)
        removed_files = 0
        root = self.config.output_dir.resolve()
        for job in gone:
            folder = job.result.get("folder")
            if not folder:
                continue
            path = Path(folder).resolve()
            if root in path.parents and path.is_dir():  # only ever inside the output folder
                removed_files += sum(1 for _ in path.rglob("*") if _.is_file())
                shutil.rmtree(path, ignore_errors=True)
        trimmed = SuggestionStore(self.jobs.db).purge(days, now)
        if gone or trimmed:
            log.info("Housekeeping: removed %d job(s), %d file(s), trimmed %d suggestion(s)", len(gone), removed_files, trimmed)
        return {"jobs": len(gone), "files": removed_files, "suggestions": trimmed}

    async def run_forever(self) -> None:
        recovered = self.jobs.recover_interrupted()
        if recovered:
            log.info("Re-queued %d job(s) interrupted by a restart", recovered)
        last_housekeeping = 0.0
        while not self._stop.is_set():
            if asyncio.get_running_loop().time() - last_housekeeping > 3600:
                try:
                    self.housekeep()
                except Exception:  # noqa: BLE001
                    log.exception("Housekeeping failed")
                last_housekeeping = asyncio.get_running_loop().time()
            try:
                ran = await self.run_once()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                log.exception("Worker loop error")
                ran = False
            if not ran:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
                except TimeoutError:
                    pass

    async def run_once(self) -> bool:
        """Run the next queued job, if any. Returns True when a job was run."""
        job = self.jobs.claim_next()
        if job is None:
            return False
        await self._run(job)
        return True

    async def _run(self, job: Job) -> None:
        settings = self.settings.get()
        pipeline = self.registry.get(job.kind)
        self.jobs.log(job.id, f"start {job.kind_label}: {job.title}")
        if pipeline is None:
            self.jobs.finish(job.id, "failed", f"No pipeline for {job.kind}.")
            return
        work_dir = Path(tempfile.mkdtemp(prefix=f"job{job.id}-"))
        ctx = PipelineContext(
            job=job,
            gateway=self.gateway,
            settings=settings,
            progress=lambda stage, pct: self.jobs.progress(job.id, stage, pct),
            log=lambda line: self.jobs.log(job.id, line),
            cancel_requested=lambda: self.jobs.is_cancel_requested(job.id),
            work_dir=work_dir,
            models_dir=self.config.models_dir,
            music_dir=self.config.photos_music_dir,
            ffmpeg=self.config.photos_ffmpeg,
            model_base_url=self.config.photos_model_base_url,
        )
        try:
            result = await pipeline(ctx)
            if result.status == "skipped":
                self.jobs.finish(job.id, "skipped", result.message, {"details": result.details})
                return
            ctx.check_cancel()
            if self.config.folder_mode:
                ctx.progress("Saving to folder", 95)
                written = self._save_to_folder(job, result)
            else:
                ctx.progress("Adding to Immich", 95)
                written = await self._deliver_to_immich(job, result, settings)
            self.jobs.finish(job.id, "done", result.message, written)
            self.jobs.log(job.id, "done")
        except JobCancelled:
            self.jobs.finish(job.id, "cancelled", "Cancelled while running.")
        except (NotBuiltYet, JobInputError) as exc:
            self.jobs.finish(job.id, "failed", str(exc))
        except ImmichError as exc:
            self.jobs.log(job.id, f"immich: {exc}")
            self.jobs.finish(job.id, "failed", str(exc))
        except Exception as exc:  # noqa: BLE001 - shown to the user, details in the log
            log.exception("Job %s failed", job.id)
            self.jobs.finish(job.id, "failed", f"Unexpected error: {type(exc).__name__}. See the service log.")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ---- delivery --------------------------------------------------------------------------

    def _save_to_folder(self, job: Job, result: PipelineResult) -> dict:
        folder = self.config.output_dir / f"{job.id:05d}-{job.kind}"
        folder.mkdir(parents=True, exist_ok=True)
        files = []
        for f in result.files:
            name = _SAFE.sub("_", f.filename)
            (folder / name).write_bytes(f.data)
            files.append({"file": name, "role": f.role, "bytes": len(f.data), "taken_at": f.taken_at.isoformat()})
        summary = {
            "job": job.id, "kind": job.kind, "title": job.title, "message": result.message,
            "sources": result.source_ids, "files": files, "details": result.details,
            "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (folder / "job.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        self.jobs.log(job.id, f"saved {len(files)} file(s) to {folder}")
        return {"folder": str(folder), "files": files, "sources": result.source_ids, "details": result.details}

    async def _deliver_to_immich(self, job: Job, result: PipelineResult, settings: AppSettings) -> dict:
        """Upload the files, then album, tag, stack and Memories entry."""
        out = settings.output
        uploaded: list[tuple[str, str]] = []  # (asset_id, role)
        for f in result.files:
            up = await self.gateway.upload(
                f.data, f.filename, taken_at=f.taken_at, job_id=job.id, kind=job.kind,
                source_ids=result.source_ids, duration_ms=f.duration_ms,
            )
            uploaded.append((up.asset_id, f.role))
            self.jobs.log(job.id, f"uploaded {f.filename} -> {up.asset_id}{' (duplicate)' if up.duplicate else ''}")
        main_ids = [a for a, role in uploaded if role == "main"]
        all_ids = [a for a, _ in uploaded]
        written: dict = {
            "outputs": [{"asset_id": a, "role": r} for a, r in uploaded],
            "sources": result.source_ids,
            "details": result.details,
        }
        if not all_ids:
            return written

        album_id = await self.gateway.ensure_album(out.album_name)
        await self.gateway.add_to_album(album_id, all_ids)
        written["album_id"] = album_id

        await self.gateway.tag(f"{out.tag_prefix}{job.kind}", all_ids)

        # Stacking only makes sense for a single source photo (cinematic + depth map).
        if out.stack_with_source and len(result.source_ids) == 1:
            written["stack_id"] = await self.gateway.stack(result.source_ids[0], all_ids)

        # Memories variation: Immich already makes "N years ago" memories by itself, so only a
        # birthday becomes a memory (Immich then shows the person's name); every other event goes
        # to an album named after it.
        event = job.params.get("event") or {}
        if event.get("title"):
            for aid in main_ids:
                await self.gateway.set_description(aid, event["title"])
            if out.named_albums:
                named = await self.gateway.ensure_album(event["title"][:100])
                await self.gateway.add_to_album(named, all_ids)
                written["event_album_id"] = named
        if out.add_to_memories and main_ids and event.get("kind") == "birthday" and event.get("person_id"):
            year = result.year or datetime.now().year
            written["memory_id"] = await self.gateway.add_memory(
                main_ids, year=year, person_id=event["person_id"], person_name=event.get("person_name")
            )
        return written
