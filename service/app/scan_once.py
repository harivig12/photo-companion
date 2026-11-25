"""Run one suggestion scan now and print the ideas (Immich is only read).

    python -m app.scan_once [--days 30]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from .config import get_config
from .db import Database
from .immich import ImmichGateway
from .jobs import JobStore
from .settings_store import SettingsStore
from .suggestions import SuggestionStore, scan
from .worker import Worker


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, help="look back this many days (default: since the last scan, or 30)")
    p.add_argument("--forget-new", action="store_true", help="drop ideas still waiting, so they can be offered again")
    p.add_argument("--run", choices=("collage", "cinematic", "highlight"), help="approve the first idea of this kind and run it")
    args = p.parse_args()
    config = get_config()
    db = Database(config.db_path)
    store = SuggestionStore(db)
    if args.forget_new:
        print(f"forgot {store.clear_new()} waiting idea(s)")
    if args.days:
        store.set_last_scan(datetime.now(timezone.utc) - timedelta(days=args.days))
    settings = SettingsStore(db).get()
    async with ImmichGateway(config) as gw:
        res = await scan(gw, store, JobStore(db), settings)
    print(f"checked {res['checked']} items since {res['since']:%d %b %Y} in {res['events']} events -> {res['added']} new idea(s)")
    for s in store.list("new"):
        print(f"  [{s['kind']:9}] {s['title']}  - {s['reason']}")
    if args.run:
        idea = next((s for s in store.list("new") if s["kind"] == args.run), None)
        if idea is None:
            print(f"no waiting {args.run} idea")
            return
        jobs = JobStore(db)
        job_id = store.approve(idea["id"], jobs)
        async with ImmichGateway(config) as gw:
            await Worker(jobs, SettingsStore(db), gw, config).run_once()
        done = jobs.get(job_id)
        print(f"ran job {job_id}: {done.status} - {done.message}")
        print(done.result.get("folder", ""))


if __name__ == "__main__":
    asyncio.run(main())
