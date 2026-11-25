"""Run one creation from the command line, without the web UI.

    python -m app.run_job collage --album <album-id> [--ratio 4:5] [--title "Trip"]
    python -m app.run_job collage --assets <id> <id> ...

Uses the same worker as the service. Honors PHOTOS_OUTPUT_MODE: with `folder`, the result is
saved under PHOTOS_OUTPUT_DIR and nothing is written to Immich.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .config import get_config
from .db import Database
from .immich import ImmichGateway
from .jobs import KINDS, JobStore
from .settings_store import SettingsStore
from .worker import Worker


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=KINDS)
    p.add_argument("--album")
    p.add_argument("--assets", nargs="*", default=[])
    p.add_argument("--title", default="")
    p.add_argument("--ratio", help="collage canvas ratio, e.g. 1:1, 4:5, 9:16, 16:9")
    p.add_argument("--style", choices=("random", "grid", "prints", "scatter", "fan"), help="collage style for this job")
    p.add_argument("--person", nargs="*", default=[], help="highlight video: Immich person IDs")
    p.add_argument("--from", dest="date_from", help="highlight video: first day, YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", help="highlight video: last day, YYYY-MM-DD")
    p.add_argument("--caption", help="title for scatter / fan collages (default: album name)")
    args = p.parse_args()

    config = get_config()
    db = Database(config.db_path)
    jobs, settings = JobStore(db), SettingsStore(db)
    if args.ratio:
        s = settings.get()
        s.collage.canvas_ratio = args.ratio
        settings.save(type(s).model_validate(s.model_dump()))
    params = {"album_id": args.album, "asset_ids": args.assets}
    if args.person:
        params["person_ids"] = args.person
    if args.date_from:
        params["date_from"] = args.date_from
    if args.date_to:
        params["date_to"] = args.date_to
    if args.style:
        params["style"] = args.style
    if args.caption:
        params["caption"] = args.caption
    job = jobs.enqueue(args.kind, args.title or f"{args.kind} (cli)", params)
    async with ImmichGateway(config) as gw:
        await Worker(jobs, settings, gw, config).run_once()
    done = jobs.get(job.id, with_log=True)
    print(json.dumps({"status": done.status, "message": done.message, "result": done.result, "log": done.log}, indent=2, default=str))
    return 0 if done.status in ("done", "skipped") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
