"""Print photo / video counts per album (read-only). Helps pick test albums.

    python -m app.album_stats [--max 1100]
"""

from __future__ import annotations

import argparse
import asyncio

from .config import get_config
from .immich import ImmichGateway


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=1100, help="skip albums bigger than this")
    args = p.parse_args()
    async with ImmichGateway(get_config()) as gw:
        for a in await gw.list_albums():
            if not 0 < a["count"] <= args.max:
                continue
            xs = await gw.album_assets(a["id"], limit=args.max)
            v = [x for x in xs if x.type == "VIDEO"]
            avg = sum((x.duration_ms or 0) for x in v) / max(1, len(v)) / 1000
            print(f"{a['name'][:30]:30} photos={len(xs) - len(v):4} videos={len(v):3} avg_video_s={avg:5.1f} id={a['id']}")


if __name__ == "__main__":
    asyncio.run(main())
