"""A fake Immich v3.4.2 server on a local port, for tests.

Real HTTP (aiohttp.web), so requests go through immichpy's actual encoding. Routes return
canned JSON shaped like the immichpy models; every request is recorded for assertions.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from app.config import Config

TS = "2026-11-24T10:00:00.000Z"
ALBUM_ID = "11111111-1111-4111-8111-111111111111"
UPLOAD_ID = "22222222-2222-4222-8222-222222222222"
DEPTH_ID = "33333333-3333-4333-8333-333333333333"
SOURCE_ID = "44444444-4444-4444-8444-444444444444"
TAG_ID = "55555555-5555-4555-8555-555555555555"
STACK_ID = "66666666-6666-4666-8666-666666666666"
MEMORY_ID = "77777777-7777-4777-8777-777777777777"
OWNER_ID = "88888888-8888-4888-8888-888888888888"


def album(album_id: str = ALBUM_ID, name: str = "Creations", count: int = 0) -> dict:
    owner = {
        "role": "owner",
        "user": {
            "avatarColor": "primary", "email": "owner@example.com", "id": OWNER_ID, "name": "Owner",
            "profileChangedAt": TS, "profileImagePath": "",
        },
    }
    return {
        "albumName": name, "albumThumbnailAssetId": None, "albumUsers": [owner], "assetCount": count,
        "createdAt": TS, "description": "", "hasSharedLink": False, "id": album_id,
        "isActivityEnabled": True, "shared": False, "updatedAt": TS,
    }


def api_key(permissions: list[str]) -> dict:
    return {"createdAt": TS, "id": "99999999-9999-4999-8999-999999999999", "name": "photos", "permissions": permissions, "updatedAt": TS}


def version(major: int = 3, minor: int = 4, patch: int = 2) -> dict:
    return {"major": major, "minor": minor, "patch": patch, "featureLevel": 1, "prerelease": None}


def tag(name: str) -> dict:
    return {"createdAt": TS, "id": TAG_ID, "name": name, "updatedAt": TS, "value": name}


def stack() -> dict:
    return {"assets": [], "id": STACK_ID, "primaryAssetId": SOURCE_ID}


def memory() -> dict:
    return {
        "assets": [], "createdAt": TS, "data": {"year": 2026}, "id": MEMORY_ID, "isSaved": False,
        "memoryAt": TS, "ownerId": OWNER_ID, "type": "on_this_day", "updatedAt": TS,
    }


@dataclass
class Recorded:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    json: Any = None
    form: dict[str, str] = field(default_factory=dict)  # multipart text fields; files as "<file:name>"


@dataclass
class Reply:
    status: int = 200
    body: Any = None


class FakeImmich:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Reply] = {}
        self.patterns: list = []
        self.requests: list[Recorded] = []
        self.server: TestServer | None = None

    def on(self, method: str, path: str, body: Any = None, status: int = 200) -> FakeImmich:
        self.routes[(method.upper(), "/api" + path)] = Reply(status, body)
        return self

    def uploads(self, asset_id: str = UPLOAD_ID, status: str = "created") -> FakeImmich:
        """POST /assets plus the follow-up metadata write."""
        self.on("POST", "/assets", {"id": asset_id, "status": status}, status=201 if status == "created" else 200)
        self.on(
            "PUT", f"/assets/{asset_id}/metadata",
            [{"key": "photos.companion", "updatedAt": TS, "value": {}}],
        )
        return self

    def write_back(self, *, album_exists: bool = True) -> FakeImmich:
        """Register every call the worker's write-back makes."""
        # GET /albums?name=X answers with an album of that exact name, so ensure_album() finds it.
        if album_exists:
            self.on_match(
                "GET", lambda r: r.path == "/api/albums",
                lambda r: Reply(200, [album(ALBUM_ID, r.query.get("name") or "Creations")]),
            )
        else:
            self.on("GET", "/albums", [])
        # PUT /assets/{id} (description) answers with the asset.
        self.on_match(
            "PUT", lambda r: r.path.startswith("/api/assets/") and r.path.count("/") == 3,
            lambda r: Reply(200, asset(r.path.split("/")[3], TS)),
        )
        if not album_exists:
            self.on("POST", "/albums", album(), status=201)
        self.on("PUT", f"/albums/{ALBUM_ID}/assets", [{"id": UPLOAD_ID, "success": True}])
        self.on("PUT", "/tags", [tag("creation:cinematic")])
        self.on("PUT", "/tags/assets", {"count": 1})
        self.on("POST", "/stacks", stack(), status=201)
        self.on("POST", "/memories", memory(), status=201)
        return self

    def calls(self, method: str, path: str) -> list[Recorded]:
        return [r for r in self.requests if r.method == method and r.path == "/api" + path]

    async def _handle(self, request: web.Request) -> web.Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        rec = Recorded(request.method, request.path, dict(request.query), headers)
        ctype = request.headers.get("Content-Type", "")
        if ctype.startswith("multipart/"):
            post = await request.post()
            for k, v in post.items():
                rec.form[k] = f"<file:{v.filename}>" if isinstance(v, web.FileField) else str(v)
        elif request.can_read_body:
            raw = await request.read()
            rec.json = json.loads(raw) if raw else None
        self.requests.append(rec)
        reply = self.routes.get((request.method, request.path))
        if reply is None:
            for (meth, pattern), r in self.patterns:
                if meth == request.method and pattern(request):
                    reply = r(request) if callable(r) else r
                    break
        if reply is None:
            return web.json_response({"message": f"no fake route for {request.method} {request.path}"}, status=404)
        if isinstance(reply.body, bytes):
            return web.Response(body=reply.body, status=reply.status, content_type="image/jpeg")
        return web.json_response(reply.body, status=reply.status)

    def on_match(self, method: str, pattern: Callable[[web.Request], bool], reply: Any) -> FakeImmich:
        """Route by predicate; `reply` is a Reply or a function request -> Reply."""
        self.patterns.append(((method.upper(), pattern), reply))
        return self

    def library(self, assets: list[dict], images: dict[str, bytes], faces: dict[str, list] | None = None) -> FakeImmich:
        """Serve an album search, asset info, faces and thumbnails/originals for these assets."""
        by_id = {a["id"]: a for a in assets}
        faces = faces or {}
        self.on("POST", "/search/metadata", {
            "albums": {"count": 0, "facets": [], "items": [], "total": 0},
            "assets": {"count": len(assets), "facets": [], "items": assets, "total": len(assets), "nextPage": None},
        })
        self.on("GET", "/albums", [album(ALBUM_ID, "Trip", len(assets))])
        self.on_match("GET", lambda r: r.path.startswith("/api/assets/") and r.path.count("/") == 3,
                      lambda r: Reply(200, by_id[r.path.split("/")[3]]))
        self.on_match("GET", lambda r: r.path.endswith(("/thumbnail", "/original", "/video/playback")),
                      lambda r: Reply(200, images[r.path.split("/")[3]]))
        self.on_match("GET", lambda r: r.path == "/api/faces", lambda r: Reply(200, faces.get(r.query.get("id"), [])))
        return self

    async def __aenter__(self) -> FakeImmich:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self.server = TestServer(app)
        await self.server.start_server()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.server:
            await self.server.close()

    @property
    def url(self) -> str:
        assert self.server is not None
        return str(self.server.make_url("")).rstrip("/")

    def config(self, base: Config) -> Config:
        """A copy of the test config pointing at this fake server."""
        return base.model_copy(update={"immich_url": self.url})


async def with_fake(fake: FakeImmich, base: Config, fn: Callable[[Config], Awaitable[Any]]) -> Any:
    async with fake:
        return await fn(fake.config(base))


def asset(asset_id: str, taken: str, filename: str = "IMG.jpg", width: int = 4000, height: int = 3000, **extra) -> dict:
    """An AssetResponseDto for Immich v3.3 (camelCase JSON)."""
    body = {
        "checksum": "x", "createdAt": TS, "duration": 0, "fileCreatedAt": taken, "fileModifiedAt": taken,
        "hasMetadata": True, "height": height, "id": asset_id, "isArchived": False, "isEdited": False,
        "isFavorite": False, "isOffline": False, "isTrashed": False, "localDateTime": taken,
        "originalFileName": filename, "originalMimeType": "image/jpeg", "originalPath": "/x/" + filename,
        "ownerId": OWNER_ID, "thumbhash": None, "type": "IMAGE", "updatedAt": TS, "visibility": "timeline",
        "width": width, "people": [],
        "exifInfo": {"dateTimeOriginal": taken, "make": "Google", "exifImageWidth": width, "exifImageHeight": height},
    }
    body.update(extra)
    return body


def face(asset_w: int, asset_h: int, box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "boundingBoxX1": x1, "boundingBoxY1": y1, "boundingBoxX2": x2, "boundingBoxY2": y2,
        "id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "imageHeight": asset_h, "imageWidth": asset_w,
        "person": None, "sourceType": "machine-learning",
    }
