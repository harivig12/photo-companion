"""Thin gateway over immichpy: only the Immich calls the companion uses.

Pipelines and the worker talk to Immich through this class, so tests can mock one place
and an Immich API change only touches this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import aiohttp
from immichpy.client.generated import models as m
from immichpy.client.main import AsyncClient

from .config import Config
from .permissions import missing_permissions

log = logging.getLogger(__name__)

# Metadata key stored on every uploaded creation, so later scans can tell
# which source photos were already used and by which job.
CREATION_METADATA_KEY = "photos.companion"


class ImmichError(RuntimeError):
    """Immich was unreachable or rejected a call. The message is safe to show in the UI."""


@dataclass(frozen=True)
class ServerStatus:
    reachable: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PermissionReport:
    key_name: str
    granted: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class Upload:
    asset_id: str
    duplicate: bool


@dataclass(frozen=True)
class FaceBox:
    """A face as fractions (0-1) of the image width and height."""

    x1: float
    y1: float
    x2: float
    y2: float
    person: str | None = None
    person_id: str | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class AssetInfo:
    id: str
    type: str  # IMAGE or VIDEO
    filename: str
    mime: str
    width: int | None
    height: int | None
    taken_at: datetime
    is_favorite: bool
    visibility: str  # timeline, archive, hidden, locked
    is_trashed: bool
    latitude: float | None = None
    longitude: float | None = None
    people: tuple[str, ...] = ()
    duration_ms: int | None = None
    make: str | None = None
    city: str | None = None

    @property
    def usable(self) -> bool:
        """Visible on the timeline and not in the trash."""
        return not self.is_trashed and self.visibility == "timeline"


def _duration_ms(value: int | str | None) -> int | None:
    """Immich v3 sends milliseconds; older servers sent '0:00:12.345678'."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    try:
        h, mnt, s = value.split(":")
        return int((int(h) * 3600 + int(mnt) * 60 + float(s)) * 1000)
    except ValueError:
        return None


def _to_info(a: Any) -> AssetInfo:
    exif = a.exif_info
    taken = (exif.date_time_original if exif and exif.date_time_original else None) or a.local_date_time or a.file_created_at
    if taken.tzinfo is None:
        taken = taken.replace(tzinfo=timezone.utc)
    visibility = a.visibility.value if hasattr(a.visibility, "value") else str(a.visibility)
    return AssetInfo(
        id=str(a.id),
        type=a.type.value if hasattr(a.type, "value") else str(a.type),
        filename=a.original_file_name,
        mime=getattr(a, "original_mime_type", None) or "",
        width=a.width or (exif.exif_image_width if exif else None),
        height=a.height or (exif.exif_image_height if exif else None),
        taken_at=taken,
        is_favorite=bool(a.is_favorite),
        visibility=visibility,
        is_trashed=bool(a.is_trashed),
        latitude=exif.latitude if exif else None,
        longitude=exif.longitude if exif else None,
        people=tuple(p.name for p in (a.people or []) if p.name),
        duration_ms=_duration_ms(a.duration),
        make=exif.make if exif else None,
        city=exif.city if exif else None,
    )


def _describe(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if status == 401:
        return "Immich rejected the API key (401). Check IMMICH_API_KEY."
    if status == 403:
        return "The API key lacks a permission for this call (403). See Settings > Permission check."
    if status is not None:
        return f"Immich returned HTTP {status}."
    if isinstance(exc, (aiohttp.ClientError, TimeoutError, OSError)):
        return "Can't reach Immich. Check IMMICH_URL and that the server is running."
    return f"Unexpected error talking to Immich: {type(exc).__name__}"


class ImmichGateway:
    def __init__(self, config: Config, session: aiohttp.ClientSession | None = None) -> None:
        self._config = config
        self._injected_session = session
        self._session: aiohttp.ClientSession | None = None
        self._client: AsyncClient | None = None

    async def __aenter__(self) -> ImmichGateway:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._client is not None:
            return
        if self._injected_session is not None:
            self._session = self._injected_session
        else:
            timeout = aiohttp.ClientTimeout(total=self._config.photos_request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        self._client = AsyncClient(
            api_key=self._config.immich_api_key.get_secret_value(),
            base_url=self._config.immich_url,
            http_client=self._session,
        )

    async def close(self) -> None:
        # Only close a session we created; an injected one belongs to the caller.
        if self._session is not None and self._injected_session is None:
            await self._session.close()
        self._session = None
        self._client = None

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("ImmichGateway.open() was not called")
        return self._client

    async def _call(self, what: str, coro: Any) -> Any:
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 - translated into a UI-safe message
            log.warning("Immich call failed: %s: %r", what, exc)
            raise ImmichError(_describe(exc)) from exc

    # ---- status and permissions -------------------------------------------------------

    async def status(self) -> ServerStatus:
        try:
            await self.client.server.ping_server()
            v = await self.client.server.get_server_version()
        except Exception as exc:  # noqa: BLE001
            return ServerStatus(reachable=False, error=_describe(exc))
        return ServerStatus(reachable=True, version=f"v{v.major}.{v.minor}.{v.patch}")

    async def permission_report(self) -> PermissionReport:
        key = await self._call("get my api key", self.client.api_keys.get_my_api_key())
        granted = tuple(p.value if hasattr(p, "value") else str(p) for p in key.permissions)
        return PermissionReport(key_name=key.name, granted=granted, missing=tuple(missing_permissions(granted)))

    # ---- reading ----------------------------------------------------------------------

    async def list_albums(self) -> list[dict[str, Any]]:
        albums = await self._call("list albums", self.client.albums.get_all_albums())
        return [{"id": str(a.id), "name": a.album_name, "count": a.asset_count} for a in albums]

    async def album_assets(self, album_id: str, limit: int = 5000) -> list[AssetInfo]:
        """Assets in an album, with EXIF and people, oldest first."""
        out: list[AssetInfo] = []
        page = 1
        while page and len(out) < limit:
            res = await self._call(
                "search album",
                self.client.search.search_assets(
                    m.MetadataSearchDto(
                        album_ids=[UUID(album_id)], with_exif=True, with_people=True, size=min(1000, limit), page=page,
                        order=m.AssetOrder.ASC,
                    )
                ),
            )
            out.extend(_to_info(a) for a in res.assets.items)
            page = int(res.assets.next_page) if res.assets.next_page else None
        return out[:limit]

    async def search(
        self,
        *,
        person_ids: list[str] | None = None,
        taken_after: datetime | None = None,
        taken_before: datetime | None = None,
        limit: int = 2000,
    ) -> list[AssetInfo]:
        """Metadata search (person and/or date range), oldest first."""
        out: list[AssetInfo] = []
        page = 1
        while page and len(out) < limit:
            res = await self._call(
                "search assets",
                self.client.search.search_assets(
                    m.MetadataSearchDto(
                        person_ids=[UUID(p) for p in person_ids] if person_ids else None,
                        taken_after=taken_after, taken_before=taken_before,
                        with_exif=True, with_people=True, size=min(1000, limit), page=page, order=m.AssetOrder.ASC,
                    )
                ),
            )
            out.extend(_to_info(a) for a in res.assets.items)
            page = int(res.assets.next_page) if res.assets.next_page else None
        return out[:limit]

    async def asset(self, asset_id: str) -> AssetInfo:
        return _to_info(await self._call("asset info", self.client.assets.get_asset_info(id=UUID(asset_id))))

    async def faces(self, asset_id: str) -> list[FaceBox]:
        """Face boxes for an asset, as fractions of the image size."""
        faces = await self._call("faces", self.client.faces.get_faces(id=UUID(asset_id)))
        boxes = []
        for f in faces:
            if not f.image_width or not f.image_height:
                continue
            boxes.append(
                FaceBox(
                    x1=f.bounding_box_x1 / f.image_width,
                    y1=f.bounding_box_y1 / f.image_height,
                    x2=f.bounding_box_x2 / f.image_width,
                    y2=f.bounding_box_y2 / f.image_height,
                    person=f.person.name if f.person and f.person.name else None,
                    person_id=str(f.person.id) if f.person else None,
                )
            )
        return boxes

    async def thumbnail(self, asset_id: str, size: str = "preview") -> bytes:
        return await self._call(
            "view asset",
            self.client.assets.view_asset(id=UUID(asset_id), size=m.AssetMediaSize(size)),
        )

    async def people_with_birthdays(self) -> list[tuple[str, str, date]]:
        """(id, name, birth date) for every named person with a birth date set in Immich."""
        out, page = [], 1
        while page:
            res = await self._call("people", self.client.people.get_all_people(size=500, page=page))
            out += [(str(p.id), p.name, p.birth_date) for p in res.people if p.name and p.birth_date]
            page = page + 1 if res.has_next_page else None
        return out

    async def ocr_hits(self, text: str, taken_after: datetime, taken_before: datetime) -> int:
        """How many photos in the window contain this text (Immich OCR, v2.2+)."""
        res = await self._call(
            "ocr search",
            self.client.search.search_assets(
                m.MetadataSearchDto(ocr=text, taken_after=taken_after, taken_before=taken_before, size=3)
            ),
        )
        return len(res.assets.items)

    async def list_people(self) -> list[dict[str, str]]:
        """Named, visible people (for the highlight-video picker)."""
        res = await self._call("people", self.client.people.get_all_people(size=500))
        return sorted(
            ({"id": str(p.id), "name": p.name} for p in res.people if p.name and not p.is_hidden),
            key=lambda p: p["name"].casefold(),
        )

    async def video(self, asset_id: str) -> bytes:
        """The web-playable version of a video (Immich's transcode; smaller than the original)."""
        return await self._call("play video", self.client.assets.play_asset_video(id=UUID(asset_id)))

    async def person_name(self, person_id: str) -> str | None:
        try:
            p = await self._call("person", self.client.people.get_person(id=UUID(person_id)))
        except ImmichError:
            return None
        return p.name or None

    async def original(self, asset_id: str) -> bytes:
        return await self._call("download asset", self.client.assets.download_asset(id=UUID(asset_id)))

    # ---- writing results back ---------------------------------------------------------

    async def upload(
        self,
        data: bytes,
        filename: str,
        *,
        taken_at: datetime,
        job_id: int,
        kind: str,
        source_ids: list[str],
        duration_ms: int | None = None,
    ) -> Upload:
        """Upload a creation, then record where it came from in the asset's metadata.

        The metadata is written with a separate JSON call (PUT /assets/{id}/metadata) because
        immichpy 10.3.0 encodes the multipart `metadata` field as a Python repr, not JSON.
        """
        res = await self._call(
            "upload asset",
            self.client.assets.upload_asset(
                asset_data=(filename, data),
                file_created_at=taken_at,
                file_modified_at=datetime.now(timezone.utc),
                filename=filename,
                duration=duration_ms,
            ),
        )
        status = res.status.value if hasattr(res.status, "value") else str(res.status)
        upload = Upload(asset_id=str(res.id), duplicate=status == "duplicate")
        if not upload.duplicate:
            meta = m.AssetMetadataUpsertItemDto(
                key=CREATION_METADATA_KEY,
                value={"job": job_id, "kind": kind, "sources": source_ids},
            )
            await self._call(
                "set asset metadata",
                self.client.assets.update_asset_metadata(
                    id=res.id, asset_metadata_upsert_dto=m.AssetMetadataUpsertDto(items=[meta])
                ),
            )
        return upload

    async def set_description(self, asset_id: str, text: str) -> None:
        await self._call(
            "update asset",
            self.client.assets.update_asset(id=UUID(asset_id), update_asset_dto=m.UpdateAssetDto(description=text)),
        )

    async def ensure_album(self, name: str) -> str:
        """Return the id of the album with this exact name, creating it when missing."""
        existing = await self._call("find album", self.client.albums.get_all_albums(name=name))
        for album in existing:
            if album.album_name == name:
                return str(album.id)
        created = await self._call(
            "create album",
            self.client.albums.create_album(
                m.CreateAlbumDto(album_name=name, description="Made by the photos companion")
            ),
        )
        return str(created.id)

    async def add_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        await self._call(
            "add to album",
            self.client.albums.add_assets_to_album(
                id=UUID(album_id), bulk_ids_dto=m.BulkIdsDto(ids=[UUID(a) for a in asset_ids])
            ),
        )

    async def tag(self, tag_name: str, asset_ids: list[str]) -> None:
        tags = await self._call("upsert tag", self.client.tags.upsert_tags(m.TagUpsertDto(tags=[tag_name])))
        await self._call(
            "tag assets",
            self.client.tags.bulk_tag_assets(
                m.TagBulkAssetsDto(asset_ids=[UUID(a) for a in asset_ids], tag_ids=[t.id for t in tags])
            ),
        )

    async def stack(self, primary_id: str, other_ids: list[str]) -> str:
        """Stack assets; the first id becomes the stack's primary asset."""
        res = await self._call(
            "create stack",
            self.client.stacks.create_stack(m.StackCreateDto(asset_ids=[UUID(primary_id), *map(UUID, other_ids)])),
        )
        return str(res.id)

    async def add_memory(
        self, asset_ids: list[str], *, year: int, person_id: str | None = None, person_name: str | None = None,
        show_days: int = 7,
    ) -> str:
        """A birthday memory in Immich's Memories strip: Immich labels it with the person's name.

        Only birthday events get a memory. Immich already makes its own "N years ago" memories,
        so other creations go to a named album instead (see Worker._deliver_to_immich).
        """
        now = datetime.now(timezone.utc)
        res = await self._call(
            "create memory",
            self.client.memories.create_memory(
                m.MemoryCreateDto(
                    type=m.MemoryType.BIRTHDAY,
                    data=m.MemoryDataDto(year=year, person_id=UUID(person_id) if person_id else None, person_name=person_name),
                    memory_at=now,
                    show_at=now,
                    hide_at=now + timedelta(days=show_days),
                    asset_ids=[UUID(a) for a in asset_ids],
                )
            ),
        )
        return str(res.id)
