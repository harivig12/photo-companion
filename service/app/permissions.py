"""Immich API key permissions the companion needs.

Taken from the `x-immich-permission` values in the Immich v3.4.2 OpenAPI spec.
"""

from __future__ import annotations

from collections.abc import Iterable

# Grouped by what they are used for, so the UI can explain a missing permission.
REQUIRED: dict[str, tuple[str, ...]] = {
    "read photos": ("asset.read", "asset.view", "asset.download", "album.read", "person.read", "face.read", "tag.read"),
    "upload results": ("asset.upload", "asset.update"),
    "creations album": ("album.create", "albumAsset.create"),
    "tags": ("tag.create", "tag.asset"),
    "stack with source": ("stack.create",),
    "memories strip": ("memory.create", "memoryAsset.create"),
}

ALL_REQUIRED: tuple[str, ...] = tuple(dict.fromkeys(p for group in REQUIRED.values() for p in group))


def missing_permissions(granted: Iterable[str]) -> list[str]:
    """Return the required permissions not covered by `granted` (an 'all' key covers everything)."""
    granted_set = set(granted)
    if "all" in granted_set:
        return []
    return [p for p in ALL_REQUIRED if p not in granted_set]


def affected_features(missing: Iterable[str]) -> list[str]:
    """Names of the feature groups that break because of the missing permissions."""
    missing_set = set(missing)
    return [name for name, perms in REQUIRED.items() if missing_set.intersection(perms)]
