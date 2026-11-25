"""Local model files: downloaded once into <data>/models, pinned by commit and checked by SHA-256.

Only licences that allow this use are listed here (see docs/plan.md).
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFile:
    name: str
    repo: str
    revision: str  # pinned commit
    path: str
    sha256: str | None  # None = record on first download (logged), then pin here
    licence: str

    def url(self, base: str) -> str:
        return f"{base.rstrip('/')}/{self.repo}/resolve/{self.revision}/{self.path}"


DEPTH_ANYTHING_V2_SMALL = ModelFile(
    name="depth-anything-v2-small.onnx",
    repo="onnx-community/depth-anything-v2-small",
    revision="4472b7362082ad9968fee890ca0f1e5aca36b93d",
    path="onnx/model.onnx",
    sha256="afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
    licence="Apache-2.0",
)


MODNET = ModelFile(
    name="modnet.onnx",
    repo="Xenova/modnet",
    revision="fa2fa546052fba4c08921230a26cc69a333fca12",
    path="onnx/model.onnx",
    sha256="07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
    licence="Apache-2.0",
)


class ModelUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(model: ModelFile, models_dir: Path, base_url: str = "https://huggingface.co") -> Path:
    """Return the local path of a model, downloading it on first use."""
    target = models_dir / model.name
    if target.exists():
        return target
    models_dir.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    url = model.url(base_url)
    log.info("Downloading %s (%s) from %s", model.name, model.licence, url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "photos-companion"})
        with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise ModelUnavailable(f"Couldn't download the {model.name} model: {exc}") from exc
    digest = _sha256(part)
    if model.sha256 and digest != model.sha256:
        part.unlink(missing_ok=True)
        raise ModelUnavailable(f"{model.name}: checksum mismatch (got {digest}).")
    if not model.sha256:
        log.warning("%s sha256=%s (pin this in app/models.py)", model.name, digest)
    os.replace(part, target)
    return target
