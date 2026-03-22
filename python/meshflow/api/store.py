"""Artifact file storage — save/load/delete/list .msgpack files."""

import re
from pathlib import Path

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class ArtifactStore:
    """Manages compiled artifacts as .msgpack files in a directory."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def _path(self, artifact_id: str) -> Path:
        if not _UUID_RE.match(artifact_id):
            raise ValueError(f"invalid artifact id: {artifact_id!r}")
        return self._base_dir / f"{artifact_id}.msgpack"

    def save(self, artifact_id: str, data: bytes) -> Path:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(artifact_id)
        path.write_bytes(data)
        return path

    def load(self, artifact_id: str) -> bytes:
        path = self._path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        return path.read_bytes()

    def delete(self, artifact_id: str) -> None:
        path = self._path(artifact_id)
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        path.unlink()

    def list(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        return sorted(p.stem for p in self._base_dir.glob("*.msgpack"))

    def exists(self, artifact_id: str) -> bool:
        return self._path(artifact_id).exists()
