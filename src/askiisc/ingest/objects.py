"""Storage for original documents: in memory for tests, MinIO for runs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

from askiisc.common.errors import NotFound


class ObjectStore(Protocol):
    """Keeps document originals by key."""

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Stores an object."""
        ...

    def get(self, key: str) -> bytes:
        """Returns an object's bytes."""
        ...


class InMemoryObjectStore:
    """A dictionary-backed ObjectStore."""

    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str]] = {}
        self._lock = threading.Lock()

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Stores an object."""
        with self._lock:
            self._objects[key] = (data, content_type)

    def get(self, key: str) -> bytes:
        """Returns an object's bytes.

        Raises:
            NotFound: If the key is unknown.
        """
        with self._lock:
            if key not in self._objects:
                raise NotFound(key)
            return self._objects[key][0]

    def keys(self) -> list[str]:
        """Returns every stored key."""
        with self._lock:
            return sorted(self._objects)


class FileObjectStore:
    """An ObjectStore in a directory (MinIO stand-in on the CSA server).

    MinIO stopped publishing community binaries, so originals are kept as
    files under the run's state directory. Keys map to relative paths.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root) or path == self._root:
            raise ValueError(f"unsafe object key: {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Stores an object (the content type is kept in a sidecar file)."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.with_name(path.name + ".type").write_text(content_type)

    def get(self, key: str) -> bytes:
        """Returns an object's bytes.

        Raises:
            NotFound: If the key is unknown.
        """
        path = self._path(key)
        if not path.is_file():
            raise NotFound(key)
        return path.read_bytes()
