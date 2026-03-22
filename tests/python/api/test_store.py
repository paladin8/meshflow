"""Tests for the artifact file store."""

from pathlib import Path

import pytest

from meshflow.api.store import ArtifactStore

# Valid UUIDs for testing
ID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
ID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class TestArtifactStore:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        data = b"hello artifact"
        store.save(ID_A, data)
        assert store.load(ID_A) == data

    def test_list_returns_stored_ids(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(ID_B, b"2")
        store.save(ID_A, b"1")
        assert store.list() == [ID_A, ID_B]

    def test_list_empty(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.list() == []

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(ID_A, b"data")
        store.delete(ID_A)
        assert not store.exists(ID_A)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="artifact not found"):
            store.load(ID_A)

    def test_delete_missing_raises(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(FileNotFoundError, match="artifact not found"):
            store.delete(ID_A)

    def test_exists_true(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(ID_A, b"data")
        assert store.exists(ID_A) is True

    def test_exists_false(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        assert store.exists(ID_A) is False

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "dir"
        store = ArtifactStore(nested)
        store.save(ID_A, b"data")
        assert store.load(ID_A) == b"data"

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        store.save(ID_A, b"old")
        store.save(ID_A, b"new")
        assert store.load(ID_A) == b"new"

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="invalid artifact id"):
            store.load("../../../etc/passwd")

    def test_rejects_non_uuid(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path)
        with pytest.raises(ValueError, match="invalid artifact id"):
            store.save("not-a-uuid", b"data")
