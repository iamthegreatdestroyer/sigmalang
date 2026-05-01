"""Tests for sigmalang.core.token_recycler.storage."""

from __future__ import annotations

from sigmalang.core.token_recycler import storage


def test_module_imports() -> None:
    """storage is importable and exports its public names."""
    assert hasattr(storage, "RecyclerStorage")
    assert hasattr(storage, "InMemoryStorage")
    assert hasattr(storage, "RedisStorage")
    assert hasattr(storage, "get_storage")
