"""Tests for sigmalang.core.token_recycler.recycler."""

from __future__ import annotations

from sigmalang.core.token_recycler import recycler


def test_module_imports() -> None:
    """recycler is importable and exports its public names."""
    assert hasattr(recycler, "TokenRecycler")
    assert hasattr(recycler, "CompressedContext")
