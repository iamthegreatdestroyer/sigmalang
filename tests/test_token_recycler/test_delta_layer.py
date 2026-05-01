"""Tests for sigmalang.core.token_recycler.delta_layer."""

from __future__ import annotations

from sigmalang.core.token_recycler import delta_layer


def test_module_imports() -> None:
    """delta_layer is importable and exports its public names."""
    assert hasattr(delta_layer, "DifferentialUpdateExtractor")
    assert hasattr(delta_layer, "DifferentialUpdate")
    assert hasattr(delta_layer, "CountMinSketch")
