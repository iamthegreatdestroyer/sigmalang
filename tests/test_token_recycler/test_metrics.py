"""Tests for sigmalang.core.token_recycler.metrics."""

from __future__ import annotations

from sigmalang.core.token_recycler import metrics


def test_module_imports() -> None:
    """metrics is importable and exports its public names."""
    assert hasattr(metrics, "TurnMetrics")
    assert hasattr(metrics, "SessionMetrics")
