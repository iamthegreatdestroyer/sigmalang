"""Tests for sigmalang.core.token_recycler.drift_detector."""

from __future__ import annotations

from sigmalang.core.token_recycler import drift_detector


def test_module_imports() -> None:
    """drift_detector is importable and exports its public names."""
    assert hasattr(drift_detector, "SemanticDriftDetector")
    assert hasattr(drift_detector, "DriftDetectionResult")
