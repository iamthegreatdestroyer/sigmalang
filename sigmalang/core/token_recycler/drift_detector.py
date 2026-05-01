"""
Semantic Drift Detection.

Monitors cosine similarity between new turns and the existing compressed
context.  When similarity falls below the agent's threshold the detector
recommends a FULL_REFRESH, PARTIAL_REFRESH, WARN, or NONE action.
"""

from __future__ import annotations

__all__ = [
    "DriftDetectionResult",
    "SemanticDriftDetector",
]


class DriftDetectionResult:
    """Placeholder — implemented in Stage 4."""


class SemanticDriftDetector:
    """Placeholder — implemented in Stage 4."""
