"""
Token Recycler Prometheus metrics and per-turn metrics dataclass.

All metrics are registered with the global sigmalang MetricsRegistry so
they appear alongside existing sigmalang_* metrics in /metrics.
"""

from __future__ import annotations

__all__ = [
    "TurnMetrics",
    "SessionMetrics",
    "RECYCLER_COMPRESSION_RATIO",
    "RECYCLER_SEMANTIC_SIMILARITY",
    "RECYCLER_DRIFT_EVENTS",
    "RECYCLER_COMPRESSION_DURATION",
    "RECYCLER_TOKENS_SAVED",
]


class TurnMetrics:
    """Placeholder — implemented in Stage 4."""


class SessionMetrics:
    """Placeholder — implemented in Stage 4."""


# Prometheus metric placeholders — replaced in Stage 4
RECYCLER_COMPRESSION_RATIO = None
RECYCLER_SEMANTIC_SIMILARITY = None
RECYCLER_DRIFT_EVENTS = None
RECYCLER_COMPRESSION_DURATION = None
RECYCLER_TOKENS_SAVED = None
