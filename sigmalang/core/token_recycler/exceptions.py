"""
Token Recycler typed exception hierarchy.

All exceptions raised by the token_recycler sub-package derive from
TokenRecyclerError so callers can catch the whole family with one clause.
"""

from __future__ import annotations

__all__ = [
    "TokenRecyclerError",
    "CompressionError",
    "ReconstructionError",
    "DriftDetectedError",
    "ProfileNotFoundError",
    "StorageError",
]


class TokenRecyclerError(Exception):
    """Base exception for all token recycler errors."""


class CompressionError(TokenRecyclerError):
    """Raised when compression fails or produces an invalid result."""


class ReconstructionError(TokenRecyclerError):
    """Raised when context reconstruction fails."""


class DriftDetectedError(TokenRecyclerError):
    """Raised when semantic drift exceeds the configured threshold.

    Attributes:
        severity: Fractional drift severity (0.0–1.0).
        action: Recommended action string (e.g. 'FULL_REFRESH').
    """

    def __init__(self, message: str, severity: float = 0.0, action: str = "FULL_REFRESH") -> None:
        super().__init__(message)
        self.severity = severity
        self.action = action


class ProfileNotFoundError(TokenRecyclerError):
    """Raised when no compression profile exists for the requested agent."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"No compression profile found for agent: {agent_id!r}")
        self.agent_id = agent_id


class StorageError(TokenRecyclerError):
    """Raised when a storage backend operation fails."""
