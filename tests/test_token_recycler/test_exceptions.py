"""Tests for sigmalang.core.token_recycler.exceptions."""

from __future__ import annotations

import pytest

from sigmalang.core.token_recycler.exceptions import (
    CompressionError,
    DriftDetectedError,
    ProfileNotFoundError,
    ReconstructionError,
    StorageError,
    TokenRecyclerError,
)


def test_module_imports() -> None:
    """All exception classes are importable."""
    assert TokenRecyclerError
    assert CompressionError
    assert ReconstructionError
    assert DriftDetectedError
    assert ProfileNotFoundError
    assert StorageError


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_all_exceptions_derive_from_base() -> None:
    for cls in (
        CompressionError,
        ReconstructionError,
        DriftDetectedError,
        ProfileNotFoundError,
        StorageError,
    ):
        assert issubclass(cls, TokenRecyclerError), f"{cls} must derive from TokenRecyclerError"


def test_base_derives_from_exception() -> None:
    assert issubclass(TokenRecyclerError, Exception)


# ---------------------------------------------------------------------------
# Raise / catch
# ---------------------------------------------------------------------------


def test_token_recycler_error_raise() -> None:
    with pytest.raises(TokenRecyclerError, match="base error"):
        raise TokenRecyclerError("base error")


def test_compression_error_raise() -> None:
    with pytest.raises(CompressionError):
        raise CompressionError("compression failed")


def test_compression_error_caught_as_base() -> None:
    with pytest.raises(TokenRecyclerError):
        raise CompressionError("also caught as base")


def test_reconstruction_error_raise() -> None:
    with pytest.raises(ReconstructionError):
        raise ReconstructionError("bad delta")


def test_storage_error_raise() -> None:
    with pytest.raises(StorageError):
        raise StorageError("redis down")


# ---------------------------------------------------------------------------
# DriftDetectedError attributes
# ---------------------------------------------------------------------------


def test_drift_detected_error_defaults() -> None:
    err = DriftDetectedError("drift!")
    assert err.severity == 0.0
    assert err.action == "FULL_REFRESH"
    assert str(err) == "drift!"


def test_drift_detected_error_custom_attrs() -> None:
    err = DriftDetectedError("severe drift", severity=0.45, action="PARTIAL_REFRESH")
    assert err.severity == pytest.approx(0.45)
    assert err.action == "PARTIAL_REFRESH"


def test_drift_detected_error_caught_as_base() -> None:
    with pytest.raises(TokenRecyclerError):
        raise DriftDetectedError("oops", severity=0.9, action="FULL_REFRESH")


# ---------------------------------------------------------------------------
# ProfileNotFoundError attributes
# ---------------------------------------------------------------------------


def test_profile_not_found_stores_agent_id() -> None:
    err = ProfileNotFoundError("MY_AGENT")
    assert err.agent_id == "MY_AGENT"
    assert "MY_AGENT" in str(err)


def test_profile_not_found_caught_as_base() -> None:
    with pytest.raises(TokenRecyclerError):
        raise ProfileNotFoundError("UNKNOWN")


def test_profile_not_found_is_not_compression_error() -> None:
    err = ProfileNotFoundError("X")
    assert not isinstance(err, CompressionError)
