"""Tests for sigmalang.core.token_recycler.reference_tokens."""

from __future__ import annotations

from sigmalang.core.token_recycler import reference_tokens


def test_module_imports() -> None:
    """reference_tokens is importable and exports its public names."""
    assert hasattr(reference_tokens, "ReferenceTokenManager")
    assert hasattr(reference_tokens, "ReferenceTokenMap")
    assert hasattr(reference_tokens, "MinimalBloomFilter")
