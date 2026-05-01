"""Tests for sigmalang.core.token_recycler.embedding_layer."""

from __future__ import annotations

from sigmalang.core.token_recycler import embedding_layer


def test_module_imports() -> None:
    """embedding_layer is importable and exports its public names."""
    assert hasattr(embedding_layer, "SemanticEmbeddingCompressor")
    assert hasattr(embedding_layer, "CompressedEmbeddings")
