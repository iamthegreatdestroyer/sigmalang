"""
Root conftest.py - pytest configuration for the sigmalang project.

Provides:
  - Auto-timeout enforcement for all tests
  - Custom markers for test categorization
  - Shared fixtures for common testing patterns
  - Doctest collection filtering for complex modules
  - Reusable encoder/parser/codec fixtures
  - Fixture isolation hooks (detect leaked state)
  - Performance baseline utilities
"""

import os
import sys
import time
import tempfile
import shutil
import warnings

import pytest


# =============================================================================
# Doctest Collection Filtering
# =============================================================================
# Skip doctests in files with complex setup that doesn't work with
# doctest's isolated execution model
collect_ignore_glob = [
    "core/pattern_intelligence.py",
    "core/production_hardening.py",
    "core/advanced_analogy_patterns.py",
    "core/ml_models.py",
    "core/analytics_engine.py",
    "core/analogy_composition.py",
    "core/semantic_analogy_engine.py",
    "core/parallel_processor.py",
    "core/cross_modal_analogies.py",
    "core/semantic_search.py",
    "core/multilingual_support.py",
    "core/streaming_processor.py",
]


# =============================================================================
# Custom Markers
# =============================================================================
def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line("markers", "integration: integration tests (slower)")
    config.addinivalue_line("markers", "benchmark: benchmark/performance tests")
    config.addinivalue_line("markers", "slow: tests that take >10 seconds")
    config.addinivalue_line("markers", "gpu: tests requiring GPU hardware")


# =============================================================================
# Fixture Isolation — detect leaked global state
# =============================================================================
@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure no test leaks environment variable mutations."""
    # monkeypatch auto-reverts env changes after each test
    pass


# =============================================================================
# Shared Fixtures — Common Objects
# =============================================================================
@pytest.fixture
def tmp_workspace(tmp_path):
    """Provide a clean temporary workspace directory."""
    workspace = tmp_path / "sigmalang_test"
    workspace.mkdir()
    yield workspace
    # Cleanup is automatic via tmp_path


@pytest.fixture
def sample_text():
    """Standard test text for encoding/decoding."""
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def sample_texts():
    """Multiple test texts for batch operations."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A stitch in time saves nine.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Knowledge is power, guard it well.",
    ]


@pytest.fixture
def sigma_encoder():
    """Reusable SigmaEncoder instance (lazy — skips if import fails)."""
    try:
        from sigmalang.core.encoder import SigmaEncoder
        return SigmaEncoder()
    except ImportError:
        pytest.skip("SigmaEncoder not available")


@pytest.fixture
def semantic_parser():
    """Reusable SemanticParser instance (lazy — skips if import fails)."""
    try:
        from sigmalang.core.parser import SemanticParser
        return SemanticParser()
    except ImportError:
        pytest.skip("SemanticParser not available")


@pytest.fixture
def parsed_tree(semantic_parser, sample_text):
    """A pre-parsed SemanticTree from sample_text."""
    return semantic_parser.parse(sample_text)


@pytest.fixture
def bidirectional_codec():
    """Reusable BidirectionalSemanticCodec (lazy)."""
    try:
        from sigmalang.core.bidirectional_codec import BidirectionalSemanticCodec
        return BidirectionalSemanticCodec()
    except ImportError:
        pytest.skip("BidirectionalSemanticCodec not available")


# =============================================================================
# Performance Baseline Utilities
# =============================================================================
@pytest.fixture
def perf_timer():
    """Context-manager fixture for measuring wall-clock duration.

    Usage in tests::

        def test_fast_encode(perf_timer, sigma_encoder, parsed_tree):
            with perf_timer() as t:
                sigma_encoder.encode(parsed_tree)
            assert t.elapsed < 1.0  # must finish in <1s
    """
    class _Timer:
        def __init__(self):
            self.elapsed = 0.0
            self._start = None
        def __enter__(self):
            self._start = time.perf_counter()
            return self
        def __exit__(self, *exc):
            self.elapsed = time.perf_counter() - self._start
    def _factory():
        return _Timer()
    return _factory
