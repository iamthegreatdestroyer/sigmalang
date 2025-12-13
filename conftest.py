"""
Root conftest.py - pytest configuration for the sigmalang project.

This file configures pytest to skip doctests in files that have complex
setup requirements. Doctests work best for simple, self-contained examples.
Files with extensive dependencies or stateful setup are better tested via
dedicated unit tests.
"""

# Skip doctests in these files - they require complex setup that doesn't
# work well with doctest's isolated execution model
collect_ignore_glob = [
    # High complexity files with many interdependent classes
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
