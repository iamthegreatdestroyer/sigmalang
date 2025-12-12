"""
ΣLANG Test Configuration & Fixtures
====================================

Provides shared test fixtures, configuration, and utilities for all test suites.

This module sets up:
- Test data generators
- Semantic tree fixtures
- Encoder/decoder test harnesses
- Performance benchmarking helpers
- Comparison utilities

Copyright 2025 - Ryot LLM Project
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import json

# Add parent to path for imports
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root.parent))

from sigmalang.core.primitives import (
    SemanticNode, SemanticTree, Glyph, GlyphStream, GlyphType,
    ExistentialPrimitive, CodePrimitive, ActionPrimitive,
    EntityPrimitive, PRIMITIVE_REGISTRY
)
from sigmalang.core.parser import SemanticParser
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
from sigmalang.training.codebook import LearnedCodebook, CodebookTrainer


# ============================================================================
# SEMANTIC TREE GENERATORS
# ============================================================================

class SemanticTreeBuilder:
    """Helper for building test semantic trees."""
    
    @staticmethod
    def simple_tree() -> SemanticTree:
        """Create a simple 2-level semantic tree."""
        root = SemanticNode(
            primitive=ExistentialPrimitive.ACTION,
            value="create",
            children=[
                SemanticNode(
                    primitive=CodePrimitive.FUNCTION,
                    value="sort_list"
                ),
                SemanticNode(
                    primitive=ExistentialPrimitive.ATTRIBUTE,
                    value="descending"
                )
            ]
        )
        return SemanticTree(root=root, source_text="Create a function that sorts a list")
    
    @staticmethod
    def complex_tree() -> SemanticTree:
        """Create a deeper, more complex semantic tree."""
        root = SemanticNode(
            primitive=ExistentialPrimitive.ACTION,
            value="implement",
            children=[
                SemanticNode(
                    primitive=CodePrimitive.CLASS,
                    value="DataProcessor",
                    children=[
                        SemanticNode(
                            primitive=CodePrimitive.FUNCTION,
                            value="__init__",
                            children=[
                                SemanticNode(
                                    primitive=CodePrimitive.PARAMETER,
                                    value="self"
                                ),
                                SemanticNode(
                                    primitive=CodePrimitive.PARAMETER,
                                    value="input_size"
                                )
                            ]
                        ),
                        SemanticNode(
                            primitive=CodePrimitive.FUNCTION,
                            value="process",
                            children=[
                                SemanticNode(
                                    primitive=ExistentialPrimitive.ENTITY,
                                    value="data"
                                ),
                                SemanticNode(
                                    primitive=ExistentialPrimitive.ATTRIBUTE,
                                    value="efficiently"
                                )
                            ]
                        )
                    ]
                ),
                SemanticNode(
                    primitive=ExistentialPrimitive.ATTRIBUTE,
                    value="error_handling"
                )
            ]
        )
        return SemanticTree(root=root, source_text="Implement a DataProcessor class with error handling")
    
    @staticmethod
    def random_tree(
        depth: int = 4,
        avg_branching: float = 2.5,
        seed: int = 42
    ) -> SemanticTree:
        """Generate random semantic tree for stress testing."""
        np.random.seed(seed)
        primitives = list(ExistentialPrimitive) + list(CodePrimitive)
        
        def build_random_node(current_depth: int) -> SemanticNode:
            primitive = primitives[np.random.randint(0, len(primitives))]
            value = f"value_{np.random.randint(0, 10000)}"
            
            children = []
            if current_depth < depth:
                num_children = max(0, int(np.random.exponential(avg_branching - 1)))
                for _ in range(num_children):
                    children.append(build_random_node(current_depth + 1))
            
            return SemanticNode(primitive=primitive, value=value, children=children)
        
        root = build_random_node(0)
        return SemanticTree(root=root, source_text="random_input")


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def simple_semantic_tree() -> SemanticTree:
    """Fixture: Simple semantic tree for basic tests."""
    return SemanticTreeBuilder.simple_tree()


@pytest.fixture
def complex_semantic_tree() -> SemanticTree:
    """Fixture: Complex semantic tree for advanced tests."""
    return SemanticTreeBuilder.complex_tree()


@pytest.fixture
def semantic_encoder() -> SigmaEncoder:
    """Fixture: Initialized ΣLANG encoder."""
    return SigmaEncoder()


@pytest.fixture
def semantic_decoder() -> SigmaDecoder:
    """Fixture: Initialized ΣLANG decoder."""
    return SigmaDecoder()


@pytest.fixture
def semantic_parser() -> SemanticParser:
    """Fixture: Initialized semantic parser."""
    return SemanticParser()


@pytest.fixture
def learned_codebook() -> LearnedCodebook:
    """Fixture: Fresh learned codebook."""
    return LearnedCodebook()


@pytest.fixture
def codebook_trainer() -> CodebookTrainer:
    """Fixture: Codebook trainer instance."""
    return CodebookTrainer()


# ============================================================================
# TEST DATA COLLECTIONS
# ============================================================================

@dataclass
class CompressionTestCase:
    """A single compression test case."""
    name: str
    input_text: str
    expected_ratio_range: Tuple[float, float]
    should_succeed: bool = True
    description: str = ""


class TestDatasets:
    """Collections of test data for various scenarios."""
    
    CODE_SNIPPETS = [
        "Create a Python function that sorts a list in descending order",
        "Write a JavaScript async function to fetch data from an API",
        "Implement a binary search algorithm in C++",
        "Build a REST API endpoint for user authentication",
        "Create a class that handles database connections with pooling",
        "Write a function to validate email addresses using regex",
        "Implement a cache with LRU eviction policy",
        "Create a decorator that logs function execution time",
        "Write a function to merge two sorted arrays efficiently",
        "Implement a trie data structure for prefix matching",
    ]
    
    QUERIES = [
        "What is the time complexity of quicksort?",
        "How do I handle exceptions in Python?",
        "What are the best practices for API design?",
        "Explain how async/await works in JavaScript",
        "What is the difference between SQL and NoSQL?",
        "How does garbage collection work in Java?",
        "What is dependency injection and why is it useful?",
        "Explain the singleton pattern and its trade-offs",
        "What is the difference between a list and a tuple?",
        "How do I optimize database queries?",
    ]
    
    EXPLANATIONS = [
        "Explain how this algorithm works step by step",
        "Walk me through this code and explain each function",
        "Describe how the OAuth2 authentication flow works",
        "Explain the concept of functional programming",
        "Describe the MVC architectural pattern",
        "Explain how REST APIs work",
        "Describe the differences between HTTP and HTTPS",
        "Explain how containers work and why they're useful",
        "Describe the event loop in JavaScript",
        "Explain how DNS resolution works",
    ]
    
    MODIFICATIONS = [
        "Refactor this code to use async/await",
        "Optimize this algorithm for better performance",
        "Add error handling to this function",
        "Update this code to use the new API",
        "Fix the memory leak in this component",
        "Convert this class to use dependency injection",
        "Modify this function to support pagination",
        "Fix the bug that causes infinite recursion",
        "Update this code to handle edge cases",
        "Refactor this to use design patterns",
    ]
    
    @classmethod
    def all_inputs(cls) -> List[str]:
        """Return all test inputs."""
        return (
            cls.CODE_SNIPPETS +
            cls.QUERIES +
            cls.EXPLANATIONS +
            cls.MODIFICATIONS
        )
    
    @classmethod
    def get_test_cases(cls) -> List[CompressionTestCase]:
        """Generate test cases with expected compression ranges."""
        return [
            CompressionTestCase(
                name=f"code_snippet_{i}",
                input_text=snippet,
                expected_ratio_range=(0.2, 0.8),  # 1.25x to 5x compression
                description="Code generation request"
            )
            for i, snippet in enumerate(cls.CODE_SNIPPETS)
        ] + [
            CompressionTestCase(
                name=f"query_{i}",
                input_text=query,
                expected_ratio_range=(0.3, 0.9),
                description="Technical query"
            )
            for i, query in enumerate(cls.QUERIES)
        ] + [
            CompressionTestCase(
                name=f"explanation_{i}",
                input_text=explanation,
                expected_ratio_range=(0.25, 0.85),
                description="Explanation request"
            )
            for i, explanation in enumerate(cls.EXPLANATIONS)
        ] + [
            CompressionTestCase(
                name=f"modification_{i}",
                input_text=modification,
                expected_ratio_range=(0.3, 0.9),
                description="Code modification request"
            )
            for i, modification in enumerate(cls.MODIFICATIONS)
        ]


# ============================================================================
# COMPARISON & VALIDATION UTILITIES
# ============================================================================

class TreeComparator:
    """Utilities for comparing semantic trees."""
    
    @staticmethod
    def trees_equal(tree1: SemanticTree, tree2: SemanticTree) -> bool:
        """Check if two semantic trees are structurally equivalent."""
        return TreeComparator._nodes_equal(tree1.root, tree2.root)
    
    @staticmethod
    def _nodes_equal(node1: SemanticNode, node2: SemanticNode) -> bool:
        """Recursively compare semantic nodes."""
        if node1.primitive != node2.primitive:
            return False
        if node1.value != node2.value:
            return False
        if len(node1.children) != len(node2.children):
            return False
        
        return all(
            TreeComparator._nodes_equal(c1, c2)
            for c1, c2 in zip(node1.children, node2.children)
        )
    
    @staticmethod
    def get_differences(
        tree1: SemanticTree,
        tree2: SemanticTree
    ) -> List[str]:
        """Get list of differences between trees."""
        differences = []
        
        def compare_nodes(n1: SemanticNode, n2: SemanticNode, path: str = "root"):
            if n1.primitive != n2.primitive:
                differences.append(
                    f"At {path}: primitive mismatch "
                    f"({n1.primitive} vs {n2.primitive})"
                )
            if n1.value != n2.value:
                differences.append(
                    f"At {path}: value mismatch ({n1.value} vs {n2.value})"
                )
            
            if len(n1.children) != len(n2.children):
                differences.append(
                    f"At {path}: child count mismatch "
                    f"({len(n1.children)} vs {len(n2.children)})"
                )
            else:
                for i, (c1, c2) in enumerate(zip(n1.children, n2.children)):
                    compare_nodes(c1, c2, f"{path}.child[{i}]")
        
        compare_nodes(tree1.root, tree2.root)
        return differences


class CompressionAnalyzer:
    """Analyze compression results."""
    
    @staticmethod
    def compute_ratio(original_size: int, compressed_size: int) -> float:
        """Compute compression ratio (0.0 = best, 1.0+ = no/negative compression)."""
        if original_size == 0:
            return 0.0
        return compressed_size / original_size
    
    @staticmethod
    def analyze_result(
        original_text: str,
        encoded_bytes: bytes,
        decoded_text: str = None
    ) -> Dict[str, Any]:
        """Comprehensive compression result analysis."""
        original_size = len(original_text.encode('utf-8'))
        compressed_size = len(encoded_bytes)
        ratio = CompressionAnalyzer.compute_ratio(original_size, compressed_size)
        
        result = {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': ratio,
            'compression_factor': original_size / compressed_size if compressed_size > 0 else 0,
            'bytes_saved': original_size - compressed_size,
            'percent_reduction': (1 - ratio) * 100,
        }
        
        if decoded_text is not None:
            decoded_size = len(decoded_text.encode('utf-8'))
            result['decoded_size_bytes'] = decoded_size
            result['round_trip_successful'] = decoded_text == original_text
        
        return result


# ============================================================================
# PYTEST CONFIGURATION & HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "round_trip: marks tests as round-trip validation"
    )


@pytest.fixture(scope="session")
def test_data_collection() -> TestDatasets:
    """Fixture: Test data collection for all tests."""
    return TestDatasets()


@pytest.fixture
def compression_results_dir(tmp_path):
    """Fixture: Temporary directory for compression test results."""
    results_dir = tmp_path / "compression_results"
    results_dir.mkdir()
    return results_dir
