"""
Benchmarking Utilities for HD vs LSH Comparison
===============================================

Phase 2A.1: Comparative analysis framework for hyperdimensional computing
vs locality-sensitive hashing approaches.

Provides:
- Dataset generation (various complexity levels)
- Metric computation (accuracy, speed, compression)
- Result tracking and reporting
- Statistical analysis

Copyright 2025 - Ryot LLM Project
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
from pathlib import Path


class DatasetComplexity(Enum):
    """Dataset complexity levels for benchmarking."""
    SIMPLE = "simple"           # Single nodes, small values
    MODERATE = "moderate"       # Small trees, varied structures
    COMPLEX = "complex"         # Large trees, deep nesting
    EXTREME = "extreme"         # Maximum complexity scenarios


class MetricType(Enum):
    """Types of metrics to track."""
    LATENCY = "latency"                 # ms per operation
    THROUGHPUT = "throughput"           # operations per second
    ACCURACY = "accuracy"               # Semantic correctness (0-1)
    COMPRESSION_RATIO = "compression"   # Original/compressed bytes
    RECALL_AT_K = "recall_at_k"         # ANN search quality
    PRECISION_AT_K = "precision_at_k"   # ANN result quality


@dataclass
class BenchmarkResult:
    """Single benchmark execution result."""
    
    approach: str                           # "HD" or "LSH"
    dataset_size: int                       # Number of semantic trees tested
    dataset_complexity: DatasetComplexity   # Complexity level
    metric_type: MetricType                 # What was measured
    values: List[float]                     # Raw measurement values
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mean(self) -> float:
        """Mean of all measurements."""
        return np.mean(self.values) if self.values else 0.0
    
    @property
    def std(self) -> float:
        """Standard deviation of measurements."""
        return np.std(self.values) if len(self.values) > 1 else 0.0
    
    @property
    def median(self) -> float:
        """Median of measurements."""
        return np.median(self.values) if self.values else 0.0
    
    @property
    def min(self) -> float:
        """Minimum value."""
        return np.min(self.values) if self.values else 0.0
    
    @property
    def max(self) -> float:
        """Maximum value."""
        return np.max(self.values) if self.values else 0.0
    
    @property
    def p95(self) -> float:
        """95th percentile."""
        return np.percentile(self.values, 95) if self.values else 0.0
    
    @property
    def p99(self) -> float:
        """99th percentile."""
        return np.percentile(self.values, 99) if self.values else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approach": self.approach,
            "dataset_size": self.dataset_size,
            "dataset_complexity": self.dataset_complexity.value,
            "metric_type": self.metric_type.value,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "p95": self.p95,
            "p99": self.p99,
            "sample_count": len(self.values),
            "metadata": self.metadata
        }


@dataclass
class ComparativeResult:
    """Comparison between two approaches."""
    
    metric_type: MetricType
    dataset_complexity: DatasetComplexity
    hd_result: BenchmarkResult
    lsh_result: BenchmarkResult
    
    @property
    def hd_better(self) -> bool:
        """Whether HD approach is faster/better."""
        metric = self.metric_type
        
        if metric in [MetricType.LATENCY]:
            # Lower is better
            return self.hd_result.mean < self.lsh_result.mean
        elif metric in [MetricType.ACCURACY, MetricType.RECALL_AT_K, 
                        MetricType.PRECISION_AT_K]:
            # Higher is better
            return self.hd_result.mean > self.lsh_result.mean
        elif metric == MetricType.THROUGHPUT:
            # Higher is better
            return self.hd_result.mean > self.lsh_result.mean
        elif metric == MetricType.COMPRESSION_RATIO:
            # Higher is better (more compression)
            return self.hd_result.mean > self.lsh_result.mean
        else:
            return False
    
    @property
    def improvement_factor(self) -> float:
        """How much better is one approach."""
        metric = self.metric_type
        
        hd_mean = self.hd_result.mean
        lsh_mean = self.lsh_result.mean
        
        if lsh_mean == 0:
            return 0.0
        
        if metric in [MetricType.LATENCY]:
            # Lower is better - ratio of lsh/hd
            if hd_mean == 0:
                return float('inf')
            return lsh_mean / hd_mean
        elif metric in [MetricType.THROUGHPUT, MetricType.ACCURACY,
                        MetricType.RECALL_AT_K, MetricType.PRECISION_AT_K]:
            # Higher is better - ratio of hd/lsh
            if lsh_mean == 0:
                return float('inf')
            return hd_mean / lsh_mean
        else:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_type": self.metric_type.value,
            "dataset_complexity": self.dataset_complexity.value,
            "hd_mean": self.hd_result.mean,
            "lsh_mean": self.lsh_result.mean,
            "hd_better": self.hd_better,
            "improvement_factor": self.improvement_factor,
            "hd_full": self.hd_result.to_dict(),
            "lsh_full": self.lsh_result.to_dict()
        }


class BenchmarkSuite:
    """Manages execution and analysis of benchmarks."""
    
    def __init__(self, name: str, output_dir: Optional[str] = None):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else Path("./benchmark_results")
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparativeResult] = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Record a benchmark result."""
        self.results.append(result)
    
    def run_comparison(
        self,
        hd_func: Callable[[], List[float]],
        lsh_func: Callable[[], List[float]],
        metric_type: MetricType,
        dataset_complexity: DatasetComplexity,
        dataset_size: int,
        iterations: int = 10
    ) -> ComparativeResult:
        """
        Run paired benchmarks comparing two approaches.
        
        Args:
            hd_func: Callable that returns list of measurements for HD approach
            lsh_func: Callable that returns list of measurements for LSH approach
            metric_type: What metric is being measured
            dataset_complexity: Complexity of dataset
            dataset_size: Number of items in dataset
            iterations: How many times to run each
        
        Returns:
            ComparativeResult with both measurements
        """
        hd_values = []
        lsh_values = []
        
        for _ in range(iterations):
            try:
                hd_values.extend(hd_func())
            except Exception as e:
                print(f"HD benchmark error: {e}")
            
            try:
                lsh_values.extend(lsh_func())
            except Exception as e:
                print(f"LSH benchmark error: {e}")
        
        hd_result = BenchmarkResult(
            approach="HD",
            dataset_size=dataset_size,
            dataset_complexity=dataset_complexity,
            metric_type=metric_type,
            values=hd_values,
            metadata={"iterations": iterations}
        )
        
        lsh_result = BenchmarkResult(
            approach="LSH",
            dataset_size=dataset_size,
            dataset_complexity=dataset_complexity,
            metric_type=metric_type,
            values=lsh_values,
            metadata={"iterations": iterations}
        )
        
        self.add_result(hd_result)
        self.add_result(lsh_result)
        
        comparison = ComparativeResult(
            metric_type=metric_type,
            dataset_complexity=dataset_complexity,
            hd_result=hd_result,
            lsh_result=lsh_result
        )
        self.comparisons.append(comparison)
        
        return comparison
    
    def save_results(self) -> Path:
        """Save all results to JSON file."""
        results_data = {
            "suite_name": self.name,
            "timestamp": time.time(),
            "total_benchmarks": len(self.results),
            "total_comparisons": len(self.comparisons),
            "results": [r.to_dict() for r in self.results],
            "comparisons": [c.to_dict() for c in self.comparisons]
        }
        
        output_file = self.output_dir / f"{self.name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return output_file
    
    def print_summary(self) -> str:
        """Generate summary report."""
        summary = f"\n{'='*80}\n"
        summary += f"BENCHMARK SUITE: {self.name}\n"
        summary += f"{'='*80}\n\n"
        
        summary += f"Total Benchmarks: {len(self.results)}\n"
        summary += f"Total Comparisons: {len(self.comparisons)}\n\n"
        
        # Group by metric type
        by_metric = defaultdict(list)
        for comp in self.comparisons:
            by_metric[comp.metric_type].append(comp)
        
        for metric_type, comps in by_metric.items():
            summary += f"\n{metric_type.value.upper()}\n"
            summary += f"{'-'*80}\n"
            
            for comp in comps:
                hd_better = "✓ HD BETTER" if comp.hd_better else "✗ LSH Better"
                summary += (
                    f"  {comp.dataset_complexity.value:12} | "
                    f"HD: {comp.hd_result.mean:10.4f} | "
                    f"LSH: {comp.lsh_result.mean:10.4f} | "
                    f"Factor: {comp.improvement_factor:6.2f}x | {hd_better}\n"
                )
        
        summary += f"\n{'='*80}\n"
        
        return summary
    
    def get_winner_summary(self) -> Dict[str, int]:
        """Count wins by approach."""
        hd_wins = sum(1 for c in self.comparisons if c.hd_better)
        lsh_wins = len(self.comparisons) - hd_wins
        
        return {
            "HD wins": hd_wins,
            "LSH wins": lsh_wins,
            "Total": len(self.comparisons),
            "HD win rate": hd_wins / len(self.comparisons) if self.comparisons else 0.0
        }


class DatasetGenerator:
    """Generate test datasets of varying complexity."""
    
    @staticmethod
    def generate_semantic_trees(
        count: int,
        complexity: DatasetComplexity,
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic semantic trees for benchmarking.
        
        Returns list of dict-based trees with:
        - id: Unique identifier
        - depth: Tree depth
        - node_count: Number of nodes
        - value: Sample string value
        - structure: Tree structure descriptor
        """
        np.random.seed(seed)
        trees = []
        
        params = {
            DatasetComplexity.SIMPLE: {
                "max_depth": 2,
                "max_children": 2,
                "avg_value_len": 10
            },
            DatasetComplexity.MODERATE: {
                "max_depth": 4,
                "max_children": 4,
                "avg_value_len": 50
            },
            DatasetComplexity.COMPLEX: {
                "max_depth": 8,
                "max_children": 8,
                "avg_value_len": 200
            },
            DatasetComplexity.EXTREME: {
                "max_depth": 16,
                "max_children": 16,
                "avg_value_len": 500
            }
        }
        
        param = params[complexity]
        
        for i in range(count):
            depth = np.random.randint(1, param["max_depth"] + 1)
            
            def gen_tree(d: int) -> Dict:
                """Recursively generate tree."""
                num_children = np.random.randint(1, param["max_children"] + 1)
                value_len = np.random.randint(
                    max(1, param["avg_value_len"] // 2),
                    param["avg_value_len"] * 2
                )
                
                # Generate random string
                chars = "abcdefghijklmnopqrstuvwxyz0123456789 "
                value = "".join(np.random.choice(list(chars), size=value_len))
                
                children = []
                if d > 1:
                    children = [gen_tree(d - 1) for _ in range(num_children)]
                
                return {
                    "value": value[:50],  # Limit for readability
                    "children": children
                }
            
            tree = gen_tree(depth)
            
            # Count nodes
            def count_nodes(t: Dict) -> int:
                return 1 + sum(count_nodes(c) for c in t.get("children", []))
            
            trees.append({
                "id": i,
                "tree": tree,
                "depth": depth,
                "node_count": count_nodes(tree),
                "value": tree["value"],
                "complexity": complexity.value
            })
        
        return trees
    
    @staticmethod
    def generate_random_embeddings(
        count: int,
        dimensionality: int = 256,
        seed: int = 42,
        sparse: bool = False,
        sparsity: float = 0.9
    ) -> np.ndarray:
        """
        Generate random embeddings for testing.
        
        Args:
            count: Number of embeddings
            dimensionality: Dimension of each embedding
            seed: Random seed
            sparse: Whether to create sparse vectors
            sparsity: Proportion of zeros (if sparse)
        
        Returns:
            Array of shape (count, dimensionality)
        """
        np.random.seed(seed)
        
        embeddings = np.random.randn(count, dimensionality)
        
        if sparse:
            mask = np.random.random((count, dimensionality)) < sparsity
            embeddings[mask] = 0
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
