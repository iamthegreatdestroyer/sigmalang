"""
NAS Architecture Evaluator - Phase 7 Track 3

Lightweight evaluator that scores candidate architectures without
full training. Uses proxy metrics:
    1. Compression ratio estimate (based on codebook capacity + layer depth)
    2. Latency estimate (based on FLOPs approximation)
    3. Memory estimate (based on parameter count)
    4. Actual compression test on small benchmark texts

Research Basis:
    - Training-free NAS (Sep 2024): score architectures without training
    - Zero-cost proxies: parameter count, gradient norm, Jacobian metrics

Usage:
    evaluator = ArchitectureEvaluator(benchmark_texts=["hello world", ...])
    score = evaluator.evaluate(config)
    print(f"Pareto score: {score.pareto_score:.4f}")
"""

import time
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from sigmalang.nas.search_space import ArchitectureConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Result
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of evaluating a single architecture."""
    architecture_id: str
    compression_ratio_estimate: float = 0.0
    latency_estimate_ms: float = 0.0
    memory_estimate_kb: float = 0.0
    actual_compression_ratio: float = 0.0
    param_count: int = 0
    complexity_score: float = 0.0
    pareto_score: float = 0.0  # Combined objective
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'architecture_id': self.architecture_id,
            'compression_ratio_estimate': round(self.compression_ratio_estimate, 2),
            'latency_estimate_ms': round(self.latency_estimate_ms, 3),
            'memory_estimate_kb': round(self.memory_estimate_kb, 1),
            'actual_compression_ratio': round(self.actual_compression_ratio, 2),
            'param_count': self.param_count,
            'complexity_score': round(self.complexity_score, 4),
            'pareto_score': round(self.pareto_score, 4),
            'evaluation_time_ms': round(self.evaluation_time_ms, 2),
        }


# =============================================================================
# Evaluator
# =============================================================================

# Proxy metric weights for Pareto scoring
DEFAULT_WEIGHTS = {
    'compression': 0.40,  # Higher compression ratio = better
    'latency': 0.30,      # Lower latency = better
    'memory': 0.15,       # Lower memory = better
    'actual': 0.15,       # Actual compression test bonus
}

# Benchmark corpus for actual compression testing
DEFAULT_BENCHMARK = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "In computer science, data compression reduces the number of bits needed to represent data. " * 5,
    "SigmaLang encodes semantic content using a universal set of 256 primitives that capture meaning. " * 5,
    "Machine learning models require efficient compression for deployment on edge devices. " * 5,
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n" * 8,
]


class ArchitectureEvaluator:
    """
    Evaluate candidate architectures using proxy metrics and
    lightweight actual compression tests.
    """

    def __init__(
        self,
        benchmark_texts: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.benchmark_texts = benchmark_texts or DEFAULT_BENCHMARK
        self.weights = weights or DEFAULT_WEIGHTS
        self._eval_count = 0

    def evaluate(self, config: ArchitectureConfig) -> EvaluationResult:
        """
        Evaluate a single architecture configuration.

        Returns EvaluationResult with proxy and actual metrics.
        """
        start = time.perf_counter()
        self._eval_count += 1

        result = EvaluationResult(
            architecture_id=config.architecture_id or f"arch-{self._eval_count}",
        )

        # 1. Proxy: compression ratio estimate
        result.compression_ratio_estimate = self._estimate_compression_ratio(config)

        # 2. Proxy: latency estimate
        result.latency_estimate_ms = self._estimate_latency(config)

        # 3. Proxy: memory estimate
        result.param_count = config.estimated_params()
        result.memory_estimate_kb = config.estimated_memory_bytes() / 1024

        # 4. Complexity score
        result.complexity_score = config.complexity_score()

        # 5. Actual compression test
        result.actual_compression_ratio = self._run_actual_compression(config)

        # 6. Pareto score
        result.pareto_score = self._compute_pareto_score(result)

        result.evaluation_time_ms = (time.perf_counter() - start) * 1000
        return result

    def evaluate_batch(
        self,
        configs: List[ArchitectureConfig]
    ) -> List[EvaluationResult]:
        """Evaluate multiple architectures."""
        return [self.evaluate(c) for c in configs]

    def _estimate_compression_ratio(self, config: ArchitectureConfig) -> float:
        """
        Proxy estimate of compression ratio based on architecture capacity.

        Deeper + wider networks with larger codebooks tend to achieve
        higher compression. This models the trend without actual encoding.
        """
        # Base ratio from codebook size (log scale)
        cb_factor = math.log2(config.codebook_size) / math.log2(1024)

        # Depth bonus (more layers = better representation = better compression)
        depth_factor = min(1.0, (config.encoder_layers + config.decoder_layers) / 12)

        # Dimension bonus (wider = more capacity)
        dim_factor = min(1.0, config.encoder_hidden_dim / 512)

        # Skip connections help (residual > dense > none)
        skip_bonus = {'none': 0.0, 'residual': 0.1, 'dense': 0.15}
        skip = skip_bonus.get(config.encoder_skip, 0.0)

        # Attention pooling helps
        pool_bonus = {'mean': 0.0, 'max': 0.02, 'attention': 0.1, 'cls_token': 0.08}
        pool = pool_bonus.get(config.encoder_pooling, 0.0)

        # Combine: base * (1 + bonuses)
        base_ratio = 10.0 + 50.0 * cb_factor * (0.3 * depth_factor + 0.4 * dim_factor + 0.3)
        ratio = base_ratio * (1 + skip + pool)

        # Quantization penalty (lower bits = less precise = slightly lower ratio)
        quant_penalty = 1.0 - 0.05 * (32 - config.quantization_bits) / 28
        ratio *= quant_penalty

        return max(1.0, ratio)

    def _estimate_latency(self, config: ArchitectureConfig) -> float:
        """
        Estimate encoding latency in milliseconds.

        Based on approximate FLOPs for the architecture.
        """
        # FLOPs per layer ≈ 2 * hidden_dim^2 (matrix multiply)
        encoder_flops = config.encoder_layers * 2 * config.encoder_hidden_dim ** 2
        decoder_flops = config.decoder_layers * 2 * config.decoder_hidden_dim ** 2

        # Codebook lookup: codebook_size * embedding_dim comparisons
        lookup_flops = config.codebook_size * config.embedding_dim

        total_flops = encoder_flops + decoder_flops + lookup_flops

        # Rough conversion: ~10 GFLOPS on CPU
        latency_s = total_flops / 10e9
        latency_ms = latency_s * 1000

        # Overhead for normalization, activation, etc.
        overhead = 0.1 * config.encoder_layers
        if config.encoder_norm != 'none':
            overhead += 0.05 * config.encoder_layers

        return max(0.01, latency_ms + overhead)

    def _run_actual_compression(self, config: ArchitectureConfig) -> float:
        """
        Run actual compression using a simulated encoder based on the architecture.

        This is a lightweight simulation, not full training. It estimates
        how well the architecture would compress by simulating the information
        bottleneck.
        """
        total_input = 0
        total_output = 0

        for text in self.benchmark_texts:
            text_bytes = len(text.encode('utf-8'))
            total_input += text_bytes

            # Simulate compression: input → encoder → codebook → output
            # Number of codebook lookups needed
            tokens = len(text.split())
            # Deeper networks group more tokens per codebook entry
            group_factor = 1.0 + 0.3 * min(config.encoder_layers, 6)
            entries_needed = max(1, int(tokens / group_factor))

            # Each entry is encoded as log2(codebook_size) bits
            bits_per_entry = math.log2(config.codebook_size)

            # With embedding dim, we can encode more info per entry
            capacity_bonus = min(1.5, config.embedding_dim / 64)
            effective_bits = bits_per_entry * capacity_bonus

            # Output size estimate
            output_bits = entries_needed * effective_bits
            output_bytes = max(1, int(output_bits / 8))

            # Add overhead for metadata
            output_bytes += 8  # header

            total_output += output_bytes

        if total_output == 0:
            return 1.0

        return total_input / total_output

    def _compute_pareto_score(self, result: EvaluationResult) -> float:
        """
        Compute multi-objective Pareto score.

        Higher = better. Combines compression, latency, memory.
        """
        w = self.weights

        # Normalize each objective to [0, 1]
        # Compression: higher is better, cap at 100x
        comp_score = min(1.0, result.compression_ratio_estimate / 100.0)

        # Latency: lower is better, 0.01ms = perfect, 10ms = worst
        lat_score = max(0.0, 1.0 - result.latency_estimate_ms / 10.0)

        # Memory: lower is better, <16KB = perfect, >1MB = worst
        mem_score = max(0.0, 1.0 - result.memory_estimate_kb / 1024.0)

        # Actual compression bonus
        actual_score = min(1.0, result.actual_compression_ratio / 100.0)

        return (
            w['compression'] * comp_score +
            w['latency'] * lat_score +
            w['memory'] * mem_score +
            w['actual'] * actual_score
        )

    @property
    def evaluation_count(self) -> int:
        return self._eval_count
