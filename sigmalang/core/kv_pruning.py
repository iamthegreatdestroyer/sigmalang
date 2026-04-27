"""
KV-Cache Pruning Strategies - Phase 7 Track 5

Attention-score-based pruning algorithms for KV cache entries.
Implements multiple eviction policies that can be composed.

Strategies:
    1. Heavy Hitter: Keep tokens with highest cumulative attention
    2. Sliding Window + Sink: Keep recent + initial attention sinks
    3. Adaptive Budget: Layer-wise budget allocation based on attention patterns
    4. Similarity Dedup: Merge near-duplicate KV entries

Research Basis:
    - KVzap (Jan 2026): Input-adaptive pruning without calibration
    - EMS (Dec 2024): Evict-then-merge with head-wise parallel compression
    - GraphKV (Aug 2025): Graph-based importance propagation

Usage:
    from sigmalang.core.kv_pruning import HeavyHitterPruner, SlidingWindowPruner

    pruner = HeavyHitterPruner(keep_ratio=0.3)
    mask = pruner.compute_mask(attention_scores, seq_len=1024)
    pruned_keys = keys[mask]
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Base Pruner
# =============================================================================

class BasePruner:
    """Base class for KV cache pruning strategies."""

    def compute_mask(
        self,
        attention_scores: np.ndarray,
        seq_len: int,
        **kwargs
    ) -> np.ndarray:
        """
        Compute a boolean mask indicating which tokens to keep.

        Args:
            attention_scores: (seq_len,) accumulated attention per token
            seq_len: Total sequence length

        Returns:
            (seq_len,) boolean mask (True = keep)
        """
        raise NotImplementedError

    def prune(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        attention_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prune KV cache tensors.

        Returns: (pruned_keys, pruned_values, kept_indices)
        """
        mask = self.compute_mask(attention_scores, keys.shape[0])
        indices = np.where(mask)[0]
        return keys[indices], values[indices], indices


# =============================================================================
# Heavy Hitter Pruner
# =============================================================================

class HeavyHitterPruner(BasePruner):
    """
    Keep tokens with the highest accumulated attention scores.
    Simple and effective — the "heavy hitters" that get attended to most.
    """

    def __init__(self, keep_ratio: float = 0.3, protect_first: int = 4, protect_last: int = 32):
        self.keep_ratio = keep_ratio
        self.protect_first = protect_first
        self.protect_last = protect_last

    def compute_mask(
        self,
        attention_scores: np.ndarray,
        seq_len: int,
        **kwargs
    ) -> np.ndarray:
        mask = np.zeros(seq_len, dtype=bool)

        # Always keep protected positions
        mask[:min(self.protect_first, seq_len)] = True
        mask[max(0, seq_len - self.protect_last):] = True

        # Budget for non-protected tokens
        already_kept = mask.sum()
        budget = max(0, int(seq_len * self.keep_ratio) - already_kept)

        if budget > 0:
            # Score non-protected tokens
            scores = attention_scores.copy()
            scores[mask] = -np.inf  # Exclude already-kept

            # Select top-budget by score
            top_indices = np.argsort(scores)[-budget:]
            mask[top_indices] = True

        return mask


# =============================================================================
# Sliding Window + Attention Sink Pruner
# =============================================================================

class SlidingWindowPruner(BasePruner):
    """
    Keep a sliding window of recent tokens plus "attention sinks"
    (initial tokens that accumulate high attention in autoregressive models).

    Based on StreamingLLM observation that first few tokens serve as
    attention sinks regardless of content.
    """

    def __init__(
        self,
        window_size: int = 256,
        sink_size: int = 4,
        extra_heavy_hitters: int = 64
    ):
        self.window_size = window_size
        self.sink_size = sink_size
        self.extra_heavy_hitters = extra_heavy_hitters

    def compute_mask(
        self,
        attention_scores: np.ndarray,
        seq_len: int,
        **kwargs
    ) -> np.ndarray:
        mask = np.zeros(seq_len, dtype=bool)

        # Attention sinks (first N tokens)
        mask[:min(self.sink_size, seq_len)] = True

        # Sliding window (last N tokens)
        window_start = max(0, seq_len - self.window_size)
        mask[window_start:] = True

        # Extra heavy hitters from the middle
        if self.extra_heavy_hitters > 0:
            middle_mask = ~mask
            middle_scores = attention_scores.copy()
            middle_scores[~middle_mask] = -np.inf

            if middle_mask.any():
                budget = min(self.extra_heavy_hitters, middle_mask.sum())
                top_indices = np.argsort(middle_scores)[-budget:]
                mask[top_indices] = True

        return mask


# =============================================================================
# Adaptive Layer Budget Pruner
# =============================================================================

class AdaptiveBudgetPruner(BasePruner):
    """
    Allocate different token budgets per layer based on attention entropy.

    Layers with high attention entropy (spread attention) need more tokens.
    Layers with low entropy (focused attention) can work with fewer tokens.

    Inspired by DynamicKV (Dec 2024).
    """

    def __init__(
        self,
        total_budget_ratio: float = 0.3,
        min_layer_ratio: float = 0.1,
        max_layer_ratio: float = 0.8
    ):
        self.total_budget_ratio = total_budget_ratio
        self.min_layer_ratio = min_layer_ratio
        self.max_layer_ratio = max_layer_ratio

    def compute_layer_budgets(
        self,
        layer_attention_entropies: List[float],
        seq_len: int
    ) -> List[int]:
        """
        Compute per-layer token budgets based on attention entropy.

        Args:
            layer_attention_entropies: Entropy of attention distribution per layer
            seq_len: Sequence length

        Returns:
            Per-layer token budget
        """
        total_budget = int(seq_len * self.total_budget_ratio)
        n_layers = len(layer_attention_entropies)

        if n_layers == 0:
            return []

        # Normalize entropies to budget allocation weights
        entropies = np.array(layer_attention_entropies, dtype=np.float32)
        if entropies.max() > entropies.min():
            weights = (entropies - entropies.min()) / (entropies.max() - entropies.min())
        else:
            weights = np.ones(n_layers) / n_layers

        # Clamp to min/max ratios
        weights = np.clip(weights, self.min_layer_ratio, self.max_layer_ratio)
        weights /= weights.sum()

        # Allocate budget
        budgets = (weights * total_budget).astype(int)

        # Ensure at least min tokens per layer
        min_tokens = max(4, int(seq_len * self.min_layer_ratio))
        budgets = np.maximum(budgets, min_tokens)

        return budgets.tolist()

    def compute_mask(
        self,
        attention_scores: np.ndarray,
        seq_len: int,
        **kwargs
    ) -> np.ndarray:
        """Single-layer mask using total budget."""
        budget = int(seq_len * self.total_budget_ratio)
        budget = max(budget, 4)

        mask = np.zeros(seq_len, dtype=bool)
        top_indices = np.argsort(attention_scores)[-budget:]
        mask[top_indices] = True
        return mask


# =============================================================================
# Similarity Dedup Pruner
# =============================================================================

class SimilarityDedupPruner(BasePruner):
    """
    Merge near-duplicate KV entries by averaging similar neighbors.
    Reduces redundancy in KV cache from repetitive content.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        keep_ratio: float = 0.5
    ):
        self.similarity_threshold = similarity_threshold
        self.keep_ratio = keep_ratio

    def compute_mask(
        self,
        attention_scores: np.ndarray,
        seq_len: int,
        keys: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        mask = np.ones(seq_len, dtype=bool)

        if keys is None or seq_len < 2:
            return mask

        # Compute pairwise cosine similarities for adjacent tokens
        norms = np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
        normalized = keys / norms

        # Check adjacent pairs for near-duplicates
        for i in range(seq_len - 1):
            if not mask[i]:
                continue
            sim = float(np.dot(normalized[i], normalized[i + 1]))
            if sim > self.similarity_threshold:
                # Keep the one with higher attention, evict the other
                if attention_scores[i] >= attention_scores[i + 1]:
                    mask[i + 1] = False
                else:
                    mask[i] = False

        # If we've removed too few, do a second pass with budget
        kept = mask.sum()
        budget = int(seq_len * self.keep_ratio)
        if kept > budget:
            # Further prune by attention score
            kept_indices = np.where(mask)[0]
            kept_scores = attention_scores[kept_indices]
            remove_count = kept - budget
            lowest = np.argsort(kept_scores)[:remove_count]
            for idx in lowest:
                mask[kept_indices[idx]] = False

        return mask

    def merge_duplicates(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        attention_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Merge near-duplicate KV entries by averaging.

        Returns merged keys, values, and new attention scores.
        """
        seq_len = keys.shape[0]
        if seq_len < 2:
            return keys, values, attention_scores

        norms = np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
        normalized = keys / norms

        merged_keys = []
        merged_values = []
        merged_scores = []
        i = 0

        while i < seq_len:
            # Find run of similar tokens
            j = i + 1
            while j < seq_len:
                sim = float(np.dot(normalized[i], normalized[j]))
                if sim < self.similarity_threshold:
                    break
                j += 1

            # Average the run
            merged_keys.append(keys[i:j].mean(axis=0))
            merged_values.append(values[i:j].mean(axis=0))
            merged_scores.append(attention_scores[i:j].max())

            i = j

        return (
            np.stack(merged_keys),
            np.stack(merged_values),
            np.array(merged_scores, dtype=np.float32)
        )


# =============================================================================
# Composed Pruner
# =============================================================================

class ComposedPruner:
    """
    Compose multiple pruning strategies sequentially.

    Example:
        pruner = ComposedPruner([
            SimilarityDedupPruner(threshold=0.95),
            HeavyHitterPruner(keep_ratio=0.3),
        ])
        pruned_keys, pruned_values, indices = pruner.prune(keys, values, scores)
    """

    def __init__(self, pruners: List[BasePruner]):
        self.pruners = pruners

    def prune(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        attention_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply all pruners sequentially."""
        current_keys = keys
        current_values = values
        current_scores = attention_scores
        all_indices = np.arange(keys.shape[0])

        for pruner in self.pruners:
            new_keys, new_values, kept = pruner.prune(
                current_keys, current_values, current_scores
            )
            all_indices = all_indices[kept]
            current_keys = new_keys
            current_values = new_values
            current_scores = current_scores[kept]

        return current_keys, current_values, all_indices
