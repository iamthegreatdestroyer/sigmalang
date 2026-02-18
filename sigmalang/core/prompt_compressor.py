"""
Attention-Only Prompt Compressor - Phase 7 Track 7

Lightweight prompt compression that reduces SigmaLang API input size
before encoding. Uses attention-only layers (no MLPs) for efficiency.

Architecture:
    Input tokens → Multi-Head Attention → Compressed tokens
    (N tokens → N/r tokens, where r is the compression ratio)

The compressor selects the most important tokens using attention
scores and merges similar tokens via weighted averaging.

Research Basis:
    - "Better Prompt Compression Without MLPs" (Jan 2025):
      Attention-only compressor (AOC) achieves comparable quality
      to full transformer compressors while being 3-5x faster.
    - Token merging (ToMe): merge similar tokens progressively

Strategies:
    1. Attention Selection: Keep top-k tokens by self-attention importance
    2. Token Merging: Merge similar adjacent tokens via cosine similarity
    3. Hybrid: Select important + merge redundant

Usage:
    from sigmalang.core.prompt_compressor import PromptCompressor

    compressor = PromptCompressor(target_ratio=0.5)
    compressed = compressor.compress("long input text here...")
    # compressed has ~50% of original tokens
"""

import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressorConfig:
    """Configuration for prompt compression."""
    target_ratio: float = 0.5       # Keep this fraction of tokens
    num_heads: int = 4              # Attention heads
    head_dim: int = 32              # Dimension per head
    strategy: str = 'hybrid'        # 'attention', 'merge', 'hybrid'
    merge_threshold: float = 0.85   # Cosine similarity for merging
    protect_first: int = 2          # Always keep first N tokens
    protect_last: int = 1           # Always keep last N tokens
    min_output_tokens: int = 4      # Minimum output length


# =============================================================================
# Token Embedder (lightweight)
# =============================================================================

class SimpleTokenEmbedder:
    """
    Lightweight token embedding using character n-grams + positional encoding.

    No pretrained weights needed — produces consistent embeddings
    for any string input.
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self._rng = np.random.RandomState(seed)
        # Hash-based embedding table
        self._hash_table = self._rng.randn(65536, dim).astype(np.float32) * 0.1

    def embed(self, tokens: List[str]) -> np.ndarray:
        """
        Embed a list of tokens into vectors.

        Returns: (N, dim)
        """
        N = len(tokens)
        embeddings = np.zeros((N, self.dim), dtype=np.float32)

        for i, token in enumerate(tokens):
            # Character trigram hashing
            for j in range(len(token) - 2):
                trigram = token[j:j + 3]
                h = hash(trigram) % 65536
                embeddings[i] += self._hash_table[h]

            # Word-level hash
            h = hash(token) % 65536
            embeddings[i] += self._hash_table[h] * 2.0

            # Positional encoding
            for d in range(0, self.dim, 2):
                freq = 1.0 / (10000.0 ** (d / self.dim))
                embeddings[i, d] += math.sin(i * freq)
                if d + 1 < self.dim:
                    embeddings[i, d + 1] += math.cos(i * freq)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings /= norms

        return embeddings


# =============================================================================
# Self-Attention (single layer, no MLP)
# =============================================================================

class AttentionOnlyLayer:
    """
    Single multi-head self-attention layer without MLP.

    Computes attention scores that indicate token importance.
    """

    def __init__(self, dim: int, num_heads: int = 4, seed: int = 42):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        rng = np.random.RandomState(seed)
        scale = math.sqrt(2.0 / (dim + self.head_dim))

        # Q, K, V projections
        self.W_q = rng.randn(dim, dim).astype(np.float32) * scale
        self.W_k = rng.randn(dim, dim).astype(np.float32) * scale
        self.W_v = rng.randn(dim, dim).astype(np.float32) * scale
        self.W_o = rng.randn(dim, dim).astype(np.float32) * scale

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through attention layer.

        Args:
            x: (N, dim) input embeddings

        Returns:
            (output, attention_weights)
            output: (N, dim)
            attention_weights: (N,) importance scores per token
        """
        N = x.shape[0]

        Q = x @ self.W_q  # (N, dim)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(N, self.num_heads, self.head_dim)
        K = K.reshape(N, self.num_heads, self.head_dim)
        V = V.reshape(N, self.num_heads, self.head_dim)

        # Attention scores per head: (num_heads, N, N)
        scale = math.sqrt(self.head_dim)
        all_weights = np.zeros((self.num_heads, N, N), dtype=np.float32)
        all_outputs = np.zeros((self.num_heads, N, self.head_dim), dtype=np.float32)

        for h in range(self.num_heads):
            scores = Q[:, h, :] @ K[:, h, :].T / scale  # (N, N)
            # Softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            weights = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-8)
            all_weights[h] = weights
            all_outputs[h] = weights @ V[:, h, :]  # (N, head_dim)

        # Concatenate heads
        output = all_outputs.transpose(1, 0, 2).reshape(N, -1)  # (N, dim)
        output = output @ self.W_o

        # Aggregate attention importance: sum of attention received per token
        # across all heads and all query positions
        importance = all_weights.sum(axis=0).sum(axis=0)  # (N,) — how much each token is attended to

        return output, importance


# =============================================================================
# Token Merger
# =============================================================================

class TokenMerger:
    """Merge similar adjacent tokens via weighted averaging."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def merge(
        self,
        tokens: List[str],
        embeddings: np.ndarray,
        importance: np.ndarray,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Merge similar adjacent tokens.

        Returns (merged_tokens, merged_embeddings, merged_importance).
        """
        N = len(tokens)
        if N < 2:
            return tokens, embeddings, importance

        # Compute adjacent cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = embeddings / norms

        merged_tokens = [tokens[0]]
        merged_embs = [embeddings[0].copy()]
        merged_imp = [importance[0]]
        merge_count = [1]

        for i in range(1, N):
            sim = float(np.dot(normalized[i], normalized[i - 1]))
            if sim > self.threshold:
                # Merge with previous: weighted average by importance
                w_prev = merged_imp[-1]
                w_curr = importance[i]
                total = w_prev + w_curr + 1e-8
                merged_embs[-1] = (w_prev * merged_embs[-1] + w_curr * embeddings[i]) / total
                merged_imp[-1] = max(w_prev, w_curr)
                merged_tokens[-1] = merged_tokens[-1] + " " + tokens[i]
                merge_count[-1] += 1
            else:
                merged_tokens.append(tokens[i])
                merged_embs.append(embeddings[i].copy())
                merged_imp.append(importance[i])
                merge_count.append(1)

        return (
            merged_tokens,
            np.stack(merged_embs),
            np.array(merged_imp, dtype=np.float32),
        )


# =============================================================================
# Prompt Compressor
# =============================================================================

class PromptCompressor:
    """
    Attention-only prompt compressor.

    Reduces input token count while preserving semantic content.
    """

    def __init__(self, config: Optional[CompressorConfig] = None):
        self.config = config or CompressorConfig()
        dim = self.config.num_heads * self.config.head_dim

        self.embedder = SimpleTokenEmbedder(dim=dim)
        self.attention = AttentionOnlyLayer(dim=dim, num_heads=self.config.num_heads)
        self.merger = TokenMerger(threshold=self.config.merge_threshold)

    def compress(self, text: str) -> str:
        """
        Compress a text prompt.

        Args:
            text: input text to compress

        Returns:
            Compressed text with fewer tokens
        """
        tokens = text.split()
        if len(tokens) <= self.config.min_output_tokens:
            return text

        result = self._compress_tokens(tokens)
        return " ".join(result['tokens'])

    def compress_detailed(self, text: str) -> Dict[str, Any]:
        """
        Compress with detailed statistics.

        Returns dict with compressed text, token counts, importance scores.
        """
        tokens = text.split()
        if len(tokens) <= self.config.min_output_tokens:
            return {
                'compressed': text,
                'original_tokens': len(tokens),
                'compressed_tokens': len(tokens),
                'ratio': 1.0,
                'strategy': 'passthrough',
            }

        result = self._compress_tokens(tokens)
        return {
            'compressed': " ".join(result['tokens']),
            'original_tokens': len(tokens),
            'compressed_tokens': len(result['tokens']),
            'ratio': round(len(tokens) / max(1, len(result['tokens'])), 2),
            'strategy': self.config.strategy,
            'tokens': result['tokens'],
        }

    def _compress_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Internal compression pipeline."""
        N = len(tokens)
        target_count = max(
            self.config.min_output_tokens,
            int(N * self.config.target_ratio)
        )

        # Step 1: Embed tokens
        embeddings = self.embedder.embed(tokens)

        # Step 2: Compute attention importance
        _, importance = self.attention.forward(embeddings)

        # Step 3: Apply strategy
        if self.config.strategy == 'attention':
            result_tokens, result_embs = self._attention_select(
                tokens, embeddings, importance, target_count
            )
        elif self.config.strategy == 'merge':
            result_tokens, result_embs, _ = self.merger.merge(
                tokens, embeddings, importance
            )
            # If still too many, do attention selection on merged result
            if len(result_tokens) > target_count:
                new_imp = np.ones(len(result_tokens), dtype=np.float32)
                result_tokens, result_embs = self._attention_select(
                    result_tokens, result_embs, new_imp, target_count
                )
        else:  # hybrid
            # First merge similar tokens
            merged_tokens, merged_embs, merged_imp = self.merger.merge(
                tokens, embeddings, importance
            )
            # Then select by importance if still too many
            if len(merged_tokens) > target_count:
                result_tokens, result_embs = self._attention_select(
                    merged_tokens, merged_embs, merged_imp, target_count
                )
            else:
                result_tokens = merged_tokens
                result_embs = merged_embs

        return {'tokens': result_tokens, 'embeddings': result_embs}

    def _attention_select(
        self,
        tokens: List[str],
        embeddings: np.ndarray,
        importance: np.ndarray,
        target_count: int,
    ) -> Tuple[List[str], np.ndarray]:
        """Select top-k tokens by attention importance."""
        N = len(tokens)

        # Build protection mask
        keep_mask = np.zeros(N, dtype=bool)
        keep_mask[:min(self.config.protect_first, N)] = True
        keep_mask[max(0, N - self.config.protect_last):] = True

        # Budget for non-protected
        already_kept = keep_mask.sum()
        budget = max(0, target_count - already_kept)

        if budget > 0:
            # Zero out already-kept scores
            scores = importance.copy()
            scores[keep_mask] = -np.inf

            top_indices = np.argsort(scores)[-budget:]
            keep_mask[top_indices] = True

        # Extract kept tokens in original order
        kept_indices = np.where(keep_mask)[0]
        result_tokens = [tokens[i] for i in kept_indices]
        result_embs = embeddings[kept_indices]

        return result_tokens, result_embs

    def get_stats(self) -> Dict[str, Any]:
        """Get compressor configuration stats."""
        return {
            'strategy': self.config.strategy,
            'target_ratio': self.config.target_ratio,
            'num_heads': self.config.num_heads,
            'head_dim': self.config.head_dim,
            'merge_threshold': self.config.merge_threshold,
        }
