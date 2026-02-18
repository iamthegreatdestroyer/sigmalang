"""
Tests for hybrid optimization pipeline — combining multiple Phase 7
optimization strategies (entropy, PQ, prompt compression, NAS evaluation).
"""

import pytest
import numpy as np

from sigmalang.core.entropy_estimator import (
    zeroth_order_entropy,
    first_order_entropy,
    EntropyAnalyzer,
)
from sigmalang.core.pq_codebook import ProductQuantizer, PQConfig
from sigmalang.core.prompt_compressor import PromptCompressor, CompressorConfig
from sigmalang.nas.evaluator import ArchitectureEvaluator
from sigmalang.nas.search_space import SearchSpace, ArchitectureConfig


class TestEntropyPQPipeline:
    """Test entropy analysis feeding into PQ compression decisions."""

    def test_entropy_guides_quantization_bits(self):
        """Higher entropy text should warrant higher quantization precision."""
        analyzer = EntropyAnalyzer()

        low_entropy = "aaa bbb aaa bbb aaa bbb aaa bbb"
        high_entropy = "The quick brown fox jumps over the lazy dog near a stream"

        low_result = analyzer.analyze(low_entropy)
        high_result = analyzer.analyze(high_entropy)

        assert high_result.entropy_h0 > low_result.entropy_h0

    def test_pq_preserves_search_quality(self):
        """PQ-compressed codebook should still support nearest-neighbor search."""
        pq = ProductQuantizer(PQConfig(num_subspaces=4, num_centroids=16))
        vectors = np.random.randn(64, 32).astype(np.float32)
        pq.train(vectors)
        codes = pq.encode(vectors)

        # Query with first vector — it should find itself as nearest
        results = pq.search(vectors[0], codes, top_k=1)
        assert results[0][0] == 0

    def test_pq_compression_stats(self):
        """Verify PQ compression ratio is reasonable."""
        pq = ProductQuantizer(PQConfig(num_subspaces=8, num_centroids=32))
        vectors = np.random.randn(256, 64).astype(np.float32)
        pq.train(vectors)
        stats = pq.compression_stats(vectors)

        assert stats['compression_ratio'] > 1.0
        assert stats['rmse'] > 0


class TestPromptCompressionQuality:
    """Test prompt compression preserves key information."""

    def test_short_text_passthrough(self):
        """Short texts should not be compressed."""
        comp = PromptCompressor(CompressorConfig(target_ratio=0.5, min_output_tokens=10))
        text = "Hello world"
        result = comp.compress_detailed(text)
        assert result['strategy'] == 'passthrough'
        assert result['compressed'] == text

    def test_compression_ratio_respected(self):
        """Compression should roughly match target ratio."""
        comp = PromptCompressor(CompressorConfig(target_ratio=0.5))
        text = " ".join(f"word_{i}" for i in range(40))
        result = comp.compress_detailed(text)
        assert result['compressed_tokens'] < result['original_tokens']
        assert result['ratio'] >= 1.0

    def test_protected_tokens_preserved(self):
        """First and last tokens should always be preserved."""
        comp = PromptCompressor(CompressorConfig(
            target_ratio=0.3, protect_first=2, protect_last=1
        ))
        words = ["START", "IMPORTANT"] + [f"filler_{i}" for i in range(30)] + ["END"]
        text = " ".join(words)
        result = comp.compress_detailed(text)
        compressed_words = result['compressed'].split()
        assert compressed_words[0] == "START"
        assert compressed_words[-1] == "END"

    def test_all_strategies_work(self):
        """All compression strategies should produce valid output."""
        text = " ".join(f"token_{i}" for i in range(30))
        for strategy in ['attention', 'merge', 'hybrid']:
            comp = PromptCompressor(CompressorConfig(
                target_ratio=0.5, strategy=strategy
            ))
            result = comp.compress(text)
            assert len(result) > 0
            assert len(result.split()) <= 30


class TestNASEvaluation:
    """Test NAS architecture evaluation pipeline."""

    def test_evaluate_returns_valid_result(self):
        """Evaluation should return all expected metrics."""
        evaluator = ArchitectureEvaluator()
        config = ArchitectureConfig(
            encoder_layers=3,
            encoder_hidden_dim=128,
            codebook_size=256,
        )
        config.architecture_id = "test-arch"
        result = evaluator.evaluate(config)

        assert result.compression_ratio_estimate > 0
        assert result.latency_estimate_ms > 0
        assert result.memory_estimate_kb > 0
        assert 0 <= result.pareto_score <= 1

    def test_deeper_networks_better_compression(self):
        """Deeper architectures should estimate higher compression ratios."""
        evaluator = ArchitectureEvaluator()

        shallow = ArchitectureConfig(encoder_layers=1, decoder_layers=1)
        shallow.architecture_id = "shallow"
        deep = ArchitectureConfig(encoder_layers=8, decoder_layers=4)
        deep.architecture_id = "deep"

        r_shallow = evaluator.evaluate(shallow)
        r_deep = evaluator.evaluate(deep)

        assert r_deep.compression_ratio_estimate > r_shallow.compression_ratio_estimate

    def test_search_space_crossover(self):
        """Crossover should produce valid architectures."""
        space = SearchSpace()
        a = space.sample_random()
        b = space.sample_random()
        child = space.crossover(a, b)

        assert child.encoder_layers in [a.encoder_layers, b.encoder_layers]
        assert child.generation == max(a.generation, b.generation) + 1

    def test_vector_encoding_roundtrip(self):
        """Architecture should survive encode to decode roundtrip."""
        space = SearchSpace()
        original = space.sample_random()
        vec = space.encode_to_vector(original)
        decoded = space.decode_from_vector(vec)

        assert decoded.encoder_layers == original.encoder_layers
        assert decoded.codebook_size == original.codebook_size


class TestEndToEndOptimizationFlow:
    """Integration test: full optimization pipeline."""

    def test_entropy_to_nas_to_pq_pipeline(self):
        """
        Full pipeline: analyze text entropy, evaluate architecture,
        compress codebook with PQ.
        """
        # Step 1: Entropy analysis
        analyzer = EntropyAnalyzer()
        text = "SigmaLang compresses semantic content using primitives. " * 20
        entropy = analyzer.analyze(text)
        assert entropy.entropy_h0 > 0

        # Step 2: NAS evaluation
        evaluator = ArchitectureEvaluator()
        config = ArchitectureConfig(
            encoder_layers=4,
            encoder_hidden_dim=256,
            codebook_size=256,
            embedding_dim=64,
        )
        config.architecture_id = "pipeline-test"
        nas_result = evaluator.evaluate(config)
        assert nas_result.pareto_score > 0

        # Step 3: PQ compress the codebook
        codebook = np.random.randn(
            config.codebook_size, config.embedding_dim
        ).astype(np.float32)

        pq = ProductQuantizer(PQConfig(num_subspaces=8, num_centroids=32))
        pq.train(codebook)
        codes = pq.encode(codebook)
        stats = pq.compression_stats(codebook)

        assert stats['compression_ratio'] > 1.0
        assert codes.shape == (256, 8)

        # Step 4: Verify search still works on PQ-compressed codebook
        query = codebook[42]
        results = pq.search(query, codes, top_k=3)
        assert results[0][0] == 42  # Should find itself
