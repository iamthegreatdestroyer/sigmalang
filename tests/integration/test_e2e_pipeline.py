"""
End-to-End Pipeline Integration Tests

Tests the complete SigmaLang pipeline from text input through parsing,
encoding, decoding, and validation of output. This verifies that all
components work together correctly in real-world scenarios.

Test Coverage:
- Full text → parse → encode → decode → verify roundtrip
- Multiple text types: natural language, code, queries
- Compression ratio validation
- Data integrity verification
- Performance benchmarks for complete pipeline
"""

import sys
import pytest
from pathlib import Path
from typing import List

# Add parent to path
sigmalang_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sigmalang_root))

from sigmalang.core.parser import SemanticParser
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
from sigmalang.core.primitives import SemanticTree
from tests.conftest import TreeComparator, CompressionAnalyzer, TestDatasets


class TestE2EBasicPipeline:
    """Basic end-to-end pipeline tests."""

    @pytest.mark.integration
    def test_simple_text_roundtrip(self):
        """Test complete pipeline with simple text input."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        input_text = "Hello, world! This is a test."

        # Parse
        tree = parser.parse(input_text)
        assert isinstance(tree, SemanticTree)
        assert tree.source_text == input_text

        # Encode
        encoded = encoder.encode(tree)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        # Decode
        decoded = decoder.decode(encoded)
        assert isinstance(decoded, SemanticTree)

        # Verify integrity
        assert TreeComparator.trees_equal(tree, decoded), \
            f"Round-trip failed: {TreeComparator.get_differences(tree, decoded)}"

    @pytest.mark.integration
    def test_code_snippet_roundtrip(self):
        """Test pipeline with Python code snippet."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

        tree = parser.parse(code)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        assert TreeComparator.trees_equal(tree, decoded)

    @pytest.mark.integration
    def test_natural_language_query_roundtrip(self):
        """Test pipeline with natural language query."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        query = "Find all users who registered in the last 30 days and have verified emails"

        tree = parser.parse(query)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        assert TreeComparator.trees_equal(tree, decoded)


class TestE2EDataIntegrity:
    """Data integrity validation across the pipeline."""

    @pytest.mark.integration
    @pytest.mark.parametrize("input_text", [
        "Simple text",
        "Text with üñíçödé characters",
        "Text\nwith\nnewlines",
        "Text\twith\ttabs",
        'Text with "quotes" and \'apostrophes\'',
        "A" * 1000,  # Large text
    ])
    def test_special_characters_integrity(self, input_text):
        """Test data integrity with special characters."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        tree = parser.parse(input_text)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        assert TreeComparator.trees_equal(tree, decoded), \
            f"Failed for input: {repr(input_text)}"

    @pytest.mark.integration
    @pytest.mark.parametrize("input_text", TestDatasets.CODE_SNIPPETS[:10])
    def test_code_snippets_integrity(self, input_text):
        """Test data integrity for various code snippets."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        tree = parser.parse(input_text)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        assert TreeComparator.trees_equal(tree, decoded)

    @pytest.mark.integration
    @pytest.mark.parametrize("input_text", TestDatasets.QUERIES[:10])
    def test_queries_integrity(self, input_text):
        """Test data integrity for various queries."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        tree = parser.parse(input_text)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        assert TreeComparator.trees_equal(tree, decoded)


class TestE2ECompression:
    """Compression ratio validation across the pipeline."""

    @pytest.mark.integration
    def test_compression_ratio_achieves_target(self):
        """Test that compression achieves reasonable ratios."""
        parser = SemanticParser()
        encoder = SigmaEncoder()

        # Use medium-length text (compression should work well)
        input_texts = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
            "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.date > '2024-01-01'",
        ]

        for input_text in input_texts:
            tree = parser.parse(input_text)
            encoded = encoder.encode(tree)

            original_size = len(input_text.encode('utf-8'))
            compressed_size = len(encoded)

            ratio = CompressionAnalyzer.compute_ratio(original_size, compressed_size)

            # For text > 80 bytes, expect reasonable compression
            if original_size > 80:
                assert ratio < 1.0, \
                    f"No compression achieved for text of {original_size} bytes: ratio={ratio}"

    @pytest.mark.integration
    def test_compression_ratio_distribution(self):
        """Test compression ratio across diverse inputs."""
        parser = SemanticParser()
        encoder = SigmaEncoder()

        test_texts = TestDatasets.CODE_SNIPPETS[:15] + TestDatasets.QUERIES[:15]
        ratios = []

        for text in test_texts:
            tree = parser.parse(text)
            encoded = encoder.encode(tree)

            original_size = len(text.encode('utf-8'))
            compressed_size = len(encoded)

            if original_size > 100:  # Only consider longer texts
                ratio = CompressionAnalyzer.compute_ratio(original_size, compressed_size)
                ratios.append(ratio)

        # Average compression should be good for longer texts
        if ratios:
            import numpy as np
            avg_ratio = np.mean(ratios)
            assert avg_ratio < 0.95, \
                f"Average compression ratio too high: {avg_ratio}"


class TestE2EPerformance:
    """Performance benchmarks for complete pipeline."""

    @pytest.mark.integration
    @pytest.mark.performance
    def test_pipeline_latency(self):
        """Test end-to-end pipeline latency."""
        import time

        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        input_text = "SELECT * FROM users WHERE created_at > NOW() - INTERVAL '30 days'"

        start = time.perf_counter()

        # Full pipeline
        tree = parser.parse(input_text)
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        end = time.perf_counter()
        latency_ms = (end - start) * 1000

        # Entire pipeline should complete in reasonable time
        assert latency_ms < 100, \
            f"Pipeline too slow: {latency_ms:.2f}ms"

    @pytest.mark.integration
    @pytest.mark.performance
    def test_batch_processing_throughput(self):
        """Test throughput for batch processing."""
        import time

        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        texts = TestDatasets.QUERIES[:50]

        start = time.perf_counter()

        for text in texts:
            tree = parser.parse(text)
            encoded = encoder.encode(tree)
            decoded = decoder.decode(encoded)

        end = time.perf_counter()
        duration = end - start
        throughput = len(texts) / duration

        # Should process at least 10 texts per second
        assert throughput >= 10, \
            f"Throughput too low: {throughput:.2f} texts/sec"


class TestE2EErrorHandling:
    """Error handling across the pipeline."""

    @pytest.mark.integration
    def test_empty_input_handling(self):
        """Test pipeline handles empty input gracefully."""
        parser = SemanticParser()
        encoder = SigmaEncoder()
        decoder = SigmaDecoder(encoder)

        tree = parser.parse("")
        encoded = encoder.encode(tree)
        decoded = decoder.decode(encoded)

        assert TreeComparator.trees_equal(tree, decoded)

    @pytest.mark.integration
    def test_malformed_encoded_data_handling(self):
        """Test decoder handles malformed data gracefully."""
        decoder = SigmaDecoder(SigmaEncoder())

        malformed_data = b'\x00\x01\x02\x03\x04'

        try:
            decoded = decoder.decode(malformed_data)
            # If it doesn't raise, that's also acceptable
        except Exception as e:
            # Should raise a specific exception, not crash
            assert isinstance(e, (ValueError, RuntimeError))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
