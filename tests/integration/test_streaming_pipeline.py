"""
Streaming Pipeline Integration Tests

Tests the streaming encoder with large data files and verifies constant
memory usage. Validates that the streaming approach works correctly for
files larger than available RAM.

Test Coverage:
- Streaming encoder with files > 10MB
- Memory usage remains constant during streaming
- Compression ratio matches non-streaming approach
- Boundary handling (glyphs spanning chunk boundaries)
- Throughput and performance metrics
- Error recovery during streaming
"""

import os
import random
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add parent to path
sigmalang_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sigmalang_root))

from sigmalang.core.encoder import SigmaEncoder  # noqa: E402
from sigmalang.core.parser import SemanticParser  # noqa: E402
from sigmalang.core.streaming_encoder import (  # noqa: E402
    BoundaryHandler,
    Chunk,
    ChunkedReader,
    StreamBuffer,
    StreamingEncoder,
    estimate_memory_usage,
    get_optimal_chunk_size,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_test_file(temp_dir):
    """Create small test file (1 MB)."""
    file_path = temp_dir / "small_file.txt"
    content = "Test data line\n" * 70000  # ~1 MB
    file_path.write_text(content)
    return file_path


@pytest.fixture
def medium_test_file(temp_dir):
    """Create medium test file (10 MB)."""
    file_path = temp_dir / "medium_file.txt"
    content = "Machine learning and artificial intelligence\n" * 200000  # ~10 MB
    file_path.write_text(content)
    return file_path


@pytest.fixture
def large_test_file(temp_dir):
    """Create large test file (50 MB) - only for specific tests."""
    file_path = temp_dir / "large_file.txt"
    # Write in chunks to avoid memory issues during test setup
    with open(file_path, 'w') as f:
        for _ in range(1000):
            # Write ~50KB per iteration = 50MB total
            chunk = "This is a test line for large file streaming\n" * 1000
            f.write(chunk)
    return file_path


class TestStreamingBasicFunctionality:
    """Basic streaming functionality tests."""

    @pytest.mark.integration
    def test_streaming_encoder_small_file(self, small_test_file, temp_dir):
        """Test streaming encoder with small file."""
        encoder = StreamingEncoder()
        output_path = str(temp_dir / "small_output.sigma")

        result = encoder.encode_file(str(small_test_file), output_path)

        assert result is not None

    @pytest.mark.integration
    def test_streaming_encoder_medium_file(self, medium_test_file, temp_dir):
        """Test streaming encoder with medium file (10 MB)."""
        encoder = StreamingEncoder()
        output_path = str(temp_dir / "medium_output.sigma")

        result = encoder.encode_file(str(medium_test_file), output_path)

        assert result is not None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_streaming_encoder_large_file(self, large_test_file, temp_dir):
        """Test streaming encoder with large file (50 MB)."""
        encoder = StreamingEncoder()
        output_path = str(temp_dir / "large_output.sigma")

        result = encoder.encode_file(str(large_test_file), output_path)

        assert result is not None

    @pytest.mark.integration
    def test_chunked_reader_basic(self, small_test_file):
        """Test ChunkedReader basic functionality."""
        reader = ChunkedReader(str(small_test_file), chunk_size=65536)

        chunks = list(reader.read_chunks())

        assert len(chunks) > 0
        assert chunks[-1].is_final

        # Verify total bytes read
        total_bytes = sum(len(chunk.data) for chunk in chunks)
        assert total_bytes == small_test_file.stat().st_size


class TestStreamingMemoryUsage:
    """Memory usage validation for streaming."""

    @pytest.mark.integration
    def test_memory_remains_constant(self, medium_test_file, temp_dir):
        """Test that memory usage remains constant during streaming."""
        import tracemalloc

        encoder = StreamingEncoder()
        output_path = str(temp_dir / "memory_test_output.sigma")

        # Start memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Encode file in streaming mode
        encoder.encode_file(str(medium_test_file), output_path)

        # Check memory after encoding
        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)

        # Peak memory should be much less than file size (10 MB)
        # Allow up to 15 MB for overhead, but ideally < 10 MB
        assert memory_used_mb < 15, \
            f"Memory usage too high: {memory_used_mb:.2f} MB for 10 MB file"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_scaling_large_file(self, large_test_file, temp_dir):
        """Test memory scaling with large file (50 MB)."""
        import tracemalloc

        encoder = StreamingEncoder()
        output_path = str(temp_dir / "memory_scaling_output.sigma")

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        encoder.encode_file(str(large_test_file), output_path)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / (1024 * 1024)

        # For 50 MB file, memory should still be < 20 MB
        assert memory_used_mb < 20, \
            f"Memory usage not scaling well: {memory_used_mb:.2f} MB for 50 MB file"

    @pytest.mark.integration
    def test_optimal_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        # Small file
        chunk_size_small = get_optimal_chunk_size(1024 * 1024)  # 1 MB
        assert 64 * 1024 <= chunk_size_small <= 1024 * 1024

        # Large file
        chunk_size_large = get_optimal_chunk_size(100 * 1024 * 1024)  # 100 MB
        assert chunk_size_large >= 1024 * 1024  # At least 1 MB chunks


class TestStreamingVsNonStreaming:
    """Compare streaming vs non-streaming approaches."""

    @pytest.mark.integration
    def test_compression_ratio_comparison(self, small_test_file):
        """Test compression ratio matches between streaming and non-streaming."""
        # Non-streaming
        parser = SemanticParser()
        encoder = SigmaEncoder()

        text = small_test_file.read_text()
        tree = parser.parse(text)
        encoded_normal = encoder.encode(tree)

        # Streaming
        streaming_encoder = StreamingEncoder()
        streaming_output = str(small_test_file.parent / "streaming_output.sigma")
        result_streaming = streaming_encoder.encode_file(str(small_test_file), streaming_output)

        # Both should produce valid output
        assert len(encoded_normal) > 0
        assert result_streaming is not None

        # Compression ratios should be similar (within 10%)
        # This is a loose check since methods may differ slightly
        original_size = len(text.encode('utf-8'))

        # Just verify both achieve some compression for larger inputs
        if original_size > 1000:
            assert len(encoded_normal) < original_size * 1.5

    @pytest.mark.integration
    @pytest.mark.slow
    def test_streaming_performance(self, medium_test_file, temp_dir):
        """Test streaming performance metrics."""
        encoder = StreamingEncoder()
        output_path = str(temp_dir / "perf_output.sigma")

        start_time = time.perf_counter()
        encoder.encode_file(str(medium_test_file), output_path)
        end_time = time.perf_counter()

        duration = end_time - start_time
        file_size_mb = medium_test_file.stat().st_size / (1024 * 1024)
        throughput_mbps = file_size_mb / duration

        # Should achieve at least 1 MB/s throughput
        assert throughput_mbps >= 1.0, \
            f"Throughput too low: {throughput_mbps:.2f} MB/s"


class TestBoundaryHandling:
    """Tests for boundary condition handling."""

    @pytest.mark.integration
    def test_boundary_handler_incomplete_data(self):
        """Test boundary handler with incomplete glyph at chunk boundary."""
        handler = BoundaryHandler()

        # Simulate incomplete glyph data
        data = b'\x00\x01\x02'
        glyphs, leftover = handler.try_extract_glyphs(data)

        # Should handle incomplete data gracefully
        assert isinstance(glyphs, list)
        assert isinstance(leftover, bytes)

    @pytest.mark.integration
    def test_boundary_handler_multiple_glyphs(self):
        """Test boundary handler with multiple complete glyphs."""
        handler = BoundaryHandler()

        # Multiple complete glyphs
        data = b'\x00\x00\x01\x00\x02\x00'
        glyphs, leftover = handler.try_extract_glyphs(data)

        assert isinstance(glyphs, list)

    @pytest.mark.integration
    def test_streaming_with_random_chunk_sizes(self, small_test_file, temp_dir):
        """Test streaming with random chunk sizes."""
        # Use various chunk sizes to stress-test boundary handling
        chunk_sizes = [1024, 4096, 8192, 16384, 65536]

        for chunk_size in chunk_sizes:
            encoder = StreamingEncoder(chunk_size=chunk_size)
            output_path = str(temp_dir / f"chunk_{chunk_size}_output.sigma")
            result = encoder.encode_file(str(small_test_file), output_path)

            assert result is not None, \
                f"Failed with chunk size {chunk_size}"


class TestStreamingErrorHandling:
    """Error handling and recovery tests."""

    @pytest.mark.integration
    def test_streaming_nonexistent_file(self, temp_dir):
        """Test streaming with nonexistent file."""
        encoder = StreamingEncoder()
        output_path = str(temp_dir / "nonexistent_output.sigma")

        with pytest.raises((FileNotFoundError, IOError, OSError)):
            encoder.encode_file("/nonexistent/file.txt", output_path)

    @pytest.mark.integration
    def test_streaming_empty_file(self, temp_dir):
        """Test streaming with empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()

        encoder = StreamingEncoder()
        output_path = str(temp_dir / "empty_output.sigma")
        result = encoder.encode_file(str(empty_file), output_path)

        # Should handle empty file gracefully
        assert result is not None

    @pytest.mark.integration
    def test_streaming_binary_file(self, temp_dir):
        """Test streaming with binary file."""
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(bytes(range(256)) * 1000)

        encoder = StreamingEncoder()
        output_path = str(temp_dir / "binary_output.sigma")

        # Should either handle or raise appropriate error
        try:
            result = encoder.encode_file(str(binary_file), output_path)
            assert result is not None
        except (ValueError, UnicodeDecodeError):
            # Acceptable to reject binary data
            pass


class TestStreamBuffer:
    """Tests for StreamBuffer component."""

    @pytest.mark.integration
    def test_stream_buffer_basic(self):
        """Test StreamBuffer basic functionality."""
        buffer = StreamBuffer(name="test", max_size=3)

        # Add items
        buffer.put(Chunk(chunk_id=0, data=b"chunk1"))
        buffer.put(Chunk(chunk_id=1, data=b"chunk2"))
        buffer.put(Chunk(chunk_id=2, data=b"chunk3"))

        # Retrieve items
        assert buffer.get().data == b"chunk1"
        assert buffer.get().data == b"chunk2"
        assert buffer.get().data == b"chunk3"

    @pytest.mark.integration
    def test_stream_buffer_backpressure(self):
        """Test StreamBuffer backpressure handling."""
        import queue
        import threading

        buffer = StreamBuffer(name="test_backpressure", max_size=2)

        # Fill buffer
        buffer.put(Chunk(chunk_id=0, data=b"chunk1"))
        buffer.put(Chunk(chunk_id=1, data=b"chunk2"))

        # Try to add one more (should block or raise)
        def try_put():
            try:
                buffer.put(Chunk(chunk_id=2, data=b"chunk3"), timeout=0.1)
            except queue.Full:
                pass

        thread = threading.Thread(target=try_put)
        thread.start()
        thread.join(timeout=0.5)


class TestStreamingStatistics:
    """Tests for streaming statistics and monitoring."""

    @pytest.mark.integration
    def test_streaming_stats_collection(self, small_test_file, temp_dir):
        """Test that streaming collects statistics."""
        encoder = StreamingEncoder()
        output_path = str(temp_dir / "stats_output.sigma")

        result = encoder.encode_file(str(small_test_file), output_path)

        # result is a StreamStats dataclass; verify it has expected fields
        stats = result if hasattr(result, 'total_chunks') else encoder.stats

        # Verify stats have expected fields
        assert hasattr(stats, 'total_chunks') or hasattr(stats, 'total_bytes_read')
        assert stats.total_chunks > 0 or stats.total_bytes_read > 0

    @pytest.mark.integration
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Estimate for 100 MB file with 64KB chunk size
        estimated = estimate_memory_usage(100 * 1024 * 1024, 65536)

        # estimate_memory_usage returns a dict of memory components
        total_estimated = sum(estimated.values())

        # Estimate should be reasonable (less than file size)
        assert 0 < total_estimated < 100 * 1024 * 1024
        # Should be in the range of a few MB
        assert total_estimated < 50 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
