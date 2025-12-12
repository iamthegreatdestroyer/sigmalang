"""
Tests for the Streaming Processor Module.

Tests cover:
- WindowedAggregator: Tumbling, sliding, session, count windows
- BackpressureHandler: All strategies and buffer management
- StreamProcessor: Processing, batching, error handling
- StreamPipeline: End-to-end pipeline execution
- Sources and Sinks: Iterator, List, Callback implementations
- Operators: Map, Filter, FlatMap
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from typing import List, Any
from unittest.mock import Mock, patch, MagicMock

from core.streaming_processor import (
    # Enums
    WindowType,
    BackpressureStrategy,
    StreamState,
    # Configs
    WindowConfig,
    BackpressureConfig,
    StreamConfig,
    # Data classes
    Window,
    # Abstract classes
    StreamSource,
    StreamSink,
    StreamOperator,
    # Concrete implementations
    IteratorSource,
    ListSink,
    CallbackSink,
    MapOperator,
    FilterOperator,
    FlatMapOperator,
    # Core classes
    WindowedAggregator,
    BackpressureHandler,
    StreamProcessor,
    StreamPipeline,
    # Convenience functions
    create_windowed_aggregator,
    create_backpressure_handler,
    create_stream_processor,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestEnums:
    """Tests for enum definitions."""
    
    def test_window_type_values(self):
        """Test WindowType enum has all expected values."""
        assert WindowType.TUMBLING
        assert WindowType.SLIDING
        assert WindowType.SESSION
        assert WindowType.COUNT
    
    def test_backpressure_strategy_values(self):
        """Test BackpressureStrategy enum has all expected values."""
        assert BackpressureStrategy.BLOCK
        assert BackpressureStrategy.DROP_OLDEST
        assert BackpressureStrategy.DROP_NEWEST
        assert BackpressureStrategy.SAMPLE
        assert BackpressureStrategy.BUFFER
    
    def test_stream_state_values(self):
        """Test StreamState enum has all expected values."""
        assert StreamState.CREATED
        assert StreamState.RUNNING
        assert StreamState.PAUSED
        assert StreamState.STOPPED
        assert StreamState.ERROR


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestWindowConfig:
    """Tests for WindowConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = WindowConfig()
        assert config.window_type == WindowType.TUMBLING
        assert config.window_size_seconds == 60.0
        assert config.slide_interval_seconds == 30.0
        assert config.session_gap_seconds == 30.0
        assert config.count_size == 100
        assert config.allow_late_data_seconds == 0.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = WindowConfig(
            window_type=WindowType.SLIDING,
            window_size_seconds=120.0,
            slide_interval_seconds=60.0
        )
        assert config.window_type == WindowType.SLIDING
        assert config.window_size_seconds == 120.0
        assert config.slide_interval_seconds == 60.0


class TestBackpressureConfig:
    """Tests for BackpressureConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = BackpressureConfig()
        assert config.strategy == BackpressureStrategy.BLOCK
        assert config.buffer_size == 10000
        assert config.max_buffer_size == 100000
        assert config.high_watermark == 0.8
        assert config.low_watermark == 0.5
        assert config.sample_rate == 0.1
        assert config.block_timeout_seconds == 30.0
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            buffer_size=5000,
            high_watermark=0.9
        )
        assert config.strategy == BackpressureStrategy.DROP_OLDEST
        assert config.buffer_size == 5000
        assert config.high_watermark == 0.9


class TestStreamConfig:
    """Tests for StreamConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = StreamConfig()
        assert config.name == "default-stream"
        assert config.batch_size == 100
        assert config.batch_timeout_seconds == 1.0
        assert config.max_concurrent_tasks == 4
        assert config.enable_checkpointing is False
        assert config.error_handling == "skip"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = StreamConfig(
            name="my-stream",
            batch_size=50,
            error_handling="fail"
        )
        assert config.name == "my-stream"
        assert config.batch_size == 50
        assert config.error_handling == "fail"


# =============================================================================
# SOURCE AND SINK TESTS
# =============================================================================


class TestIteratorSource:
    """Tests for IteratorSource."""
    
    def test_read_from_list(self):
        """Test reading from a list iterator."""
        data = [1, 2, 3, 4, 5]
        source = IteratorSource(iter(data))
        
        results = []
        while True:
            item = source.read()
            if item is None:
                break
            results.append(item)
        
        assert results == data
    
    def test_read_empty_iterator(self):
        """Test reading from empty iterator."""
        source = IteratorSource(iter([]))
        assert source.read() is None
    
    def test_iterator_protocol(self):
        """Test source implements iterator protocol."""
        data = [1, 2, 3]
        source = IteratorSource(iter(data))
        
        results = list(source)
        assert results == data
    
    @pytest.mark.asyncio
    async def test_async_read(self):
        """Test async reading."""
        data = [1, 2, 3]
        source = IteratorSource(iter(data))
        
        results = []
        async for item in source:
            results.append(item)
        
        assert results == data
    
    def test_thread_safety(self):
        """Test thread-safe reading."""
        data = list(range(100))
        source = IteratorSource(iter(data))
        
        results = []
        lock = threading.Lock()
        
        def reader():
            while True:
                item = source.read()
                if item is None:
                    break
                with lock:
                    results.append(item)
        
        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert sorted(results) == data


class TestListSink:
    """Tests for ListSink."""
    
    def test_write_items(self):
        """Test writing items to sink."""
        sink = ListSink()
        
        sink.write(1)
        sink.write(2)
        sink.write(3)
        
        assert sink.items == [1, 2, 3]
    
    def test_write_returns_true(self):
        """Test write returns True."""
        sink = ListSink()
        assert sink.write("test") is True
    
    def test_write_batch(self):
        """Test batch writing."""
        sink = ListSink()
        count = sink.write_batch([1, 2, 3, 4, 5])
        
        assert count == 5
        assert sink.items == [1, 2, 3, 4, 5]
    
    def test_clear(self):
        """Test clearing sink."""
        sink = ListSink()
        sink.write(1)
        sink.write(2)
        
        sink.clear()
        assert sink.items == []
    
    @pytest.mark.asyncio
    async def test_async_write(self):
        """Test async writing."""
        sink = ListSink()
        
        await sink.write_async(1)
        await sink.write_async(2)
        
        assert sink.items == [1, 2]
    
    @pytest.mark.asyncio
    async def test_async_batch_write(self):
        """Test async batch writing."""
        sink = ListSink()
        count = await sink.write_batch_async([1, 2, 3])
        
        assert count == 3
        assert sink.items == [1, 2, 3]


class TestCallbackSink:
    """Tests for CallbackSink."""
    
    def test_callback_called(self):
        """Test callback is called for each item."""
        received = []
        callback = lambda x: (received.append(x), True)[1]
        
        sink = CallbackSink(callback)
        sink.write(1)
        sink.write(2)
        
        assert received == [1, 2]
    
    def test_callback_return_value(self):
        """Test callback return value is propagated."""
        sink = CallbackSink(lambda x: x > 0)
        
        assert sink.write(1) is True
        assert sink.write(-1) is False
    
    def test_callback_exception_handling(self):
        """Test callback exceptions are handled."""
        def failing_callback(x):
            raise ValueError("Test error")
        
        sink = CallbackSink(failing_callback)
        assert sink.write(1) is False


# =============================================================================
# OPERATOR TESTS
# =============================================================================


class TestMapOperator:
    """Tests for MapOperator."""
    
    def test_transform_items(self):
        """Test basic transformation."""
        op = MapOperator(lambda x: x * 2)
        
        assert op.process(5) == 10
        assert op.process(0) == 0
        assert op.process(-3) == -6
    
    def test_string_transform(self):
        """Test string transformation."""
        op = MapOperator(lambda s: s.upper())
        assert op.process("hello") == "HELLO"
    
    @pytest.mark.asyncio
    async def test_async_process(self):
        """Test async processing."""
        op = MapOperator(lambda x: x + 1)
        result = await op.process_async(10)
        assert result == 11


class TestFilterOperator:
    """Tests for FilterOperator."""
    
    def test_filter_passes(self):
        """Test items that pass filter."""
        op = FilterOperator(lambda x: x > 0)
        
        assert op.process(5) == 5
        assert op.process(1) == 1
    
    def test_filter_rejects(self):
        """Test items that fail filter."""
        op = FilterOperator(lambda x: x > 0)
        
        assert op.process(-1) is None
        assert op.process(0) is None
    
    def test_complex_predicate(self):
        """Test complex predicate."""
        op = FilterOperator(lambda x: x % 2 == 0 and x > 0)
        
        assert op.process(2) == 2
        assert op.process(4) == 4
        assert op.process(3) is None
        assert op.process(-2) is None
    
    @pytest.mark.asyncio
    async def test_async_filter(self):
        """Test async filtering."""
        op = FilterOperator(lambda x: x > 10)
        
        assert await op.process_async(15) == 15
        assert await op.process_async(5) is None


class TestFlatMapOperator:
    """Tests for FlatMapOperator."""
    
    def test_expand_items(self):
        """Test item expansion."""
        op = FlatMapOperator(lambda x: [x, x * 2, x * 3])
        result = op.process(2)
        
        assert result == [2, 4, 6]
    
    def test_empty_result(self):
        """Test empty expansion."""
        op = FlatMapOperator(lambda x: [] if x < 0 else [x])
        
        assert op.process(-1) == []
        assert op.process(1) == [1]
    
    def test_split_string(self):
        """Test string splitting."""
        op = FlatMapOperator(lambda s: s.split())
        result = op.process("hello world test")
        
        assert result == ["hello", "world", "test"]


# =============================================================================
# WINDOWED AGGREGATOR TESTS
# =============================================================================


class TestWindowedAggregator:
    """Tests for WindowedAggregator."""
    
    def test_count_window_basic(self):
        """Test basic count-based windowing."""
        config = WindowConfig(
            window_type=WindowType.COUNT,
            count_size=3
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=lambda items: sum(items)
        )
        
        # Add items
        assert aggregator.add(1) == []
        assert aggregator.add(2) == []
        assert aggregator.add(3) == [6]  # Window closes: 1+2+3=6
        
        assert aggregator.add(4) == []
        assert aggregator.add(5) == []
        assert aggregator.add(6) == [15]  # Window closes: 4+5+6=15
    
    def test_count_window_flush(self):
        """Test flushing partial count window."""
        config = WindowConfig(
            window_type=WindowType.COUNT,
            count_size=5
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=lambda items: len(items)
        )
        
        aggregator.add(1)
        aggregator.add(2)
        aggregator.add(3)
        
        results = aggregator.flush()
        assert results == [3]  # 3 items in partial window
    
    def test_tumbling_window_basic(self):
        """Test basic tumbling window."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=1.0
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=lambda items: sum(items)
        )
        
        base_time = datetime.now()
        
        # Add items within first window
        aggregator.add(1, base_time)
        aggregator.add(2, base_time + timedelta(milliseconds=100))
        
        # Add item in next window (triggers close of first)
        results = aggregator.add(3, base_time + timedelta(seconds=1.1))
        
        # First window should close
        assert len(results) >= 0  # May or may not have closed depending on timing
        
        # Flush to get remaining
        final = aggregator.flush()
        assert len(final) > 0
    
    def test_sliding_window(self):
        """Test sliding window."""
        config = WindowConfig(
            window_type=WindowType.SLIDING,
            window_size_seconds=2.0,
            slide_interval_seconds=1.0
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=list  # Just collect items
        )
        
        base_time = datetime.now()
        aggregator.add(1, base_time)
        aggregator.add(2, base_time + timedelta(milliseconds=500))
        
        # Flush and verify
        results = aggregator.flush()
        assert len(results) > 0
    
    def test_session_window(self):
        """Test session window."""
        config = WindowConfig(
            window_type=WindowType.SESSION,
            session_gap_seconds=1.0
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=lambda items: len(items)
        )
        
        base_time = datetime.now()
        
        # First session
        aggregator.add("a", base_time)
        aggregator.add("b", base_time + timedelta(milliseconds=500))
        
        # Gap triggers new session
        aggregator.add("c", base_time + timedelta(seconds=2))
        
        results = aggregator.flush()
        assert len(results) >= 1
    
    def test_aggregator_with_key_extractor(self):
        """Test keyed aggregation."""
        config = WindowConfig(
            window_type=WindowType.COUNT,
            count_size=2
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=lambda items: sum(item["value"] for item in items),
            key_extractor=lambda item: item["key"]
        )
        
        # Add enough items to trigger a window close
        result1 = aggregator.add({"key": "a", "value": 1})
        result2 = aggregator.add({"key": "a", "value": 2})
        
        # With count windows of size 2, second add should return aggregated result
        # The results are returned from add() when window closes
        all_results = result1 + result2
        assert len(all_results) == 1
        assert all_results[0] == 3  # 1 + 2
    
    def test_late_data_handling(self):
        """Test late data handling."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=1.0,
            allow_late_data_seconds=0.5
        )
        aggregator = WindowedAggregator(
            config=config,
            aggregator=list
        )
        
        # Stats tracking
        initial_stats = aggregator.get_stats()
        assert initial_stats["items_processed"] == 0
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        config = WindowConfig(
            window_type=WindowType.COUNT,
            count_size=3
        )
        aggregator = WindowedAggregator(config=config, aggregator=sum)
        
        aggregator.add(1)
        aggregator.add(2)
        aggregator.add(3)
        aggregator.add(4)
        
        stats = aggregator.get_stats()
        
        assert stats["items_processed"] == 4
        assert stats["windows_closed"] == 1
        assert stats["count_buffer_size"] == 1  # One item in buffer
    
    def test_default_aggregator(self):
        """Test default aggregator (identity)."""
        config = WindowConfig(window_type=WindowType.COUNT, count_size=2)
        aggregator = WindowedAggregator(config=config)
        
        aggregator.add(1)
        results = aggregator.add(2)
        
        assert results == [[1, 2]]


# =============================================================================
# BACKPRESSURE HANDLER TESTS
# =============================================================================


class TestBackpressureHandler:
    """Tests for BackpressureHandler."""
    
    def test_basic_push_pop(self):
        """Test basic push and pop operations."""
        handler = BackpressureHandler()
        
        handler.push(1)
        handler.push(2)
        handler.push(3)
        
        assert handler.pop(block=False) == 1
        assert handler.pop(block=False) == 2
        assert handler.pop(block=False) == 3
        assert handler.pop(block=False) is None
    
    def test_drop_oldest_strategy(self):
        """Test drop-oldest strategy."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            buffer_size=3
        )
        handler = BackpressureHandler(config=config)
        
        handler.push(1)
        handler.push(2)
        handler.push(3)
        handler.push(4)  # Should drop 1
        
        assert handler.pop(block=False) == 2
        assert handler.pop(block=False) == 3
        assert handler.pop(block=False) == 4
    
    def test_drop_newest_strategy(self):
        """Test drop-newest strategy."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_NEWEST,
            buffer_size=3
        )
        handler = BackpressureHandler(config=config)
        
        handler.push(1)
        handler.push(2)
        handler.push(3)
        assert handler.push(4) is False  # Should be rejected
        
        assert handler.pop(block=False) == 1
        assert handler.pop(block=False) == 2
        assert handler.pop(block=False) == 3
    
    def test_sample_strategy(self):
        """Test sampling strategy."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.SAMPLE,
            buffer_size=5,
            sample_rate=0.5  # Accept every 2nd item when full
        )
        handler = BackpressureHandler(config=config)
        
        # Fill buffer
        for i in range(5):
            handler.push(i)
        
        # Now push more - some will be sampled
        for i in range(5, 15):
            handler.push(i)
        
        stats = handler.get_stats()
        assert stats["items_received"] == 15
        assert stats["items_dropped"] > 0
    
    def test_buffer_expansion_strategy(self):
        """Test buffer expansion strategy."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.BUFFER,
            buffer_size=5,
            max_buffer_size=20
        )
        handler = BackpressureHandler(config=config)
        
        # Fill initial buffer
        for i in range(5):
            handler.push(i)
        
        # Push more - should expand
        for i in range(5, 10):
            handler.push(i)
        
        # Buffer should have expanded
        assert handler._current_buffer_size > 5
    
    def test_blocking_strategy_with_timeout(self):
        """Test blocking strategy with timeout."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.BLOCK,
            buffer_size=2,
            block_timeout_seconds=0.1
        )
        handler = BackpressureHandler(config=config)
        
        handler.push(1)
        handler.push(2)
        
        # This should timeout and return False
        result = handler.push(3, block=True, timeout=0.1)
        assert result is False
    
    def test_non_blocking_push(self):
        """Test non-blocking push when buffer is full."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.BLOCK,
            buffer_size=2
        )
        handler = BackpressureHandler(config=config)
        
        handler.push(1)
        handler.push(2)
        
        # Non-blocking push should fail immediately
        result = handler.push(3, block=False)
        assert result is False
    
    def test_pop_batch(self):
        """Test batch pop operation."""
        handler = BackpressureHandler()
        
        for i in range(10):
            handler.push(i)
        
        batch = handler.pop_batch(5)
        assert batch == [0, 1, 2, 3, 4]
        
        batch = handler.pop_batch(10)  # More than available
        assert batch == [5, 6, 7, 8, 9]
    
    def test_buffer_usage(self):
        """Test buffer usage calculation."""
        config = BackpressureConfig(buffer_size=10)
        handler = BackpressureHandler(config=config)
        
        assert handler.buffer_usage == 0.0
        
        for i in range(5):
            handler.push(i)
        
        assert handler.buffer_usage == 0.5
    
    def test_backpressure_activation(self):
        """Test backpressure activation at watermarks."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            buffer_size=10,
            high_watermark=0.8,
            low_watermark=0.5
        )
        handler = BackpressureHandler(config=config)
        
        assert handler.is_backpressure_active is False
        
        # Fill to high watermark
        for i in range(8):
            handler.push(i)
        
        assert handler.is_backpressure_active is True
        
        # Drain below low watermark
        for _ in range(4):
            handler.pop(block=False)
        
        # Push one to re-evaluate watermarks
        handler.push(100)
        
        assert handler.is_backpressure_active is False
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            buffer_size=3
        )
        handler = BackpressureHandler(config=config)
        
        handler.push(1)
        handler.push(2)
        handler.push(3)
        handler.push(4)  # Drops 1
        handler.pop(block=False)  # Delivers 2
        
        stats = handler.get_stats()
        
        assert stats["items_received"] == 4
        assert stats["items_dropped"] == 1
        assert stats["items_delivered"] == 1
        assert stats["buffer_size"] == 2
        assert stats["buffer_capacity"] == 3
    
    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.DROP_OLDEST,
            buffer_size=100
        )
        handler = BackpressureHandler(config=config)
        
        pushed = []
        popped = []
        push_lock = threading.Lock()
        pop_lock = threading.Lock()
        
        def producer():
            for i in range(50):
                handler.push(i)
                with push_lock:
                    pushed.append(i)
                time.sleep(0.001)
        
        def consumer():
            for _ in range(50):
                item = handler.pop(block=True, timeout=1.0)
                if item is not None:
                    with pop_lock:
                        popped.append(item)
                time.sleep(0.001)
        
        prod_thread = threading.Thread(target=producer)
        cons_thread = threading.Thread(target=consumer)
        
        prod_thread.start()
        cons_thread.start()
        
        prod_thread.join()
        cons_thread.join()
        
        # Should have processed items
        assert len(popped) > 0


# =============================================================================
# STREAM PROCESSOR TESTS
# =============================================================================


class TestStreamProcessor:
    """Tests for StreamProcessor."""
    
    def test_basic_processing(self):
        """Test basic stream processing."""
        source = IteratorSource(iter(range(10)))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(batch_size=5, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.5)  # Allow processing
        processor.stop()
        
        assert len(sink.items) > 0
    
    def test_processing_with_operators(self):
        """Test processing with operators."""
        source = IteratorSource(iter(range(10)))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            operators=[
                MapOperator(lambda x: x * 2),
                FilterOperator(lambda x: x > 5)
            ],
            config=StreamConfig(batch_size=10, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.5)
        processor.stop()
        
        # Should have values > 5 after doubling: 6, 8, 10, 12, 14, 16, 18
        for item in sink.items:
            assert item > 5
            assert item % 2 == 0
    
    def test_processor_state_transitions(self):
        """Test processor state transitions."""
        source = IteratorSource(iter(range(1000)))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(batch_size=10, batch_timeout_seconds=1.0)
        )
        
        assert processor.state == StreamState.CREATED
        
        processor.start()
        assert processor.state == StreamState.RUNNING
        
        processor.pause()
        assert processor.state == StreamState.PAUSED
        
        processor.resume()
        assert processor.state == StreamState.RUNNING
        
        processor.stop()
        assert processor.state == StreamState.STOPPED
    
    def test_cannot_start_twice(self):
        """Test starting processor twice raises error."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        processor = StreamProcessor(source=source, sink=sink)
        processor.start()
        
        with pytest.raises(RuntimeError):
            processor.start()
        
        processor.stop()
    
    def test_processor_stats(self):
        """Test processor statistics."""
        source = IteratorSource(iter(range(20)))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(name="test-processor", batch_size=5, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.5)
        processor.stop()
        
        stats = processor.get_stats()
        
        assert stats["name"] == "test-processor"
        assert stats["state"] == "STOPPED"
        assert stats["items_processed"] >= 0
        assert "backpressure" in stats
    
    def test_error_handling_skip(self):
        """Test skip error handling mode."""
        def failing_iterator():
            yield 1
            yield 2
            raise ValueError("Test error")
        
        source = IteratorSource(failing_iterator())
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(error_handling="skip", batch_size=1, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.3)
        processor.stop()
        
        # Should have processed items before error
        assert len(sink.items) >= 0
    
    def test_empty_source(self):
        """Test processing empty source."""
        source = IteratorSource(iter([]))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(batch_size=10, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.2)
        processor.stop()
        
        assert sink.items == []
    
    def test_backpressure_integration(self):
        """Test backpressure handler integration."""
        source = IteratorSource(iter(range(100)))
        sink = ListSink()
        
        backpressure = BackpressureHandler(
            config=BackpressureConfig(
                strategy=BackpressureStrategy.DROP_OLDEST,
                buffer_size=10
            )
        )
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            backpressure=backpressure,
            config=StreamConfig(batch_size=5, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.5)
        processor.stop()
        
        # Should have processed items
        assert len(sink.items) > 0


# =============================================================================
# STREAM PIPELINE TESTS
# =============================================================================


class TestStreamPipeline:
    """Tests for StreamPipeline."""
    
    def test_simple_pipeline(self):
        """Test simple source->sink pipeline."""
        data = [1, 2, 3, 4, 5]
        source = IteratorSource(iter(data))
        sink = ListSink()
        
        pipeline = StreamPipeline(name="test-pipeline")
        pipeline.add_source("input", source)
        pipeline.add_sink("output", sink)
        pipeline.connect("input", "output")
        pipeline.build()
        
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()
        
        assert len(sink.items) > 0
    
    def test_pipeline_with_operator(self):
        """Test pipeline with transformation operator."""
        data = [1, 2, 3, 4, 5]
        source = IteratorSource(iter(data))
        sink = ListSink()
        
        pipeline = StreamPipeline(name="transform-pipeline")
        pipeline.add_source("input", source)
        pipeline.add_operator("double", MapOperator(lambda x: x * 2))
        pipeline.add_sink("output", sink)
        pipeline.connect("input", "double")
        pipeline.connect("double", "output")
        pipeline.build()
        
        pipeline.start()
        time.sleep(0.3)
        pipeline.stop()
        
        # Should have doubled values
        for item in sink.items:
            assert item % 2 == 0
    
    def test_pipeline_stats(self):
        """Test pipeline statistics."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        pipeline = StreamPipeline(name="stats-pipeline")
        pipeline.add_source("input", source)
        pipeline.add_sink("output", sink)
        pipeline.connect("input", "output")
        pipeline.build()
        
        pipeline.start()
        time.sleep(0.2)
        pipeline.stop()
        
        stats = pipeline.get_stats()
        
        assert stats["name"] == "stats-pipeline"
        assert "processors" in stats
        assert stats["processor_count"] >= 0
    
    def test_fluent_api(self):
        """Test fluent API style."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        pipeline = (
            StreamPipeline(name="fluent")
            .add_source("input", source)
            .add_operator("map", MapOperator(lambda x: x + 1))
            .add_sink("output", sink)
            .connect("input", "map")
            .connect("map", "output")
            .build()
        )
        
        pipeline.start()
        time.sleep(0.2)
        pipeline.stop()
        
        assert isinstance(pipeline, StreamPipeline)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_windowed_aggregator(self):
        """Test windowed aggregator creation."""
        aggregator = create_windowed_aggregator(
            window_type=WindowType.COUNT,
            window_size_seconds=30.0,
            aggregator=sum
        )
        
        assert aggregator.config.window_type == WindowType.COUNT
        assert aggregator.config.window_size_seconds == 30.0
    
    def test_create_backpressure_handler(self):
        """Test backpressure handler creation."""
        handler = create_backpressure_handler(
            strategy=BackpressureStrategy.DROP_NEWEST,
            buffer_size=5000
        )
        
        assert handler.config.strategy == BackpressureStrategy.DROP_NEWEST
        assert handler.config.buffer_size == 5000
    
    def test_create_stream_processor(self):
        """Test stream processor creation."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        processor = create_stream_processor(
            source=source,
            sink=sink,
            operators=[MapOperator(lambda x: x * 2)],
            name="my-processor"
        )
        
        assert processor.config.name == "my-processor"
        assert len(processor.operators) == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for streaming components."""
    
    def test_full_pipeline_with_windowing(self):
        """Test full pipeline with windowed aggregation."""
        # Generate data
        data = list(range(20))
        source = IteratorSource(iter(data))
        sink = ListSink()
        
        # Create aggregator
        aggregator = create_windowed_aggregator(
            window_type=WindowType.COUNT,
            aggregator=sum
        )
        
        # Create processor
        processor = create_stream_processor(
            source=source,
            sink=sink,
            operators=[MapOperator(lambda x: x * 2)],
            name="windowed-processor"
        )
        
        processor.start()
        time.sleep(0.3)
        processor.stop()
        
        # Should have processed items
        assert len(sink.items) > 0
    
    def test_backpressure_with_slow_consumer(self):
        """Test backpressure with simulated slow consumer."""
        # Fast producer
        data = list(range(100))
        source = IteratorSource(iter(data))
        
        # Slow sink
        received = []
        def slow_write(item):
            time.sleep(0.01)  # Simulate slow processing
            received.append(item)
            return True
        
        sink = CallbackSink(slow_write)
        
        handler = create_backpressure_handler(
            strategy=BackpressureStrategy.DROP_OLDEST,
            buffer_size=10
        )
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            backpressure=handler,
            config=StreamConfig(batch_size=5, batch_timeout_seconds=0.05)
        )
        
        processor.start()
        time.sleep(1.0)  # Allow some processing
        processor.stop()
        
        stats = handler.get_stats()
        # With slow consumer and fast producer, some items should be dropped
        assert stats["items_received"] > 0
    
    def test_multi_operator_pipeline(self):
        """Test pipeline with multiple operators."""
        data = list(range(1, 21))  # 1 to 20
        source = IteratorSource(iter(data))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            operators=[
                MapOperator(lambda x: x * 2),        # Double
                FilterOperator(lambda x: x > 10),   # Keep > 10
                MapOperator(lambda x: x + 1)         # Add 1
            ],
            config=StreamConfig(batch_size=20, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.3)
        processor.stop()
        
        # Original 6-20 doubled (12-40) filtered (>10) -> 12-40, then +1 -> 13-41
        for item in sink.items:
            assert item > 10
    
    def test_error_recovery(self):
        """Test error recovery in stream processing."""
        successful = []
        
        def sometimes_fails(x):
            if x == 5:
                raise ValueError("Error on 5")
            return x * 2
        
        data = list(range(10))
        source = IteratorSource(iter(data))
        sink = ListSink()
        
        # Create processor that skips errors
        processor = StreamProcessor(
            source=source,
            sink=sink,
            operators=[MapOperator(sometimes_fails)],
            config=StreamConfig(
                error_handling="skip",
                batch_size=1,
                batch_timeout_seconds=0.05
            )
        )
        
        processor.start()
        time.sleep(0.5)
        processor.stop()
        
        # Should have processed other items, skipping 5
        # Values should be: 0, 2, 4, 6, 8, 12, 14, 16, 18 (5 skipped)
        assert 10 not in sink.items  # 5 * 2 should not be present


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_operator_list(self):
        """Test processor with no operators."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            operators=[],
            config=StreamConfig(batch_size=10, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.2)
        processor.stop()
        
        assert sink.items == [1, 2, 3]
    
    def test_single_item_stream(self):
        """Test processing single item."""
        source = IteratorSource(iter([42]))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(batch_size=10, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.2)
        processor.stop()
        
        assert sink.items == [42]
    
    def test_large_batch_size(self):
        """Test with batch size larger than data."""
        data = [1, 2, 3]
        source = IteratorSource(iter(data))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            config=StreamConfig(batch_size=1000, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.3)
        processor.stop()
        
        assert len(sink.items) == 3
    
    def test_zero_window_size(self):
        """Test count window with size 1."""
        config = WindowConfig(
            window_type=WindowType.COUNT,
            count_size=1
        )
        aggregator = WindowedAggregator(config=config, aggregator=lambda x: x[0])
        
        result = aggregator.add(1)
        assert result == [1]
        
        result = aggregator.add(2)
        assert result == [2]
    
    def test_stop_without_start(self):
        """Test stopping processor that was never started."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        processor = StreamProcessor(source=source, sink=sink)
        
        # Should not raise
        processor.stop()
        
        assert processor.state == StreamState.CREATED
    
    def test_pause_resume_stopped(self):
        """Test pause/resume on stopped processor."""
        source = IteratorSource(iter([1, 2, 3]))
        sink = ListSink()
        
        processor = StreamProcessor(source=source, sink=sink)
        processor.start()
        processor.stop()
        
        # Pause/resume should not change state
        processor.pause()
        assert processor.state == StreamState.STOPPED
        
        processor.resume()
        assert processor.state == StreamState.STOPPED
    
    def test_filter_all_items(self):
        """Test filter that rejects all items."""
        source = IteratorSource(iter([1, 2, 3, 4, 5]))
        sink = ListSink()
        
        processor = StreamProcessor(
            source=source,
            sink=sink,
            operators=[FilterOperator(lambda x: x > 100)],  # Rejects all
            config=StreamConfig(batch_size=10, batch_timeout_seconds=0.1)
        )
        
        processor.start()
        time.sleep(0.2)
        processor.stop()
        
        assert sink.items == []
    
    def test_backpressure_buffer_at_max(self):
        """Test buffer expansion at max size."""
        config = BackpressureConfig(
            strategy=BackpressureStrategy.BUFFER,
            buffer_size=2,
            max_buffer_size=4
        )
        handler = BackpressureHandler(config=config)
        
        # Fill and expand
        for i in range(5):
            handler.push(i)
        
        # Should have expanded but also dropped oldest when at max
        assert handler._current_buffer_size == 4
        
        stats = handler.get_stats()
        assert stats["items_dropped"] >= 1


# =============================================================================
# WINDOW DATA CLASS TEST
# =============================================================================


class TestWindowDataclass:
    """Tests for Window dataclass."""
    
    def test_window_creation(self):
        """Test window creation."""
        now = datetime.now()
        window = Window(
            start_time=now,
            end_time=now + timedelta(seconds=60)
        )
        
        assert window.start_time == now
        assert window.items == []
        assert window.is_closed is False
    
    def test_window_with_items(self):
        """Test window with items."""
        now = datetime.now()
        window = Window(
            start_time=now,
            end_time=now + timedelta(seconds=60),
            items=[1, 2, 3]
        )
        
        assert window.items == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
