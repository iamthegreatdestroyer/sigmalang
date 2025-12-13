"""
Streaming Processor Module for SigmaLang.

This module provides real-time data streaming capabilities with
windowed aggregation and backpressure handling.

Components:
- StreamProcessor: Core streaming processor with async support
- WindowedAggregator: Time-based and count-based windowing
- BackpressureHandler: Flow control and buffer management
- StreamSource: Abstract data source interface
- StreamSink: Abstract data sink interface
- StreamPipeline: End-to-end streaming pipeline
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================


T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")


# =============================================================================
# ENUMS
# =============================================================================


class WindowType(Enum):
    """Types of streaming windows."""
    
    TUMBLING = auto()    # Non-overlapping fixed-size windows
    SLIDING = auto()     # Overlapping windows with slide interval
    SESSION = auto()     # Gap-based windows
    COUNT = auto()       # Count-based windows


class BackpressureStrategy(Enum):
    """Strategies for handling backpressure."""
    
    BLOCK = auto()       # Block producer until consumer catches up
    DROP_OLDEST = auto() # Drop oldest items when buffer is full
    DROP_NEWEST = auto() # Drop newest items when buffer is full
    SAMPLE = auto()      # Sample items at a rate
    BUFFER = auto()      # Expand buffer (up to limit)


class StreamState(Enum):
    """State of a stream processor."""
    
    CREATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WindowConfig:
    """Configuration for windowed aggregation."""
    
    window_type: WindowType = WindowType.TUMBLING
    window_size_seconds: float = 60.0
    slide_interval_seconds: float = 30.0  # For sliding windows
    session_gap_seconds: float = 30.0     # For session windows
    count_size: int = 100                 # For count windows
    allow_late_data_seconds: float = 0.0  # Grace period for late arrivals


@dataclass
class BackpressureConfig:
    """Configuration for backpressure handling."""
    
    strategy: BackpressureStrategy = BackpressureStrategy.BLOCK
    buffer_size: int = 10000
    max_buffer_size: int = 100000
    high_watermark: float = 0.8  # Start backpressure at 80% full
    low_watermark: float = 0.5   # Stop backpressure at 50% full
    sample_rate: float = 0.1    # For SAMPLE strategy
    block_timeout_seconds: float = 30.0


@dataclass
class StreamConfig:
    """Configuration for stream processing."""
    
    name: str = "default-stream"
    batch_size: int = 100
    batch_timeout_seconds: float = 1.0
    max_concurrent_tasks: int = 4
    enable_checkpointing: bool = False
    checkpoint_interval_seconds: float = 60.0
    error_handling: str = "skip"  # skip, retry, fail


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================


class StreamSource(ABC, Generic[T]):
    """Abstract base class for stream data sources."""
    
    @abstractmethod
    def read(self) -> Optional[T]:
        """Read a single item from the source."""
        pass
    
    @abstractmethod
    async def read_async(self) -> Optional[T]:
        """Read a single item asynchronously."""
        pass
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over source items."""
        while True:
            item = self.read()
            if item is None:
                break
            yield item
    
    async def __aiter__(self) -> AsyncIterator[T]:
        """Async iterate over source items."""
        while True:
            item = await self.read_async()
            if item is None:
                break
            yield item


class StreamSink(ABC, Generic[T]):
    """Abstract base class for stream data sinks."""
    
    @abstractmethod
    def write(self, item: T) -> bool:
        """Write a single item to the sink."""
        pass
    
    @abstractmethod
    async def write_async(self, item: T) -> bool:
        """Write a single item asynchronously."""
        pass
    
    def write_batch(self, items: List[T]) -> int:
        """Write a batch of items. Returns count of successful writes."""
        count = 0
        for item in items:
            if self.write(item):
                count += 1
        return count
    
    async def write_batch_async(self, items: List[T]) -> int:
        """Write a batch asynchronously."""
        count = 0
        for item in items:
            if await self.write_async(item):
                count += 1
        return count


class StreamOperator(ABC, Generic[T, U]):
    """Abstract base class for stream operators."""
    
    @abstractmethod
    def process(self, item: T) -> Optional[U]:
        """Process a single item."""
        pass
    
    @abstractmethod
    async def process_async(self, item: T) -> Optional[U]:
        """Process a single item asynchronously."""
        pass


# =============================================================================
# CONCRETE SOURCES AND SINKS
# =============================================================================


class IteratorSource(StreamSource[T]):
    """Stream source from an iterator."""
    
    def __init__(self, iterator: Iterator[T]):
        self._iterator = iterator
        self._lock = threading.Lock()
    
    def read(self) -> Optional[T]:
        with self._lock:
            try:
                return next(self._iterator)
            except StopIteration:
                return None
    
    async def read_async(self) -> Optional[T]:
        return self.read()


class ListSink(StreamSink[T]):
    """Stream sink that collects items in a list."""
    
    def __init__(self):
        self._items: List[T] = []
        self._lock = threading.Lock()
    
    def write(self, item: T) -> bool:
        with self._lock:
            self._items.append(item)
            return True
    
    async def write_async(self, item: T) -> bool:
        return self.write(item)
    
    @property
    def items(self) -> List[T]:
        """Get collected items."""
        with self._lock:
            return list(self._items)
    
    def clear(self) -> None:
        """Clear collected items."""
        with self._lock:
            self._items.clear()


class CallbackSink(StreamSink[T]):
    """Stream sink that calls a callback for each item."""
    
    def __init__(self, callback: Callable[[T], bool]):
        self._callback = callback
    
    def write(self, item: T) -> bool:
        try:
            return self._callback(item)
        except Exception as e:
            logger.error(f"Callback sink error: {e}")
            return False
    
    async def write_async(self, item: T) -> bool:
        return self.write(item)


# =============================================================================
# STREAM OPERATORS
# =============================================================================


class MapOperator(StreamOperator[T, U]):
    """Transforms each item using a function."""
    
    def __init__(self, func: Callable[[T], U]):
        self._func = func
    
    def process(self, item: T) -> Optional[U]:
        return self._func(item)
    
    async def process_async(self, item: T) -> Optional[U]:
        return self._func(item)


class FilterOperator(StreamOperator[T, T]):
    """Filters items based on a predicate."""
    
    def __init__(self, predicate: Callable[[T], bool]):
        self._predicate = predicate
    
    def process(self, item: T) -> Optional[T]:
        return item if self._predicate(item) else None
    
    async def process_async(self, item: T) -> Optional[T]:
        return self.process(item)


class FlatMapOperator(StreamOperator[T, List[U]]):
    """Transforms each item into zero or more items."""
    
    def __init__(self, func: Callable[[T], List[U]]):
        self._func = func
    
    def process(self, item: T) -> Optional[List[U]]:
        return self._func(item)
    
    async def process_async(self, item: T) -> Optional[List[U]]:
        return self._func(item)


# =============================================================================
# WINDOWED AGGREGATOR
# =============================================================================


@dataclass
class Window(Generic[T]):
    """Represents a single window of items."""
    
    start_time: datetime
    end_time: datetime
    items: List[T] = field(default_factory=list)
    is_closed: bool = False


class WindowedAggregator(Generic[T, U]):
    """
    Aggregates stream items over time or count-based windows.
    
    Supports tumbling, sliding, session, and count-based windows
    with configurable aggregation functions.
    
    Example:
        >>> aggregator = WindowedAggregator(
        ...     config=WindowConfig(window_type=WindowType.TUMBLING, window_size_seconds=60),
        ...     aggregator=lambda items: sum(items)
        ... )  # doctest: +SKIP
        >>> aggregator.add(1)  # doctest: +SKIP
        >>> aggregator.add(2)  # doctest: +SKIP
        >>> results = aggregator.flush()  # doctest: +SKIP
    """
    
    def __init__(
        self,
        config: Optional[WindowConfig] = None,
        aggregator: Optional[Callable[[List[T]], U]] = None,
        key_extractor: Optional[Callable[[T], K]] = None
    ):
        self.config = config or WindowConfig()
        self._aggregator = aggregator or (lambda x: x)  # type: ignore
        self._key_extractor = key_extractor
        
        self._windows: Dict[Any, Window[T]] = {}
        self._count_buffer: List[T] = []
        self._lock = threading.Lock()
        
        self._items_processed = 0
        self._windows_closed = 0
        self._late_items_dropped = 0
    
    def add(self, item: T, timestamp: Optional[datetime] = None) -> List[U]:
        """
        Add an item to the aggregator.
        
        Returns list of aggregated results for any closed windows.
        """
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            self._items_processed += 1
            
            if self.config.window_type == WindowType.COUNT:
                return self._add_count_window(item)
            else:
                return self._add_time_window(item, timestamp)
    
    def _add_count_window(self, item: T) -> List[U]:
        """Add item to count-based window."""
        self._count_buffer.append(item)
        
        results = []
        while len(self._count_buffer) >= self.config.count_size:
            window_items = self._count_buffer[:self.config.count_size]
            self._count_buffer = self._count_buffer[self.config.count_size:]
            
            result = self._aggregator(window_items)
            results.append(result)
            self._windows_closed += 1
        
        return results
    
    def _add_time_window(self, item: T, timestamp: datetime) -> List[U]:
        """Add item to time-based window."""
        key = self._key_extractor(item) if self._key_extractor else "default"
        
        results = []
        
        # Check for closed windows first
        closed_windows = self._close_expired_windows(timestamp)
        for window in closed_windows:
            if window.items:
                result = self._aggregator(window.items)
                results.append(result)
        
        # Get or create current window
        window = self._get_or_create_window(key, timestamp)
        
        if window:
            # Check for late data
            if timestamp < window.start_time:
                grace_period = timedelta(seconds=self.config.allow_late_data_seconds)
                if timestamp < window.start_time - grace_period:
                    self._late_items_dropped += 1
                    return results
            
            window.items.append(item)
        
        return results
    
    def _get_or_create_window(
        self,
        key: Any,
        timestamp: datetime
    ) -> Optional[Window[T]]:
        """Get existing window or create new one."""
        if self.config.window_type == WindowType.TUMBLING:
            window_duration = timedelta(seconds=self.config.window_size_seconds)
            window_start = datetime.fromtimestamp(
                (timestamp.timestamp() // self.config.window_size_seconds) 
                * self.config.window_size_seconds
            )
            window_end = window_start + window_duration
            
        elif self.config.window_type == WindowType.SLIDING:
            # For sliding windows, create multiple overlapping windows
            window_duration = timedelta(seconds=self.config.window_size_seconds)
            window_start = datetime.fromtimestamp(
                (timestamp.timestamp() // self.config.slide_interval_seconds)
                * self.config.slide_interval_seconds
            )
            window_end = window_start + window_duration
            
        elif self.config.window_type == WindowType.SESSION:
            # Session windows are key-based
            existing = self._windows.get(key)
            if existing and not existing.is_closed:
                gap = timedelta(seconds=self.config.session_gap_seconds)
                if timestamp <= existing.end_time + gap:
                    # Extend existing session
                    existing.end_time = timestamp + gap
                    return existing
            
            # Create new session
            gap = timedelta(seconds=self.config.session_gap_seconds)
            window_start = timestamp
            window_end = timestamp + gap
        else:
            return None
        
        window_key = (key, window_start)
        
        if window_key not in self._windows:
            self._windows[window_key] = Window(
                start_time=window_start,
                end_time=window_end,
                items=[]
            )
        
        return self._windows[window_key]
    
    def _close_expired_windows(self, current_time: datetime) -> List[Window[T]]:
        """Close windows that have expired."""
        closed = []
        grace_period = timedelta(seconds=self.config.allow_late_data_seconds)
        
        for key, window in list(self._windows.items()):
            if not window.is_closed and current_time > window.end_time + grace_period:
                window.is_closed = True
                closed.append(window)
                del self._windows[key]
                self._windows_closed += 1
        
        return closed
    
    def flush(self) -> List[U]:
        """Flush all open windows and return aggregated results."""
        with self._lock:
            results = []
            
            # Flush time-based windows
            for window in self._windows.values():
                if window.items:
                    result = self._aggregator(window.items)
                    results.append(result)
            self._windows.clear()
            
            # Flush count-based buffer
            if self._count_buffer:
                result = self._aggregator(self._count_buffer)
                results.append(result)
                self._count_buffer.clear()
            
            return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        with self._lock:
            return {
                "items_processed": self._items_processed,
                "windows_closed": self._windows_closed,
                "open_windows": len(self._windows),
                "count_buffer_size": len(self._count_buffer),
                "late_items_dropped": self._late_items_dropped,
            }


# =============================================================================
# BACKPRESSURE HANDLER
# =============================================================================


class BackpressureHandler(Generic[T]):
    """
    Handles backpressure in streaming pipelines.
    
    Implements various strategies for handling situations where
    producers are faster than consumers.
    
    Example:
        >>> handler = BackpressureHandler(config=BackpressureConfig(
        ...     strategy=BackpressureStrategy.DROP_OLDEST,
        ...     buffer_size=1000
        ... ))
        >>> item = "test_data"
        >>> handler.push(item)  # May drop old items if buffer full
        >>> retrieved = handler.pop()  # Get next item
        >>> retrieved == item
        True
    """
    
    def __init__(self, config: Optional[BackpressureConfig] = None):
        self.config = config or BackpressureConfig()
        
        self._buffer: Deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        self._current_buffer_size = self.config.buffer_size
        self._items_received = 0
        self._items_dropped = 0
        self._items_delivered = 0
        self._backpressure_active = False
        
        # For SAMPLE strategy
        self._sample_counter = 0
    
    def push(self, item: T, block: bool = True, timeout: float = 0.0) -> bool:
        """
        Push an item to the buffer.
        
        Returns True if item was accepted, False if dropped or timeout.
        """
        with self._lock:
            self._items_received += 1
            
            # Check buffer state
            buffer_usage = len(self._buffer) / self._current_buffer_size
            
            # Activate backpressure if above high watermark
            if buffer_usage >= self.config.high_watermark:
                self._backpressure_active = True
            elif buffer_usage <= self.config.low_watermark:
                self._backpressure_active = False
            
            # Apply strategy
            if self.config.strategy == BackpressureStrategy.BLOCK:
                return self._push_blocking(item, block, timeout)
            elif self.config.strategy == BackpressureStrategy.DROP_OLDEST:
                return self._push_drop_oldest(item)
            elif self.config.strategy == BackpressureStrategy.DROP_NEWEST:
                return self._push_drop_newest(item)
            elif self.config.strategy == BackpressureStrategy.SAMPLE:
                return self._push_sample(item)
            elif self.config.strategy == BackpressureStrategy.BUFFER:
                return self._push_expand_buffer(item)
            else:
                self._buffer.append(item)
                return True
    
    def _push_blocking(
        self,
        item: T,
        block: bool,
        timeout: float
    ) -> bool:
        """Push with blocking strategy."""
        if len(self._buffer) >= self._current_buffer_size:
            if not block:
                self._items_dropped += 1
                return False
            
            # Wait for space
            actual_timeout = timeout if timeout > 0 else self.config.block_timeout_seconds
            if not self._not_full.wait(timeout=actual_timeout):
                self._items_dropped += 1
                return False
        
        self._buffer.append(item)
        self._not_empty.notify()
        return True
    
    def _push_drop_oldest(self, item: T) -> bool:
        """Push with drop-oldest strategy."""
        if len(self._buffer) >= self._current_buffer_size:
            self._buffer.popleft()
            self._items_dropped += 1
        
        self._buffer.append(item)
        self._not_empty.notify()
        return True
    
    def _push_drop_newest(self, item: T) -> bool:
        """Push with drop-newest strategy."""
        if len(self._buffer) >= self._current_buffer_size:
            self._items_dropped += 1
            return False
        
        self._buffer.append(item)
        self._not_empty.notify()
        return True
    
    def _push_sample(self, item: T) -> bool:
        """Push with sampling strategy."""
        self._sample_counter += 1
        
        # Only accept items at sample rate when buffer is full
        if len(self._buffer) >= self._current_buffer_size:
            sample_threshold = int(1.0 / self.config.sample_rate)
            if self._sample_counter % sample_threshold != 0:
                self._items_dropped += 1
                return False
            
            # Drop oldest to make room
            self._buffer.popleft()
        
        self._buffer.append(item)
        self._not_empty.notify()
        return True
    
    def _push_expand_buffer(self, item: T) -> bool:
        """Push with buffer expansion strategy."""
        if len(self._buffer) >= self._current_buffer_size:
            if self._current_buffer_size < self.config.max_buffer_size:
                # Expand buffer
                self._current_buffer_size = min(
                    self._current_buffer_size * 2,
                    self.config.max_buffer_size
                )
                logger.info(f"Expanded buffer to {self._current_buffer_size}")
            else:
                # Buffer at max, drop oldest
                self._buffer.popleft()
                self._items_dropped += 1
        
        self._buffer.append(item)
        self._not_empty.notify()
        return True
    
    def pop(self, block: bool = True, timeout: float = 0.0) -> Optional[T]:
        """
        Pop an item from the buffer.
        
        Returns None if no item available and not blocking.
        """
        with self._not_empty:
            if not self._buffer:
                if not block:
                    return None
                
                if not self._not_empty.wait(timeout=timeout if timeout > 0 else None):
                    return None
                
                if not self._buffer:
                    return None
            
            item = self._buffer.popleft()
            self._items_delivered += 1
            
            # Notify producers if using blocking strategy
            if self.config.strategy == BackpressureStrategy.BLOCK:
                self._not_full.notify()
            
            return item
    
    def pop_batch(self, max_items: int, timeout: float = 0.0) -> List[T]:
        """Pop a batch of items from the buffer."""
        with self._not_empty:
            if not self._buffer:
                if timeout > 0:
                    self._not_empty.wait(timeout=timeout)
            
            items = []
            for _ in range(min(max_items, len(self._buffer))):
                items.append(self._buffer.popleft())
                self._items_delivered += 1
            
            if self.config.strategy == BackpressureStrategy.BLOCK:
                self._not_full.notify_all()
            
            return items
    
    @property
    def is_backpressure_active(self) -> bool:
        """Check if backpressure is currently active."""
        return self._backpressure_active
    
    @property
    def buffer_usage(self) -> float:
        """Get current buffer usage as percentage."""
        with self._lock:
            return len(self._buffer) / self._current_buffer_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "buffer_capacity": self._current_buffer_size,
                "buffer_usage": len(self._buffer) / self._current_buffer_size,
                "items_received": self._items_received,
                "items_dropped": self._items_dropped,
                "items_delivered": self._items_delivered,
                "backpressure_active": self._backpressure_active,
                "drop_rate": (
                    self._items_dropped / self._items_received 
                    if self._items_received > 0 else 0.0
                ),
            }


# =============================================================================
# STREAM PROCESSOR
# =============================================================================


class StreamProcessor(Generic[T, U]):
    """
    Core stream processor with async support.
    
    Processes streaming data through a pipeline of operators
    with configurable batching, error handling, and checkpointing.
    
    Example:
        >>> processor = StreamProcessor(
        ...     source=IteratorSource(iter(range(100))),
        ...     sink=ListSink(),
        ...     operators=[MapOperator(lambda x: x * 2)]
        ... )
        >>> processor.start()
        >>> processor.stop()
    """
    
    def __init__(
        self,
        source: StreamSource[T],
        sink: StreamSink[U],
        operators: Optional[List[StreamOperator]] = None,
        config: Optional[StreamConfig] = None,
        backpressure: Optional[BackpressureHandler] = None
    ):
        self.source = source
        self.sink = sink
        self.operators = operators or []
        self.config = config or StreamConfig()
        self.backpressure = backpressure or BackpressureHandler()
        
        self._state = StreamState.CREATED
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        
        self._executor: Optional[ThreadPoolExecutor] = None
        self._processing_thread: Optional[threading.Thread] = None
        
        self._items_processed = 0
        self._items_failed = 0
        self._batches_processed = 0
        self._last_error: Optional[Exception] = None
        self._start_time: Optional[datetime] = None
    
    @property
    def state(self) -> StreamState:
        """Get current processor state."""
        return self._state
    
    def start(self) -> None:
        """Start the stream processor."""
        with self._lock:
            if self._state != StreamState.CREATED:
                raise RuntimeError(f"Cannot start processor in state {self._state}")
            
            self._state = StreamState.RUNNING
            self._start_time = datetime.now()
            self._stop_event.clear()
            
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_tasks
            )
            
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self._processing_thread.start()
            
            logger.info(f"StreamProcessor '{self.config.name}' started")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the stream processor."""
        with self._lock:
            if self._state not in (StreamState.RUNNING, StreamState.PAUSED):
                return
            
            self._state = StreamState.STOPPED
            self._stop_event.set()
            self._pause_event.set()  # Unpause to allow clean shutdown
        
        if self._processing_thread:
            self._processing_thread.join(timeout=timeout)
        
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.info(f"StreamProcessor '{self.config.name}' stopped")
    
    def pause(self) -> None:
        """Pause the stream processor."""
        with self._lock:
            if self._state == StreamState.RUNNING:
                self._state = StreamState.PAUSED
                self._pause_event.clear()
                logger.info(f"StreamProcessor '{self.config.name}' paused")
    
    def resume(self) -> None:
        """Resume the stream processor."""
        with self._lock:
            if self._state == StreamState.PAUSED:
                self._state = StreamState.RUNNING
                self._pause_event.set()
                logger.info(f"StreamProcessor '{self.config.name}' resumed")
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        batch: List[T] = []
        last_batch_time = time.time()
        
        while not self._stop_event.is_set():
            # Wait if paused
            self._pause_event.wait()
            
            if self._stop_event.is_set():
                break
            
            try:
                # Read from source
                item = self.source.read()
                
                if item is not None:
                    batch.append(item)
                
                # Process batch if ready
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and time.time() - last_batch_time >= self.config.batch_timeout_seconds) or
                    (item is None and batch)
                )
                
                if should_process:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                
                # End of source
                if item is None:
                    break
                    
            except Exception as e:
                self._handle_error(e)
                if self.config.error_handling == "fail":
                    break
        
        # Process any remaining items
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[T]) -> None:
        """Process a batch of items."""
        for item in batch:
            try:
                result = self._apply_operators(item)
                
                if result is not None:
                    # Handle backpressure
                    self.backpressure.push(result)
                    
                    # Drain from backpressure buffer to sink
                    while True:
                        output = self.backpressure.pop(block=False)
                        if output is None:
                            break
                        self.sink.write(output)
                    
                    self._items_processed += 1
                    
            except Exception as e:
                self._items_failed += 1
                self._handle_error(e)
                
                if self.config.error_handling == "fail":
                    raise
        
        self._batches_processed += 1
    
    def _apply_operators(self, item: Any) -> Any:
        """Apply all operators to an item."""
        result = item
        
        for operator in self.operators:
            if result is None:
                return None
            
            if isinstance(operator, FlatMapOperator):
                # FlatMapOperator returns list, take first for now
                results = operator.process(result)
                if results:
                    result = results[0]  # Simplified
                else:
                    return None
            else:
                result = operator.process(result)
        
        return result
    
    def _handle_error(self, error: Exception) -> None:
        """Handle processing error."""
        self._last_error = error
        logger.error(f"StreamProcessor error: {error}")
        
        if self.config.error_handling == "fail":
            with self._lock:
                self._state = StreamState.ERROR
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        runtime = 0.0
        if self._start_time:
            runtime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "name": self.config.name,
            "state": self._state.name,
            "items_processed": self._items_processed,
            "items_failed": self._items_failed,
            "batches_processed": self._batches_processed,
            "runtime_seconds": runtime,
            "throughput_per_second": (
                self._items_processed / runtime if runtime > 0 else 0.0
            ),
            "last_error": str(self._last_error) if self._last_error else None,
            "backpressure": self.backpressure.get_stats(),
        }


# =============================================================================
# STREAM PIPELINE
# =============================================================================


class StreamPipeline:
    """
    End-to-end streaming pipeline with multiple stages.
    
    Example:
        >>> pipeline = StreamPipeline(name="my-pipeline")  # doctest: +SKIP
        >>> data = [1, 2, 3]  # doctest: +SKIP
        >>> pipeline.add_source("input", IteratorSource(data))  # doctest: +SKIP
        >>> pipeline.add_operator("transform", MapOperator(lambda x: x * 2))  # doctest: +SKIP
        >>> pipeline.add_sink("output", ListSink())  # doctest: +SKIP
        >>> pipeline.connect("input", "transform")  # doctest: +SKIP
        >>> pipeline.connect("transform", "output")  # doctest: +SKIP
        >>> pipeline.start()  # doctest: +SKIP
    """
    
    def __init__(self, name: str = "pipeline"):
        self.name = name
        
        self._sources: Dict[str, StreamSource] = {}
        self._operators: Dict[str, StreamOperator] = {}
        self._sinks: Dict[str, StreamSink] = {}
        self._connections: Dict[str, List[str]] = {}
        
        self._processors: List[StreamProcessor] = []
        self._state = StreamState.CREATED
    
    def add_source(self, name: str, source: StreamSource) -> StreamPipeline:
        """Add a source to the pipeline."""
        self._sources[name] = source
        return self
    
    def add_operator(self, name: str, operator: StreamOperator) -> StreamPipeline:
        """Add an operator to the pipeline."""
        self._operators[name] = operator
        return self
    
    def add_sink(self, name: str, sink: StreamSink) -> StreamPipeline:
        """Add a sink to the pipeline."""
        self._sinks[name] = sink
        return self
    
    def connect(self, from_node: str, to_node: str) -> StreamPipeline:
        """Connect two nodes in the pipeline."""
        if from_node not in self._connections:
            self._connections[from_node] = []
        self._connections[from_node].append(to_node)
        return self
    
    def build(self) -> StreamPipeline:
        """Build the pipeline from connections."""
        # Simple implementation: create processors for each source->sink path
        for source_name, source in self._sources.items():
            # Find connected operators and sink
            operators = []
            current = source_name
            
            while current in self._connections:
                next_nodes = self._connections[current]
                if not next_nodes:
                    break
                
                next_node = next_nodes[0]  # Take first connection
                
                if next_node in self._operators:
                    operators.append(self._operators[next_node])
                    current = next_node
                elif next_node in self._sinks:
                    sink = self._sinks[next_node]
                    
                    processor = StreamProcessor(
                        source=source,
                        sink=sink,
                        operators=operators,
                        config=StreamConfig(name=f"{self.name}:{source_name}->{next_node}")
                    )
                    self._processors.append(processor)
                    break
                else:
                    break
        
        return self
    
    def start(self) -> None:
        """Start all processors in the pipeline."""
        self._state = StreamState.RUNNING
        for processor in self._processors:
            processor.start()
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop all processors in the pipeline."""
        self._state = StreamState.STOPPED
        for processor in self._processors:
            processor.stop(timeout=timeout / len(self._processors) if self._processors else timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "name": self.name,
            "state": self._state.name,
            "processor_count": len(self._processors),
            "processors": [p.get_stats() for p in self._processors],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_windowed_aggregator(
    window_type: WindowType = WindowType.TUMBLING,
    window_size_seconds: float = 60.0,
    aggregator: Optional[Callable[[List[T]], U]] = None
) -> WindowedAggregator[T, U]:
    """Create a windowed aggregator with common defaults."""
    config = WindowConfig(
        window_type=window_type,
        window_size_seconds=window_size_seconds
    )
    return WindowedAggregator(config=config, aggregator=aggregator)


def create_backpressure_handler(
    strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
    buffer_size: int = 10000
) -> BackpressureHandler:
    """Create a backpressure handler with common defaults."""
    config = BackpressureConfig(
        strategy=strategy,
        buffer_size=buffer_size
    )
    return BackpressureHandler(config=config)


def create_stream_processor(
    source: StreamSource[T],
    sink: StreamSink[U],
    operators: Optional[List[StreamOperator]] = None,
    name: str = "processor"
) -> StreamProcessor[T, U]:
    """Create a stream processor with common defaults."""
    config = StreamConfig(name=name)
    return StreamProcessor(
        source=source,
        sink=sink,
        operators=operators,
        config=config
    )
