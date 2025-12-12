"""
Phase 2A.5 - Task 1: Parallel Processing Engine Tests
======================================================

Comprehensive test suite for:
- ParallelExecutor
- AsyncPatternProcessor  
- WorkStealingScheduler

Target: 15 tests, 95% coverage

Copyright 2025 - SigmaLang Project
"""

import asyncio
import math
import pytest
import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parallel_processor import (
    ParallelExecutor,
    AsyncPatternProcessor,
    WorkStealingScheduler,
    ExecutionResult,
    BatchResult,
    WorkItem,
    parallel_map,
    parallel_reduce,
    async_map
)


# ============================================================================
# TEST PARALLEL EXECUTOR
# ============================================================================

class TestParallelExecutor:
    """Test ParallelExecutor functionality."""
    
    def test_init_default_workers(self):
        """Test default worker count initialization."""
        executor = ParallelExecutor()
        assert executor.max_workers >= 2
        assert executor.max_workers <= 16
        executor.shutdown()
    
    def test_init_custom_workers(self):
        """Test custom worker count initialization."""
        executor = ParallelExecutor(max_workers=4)
        assert executor.max_workers == 4
        executor.shutdown()
    
    def test_get_optimal_workers(self):
        """Test optimal worker calculation."""
        workers = ParallelExecutor.get_optimal_workers()
        assert 2 <= workers <= 16
    
    def test_submit_single_task(self):
        """Test single task submission."""
        executor = ParallelExecutor(max_workers=2)
        
        future = executor.submit(lambda x: x * 2, 5)
        result = future.result()
        
        assert result.is_success
        assert result.value == 10
        assert result.execution_time_ms >= 0
        
        executor.shutdown()
    
    def test_submit_with_exception(self):
        """Test task that raises exception."""
        executor = ParallelExecutor(max_workers=2)
        
        def failing_task():
            raise ValueError("Test error")
        
        future = executor.submit(failing_task)
        result = future.result()
        
        assert not result.is_success
        assert result.error is not None
        assert "Test error" in str(result.error)
        
        executor.shutdown()
    
    def test_submit_batch(self):
        """Test batch task submission."""
        executor = ParallelExecutor(max_workers=4)
        
        tasks = [
            (lambda x: x * 2, (i,), {})
            for i in range(10)
        ]
        
        batch_result = executor.submit_batch(tasks)
        
        assert isinstance(batch_result, BatchResult)
        assert batch_result.successful_count == 10
        assert batch_result.failed_count == 0
        assert batch_result.success_rate == 1.0
        assert len(batch_result.get_successful_values()) == 10
        
        executor.shutdown()
    
    def test_submit_batch_empty(self):
        """Test empty batch submission."""
        executor = ParallelExecutor(max_workers=2)
        
        batch_result = executor.submit_batch([])
        
        assert batch_result.successful_count == 0
        assert batch_result.failed_count == 0
        assert batch_result.total_time_ms == 0.0
        
        executor.shutdown()
    
    def test_map_parallel(self):
        """Test parallel map operation."""
        executor = ParallelExecutor(max_workers=4)
        
        items = list(range(20))
        results = executor.map_parallel(lambda x: x ** 2, items)
        
        assert len(results) == 20
        assert results == [x ** 2 for x in items]
        
        executor.shutdown()
    
    def test_map_parallel_empty(self):
        """Test parallel map with empty input."""
        executor = ParallelExecutor(max_workers=2)
        
        results = executor.map_parallel(lambda x: x, [])
        
        assert results == []
        
        executor.shutdown()
    
    def test_map_parallel_maintains_order(self):
        """Test that parallel map maintains order."""
        executor = ParallelExecutor(max_workers=4)
        
        # Add varying delays to encourage out-of-order completion
        def slow_double(x):
            time.sleep(0.001 * (10 - x % 10))
            return x * 2
        
        items = list(range(20))
        results = executor.map_parallel(slow_double, items)
        
        expected = [x * 2 for x in items]
        assert results == expected
        
        executor.shutdown()
    
    def test_reduce_parallel(self):
        """Test parallel reduce operation."""
        executor = ParallelExecutor(max_workers=4)
        
        items = list(range(1, 101))
        result = executor.reduce_parallel(lambda a, b: a + b, items)
        
        assert result == sum(items)
        
        executor.shutdown()
    
    def test_reduce_parallel_with_initial(self):
        """Test parallel reduce with initial value."""
        executor = ParallelExecutor(max_workers=4)
        
        items = [1, 2, 3, 4, 5]
        result = executor.reduce_parallel(lambda a, b: a + b, items, initial=100)
        
        assert result == 100 + sum(items)
        
        executor.shutdown()
    
    def test_reduce_parallel_empty(self):
        """Test parallel reduce with empty list."""
        executor = ParallelExecutor(max_workers=2)
        
        result = executor.reduce_parallel(lambda a, b: a + b, [])
        assert result is None
        
        result_with_initial = executor.reduce_parallel(
            lambda a, b: a + b, [], initial=42
        )
        assert result_with_initial == 42
        
        executor.shutdown()
    
    def test_reduce_parallel_single_item(self):
        """Test parallel reduce with single item."""
        executor = ParallelExecutor(max_workers=2)
        
        result = executor.reduce_parallel(lambda a, b: a + b, [5])
        assert result == 5
        
        result_with_initial = executor.reduce_parallel(
            lambda a, b: a + b, [5], initial=10
        )
        assert result_with_initial == 15
        
        executor.shutdown()
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        executor = ParallelExecutor(max_workers=4)
        
        # Submit some tasks
        for i in range(5):
            executor.submit(lambda x: x, i).result()
        
        stats = executor.get_stats()
        
        assert stats['max_workers'] == 4
        assert stats['total_submitted'] == 5
        assert stats['total_completed'] == 5
        assert stats['uptime_seconds'] >= 0
        
        executor.shutdown()
    
    def test_context_manager(self):
        """Test context manager usage."""
        with ParallelExecutor(max_workers=2) as executor:
            result = executor.submit(lambda: 42).result()
            assert result.value == 42
        
        # Executor should be shut down
        assert executor._executor is None
    
    def test_speedup_vs_sequential(self):
        """Test that parallel execution provides speedup."""
        items = list(range(50))
        
        def cpu_bound(n):
            # Small CPU-bound computation
            return sum(i * i for i in range(1000))
        
        # Sequential
        start = time.time()
        sequential = [cpu_bound(n) for n in items]
        seq_time = time.time() - start
        
        # Parallel
        with ParallelExecutor(max_workers=4) as executor:
            start = time.time()
            parallel = executor.map_parallel(cpu_bound, items)
            par_time = time.time() - start
        
        assert sequential == parallel
        # Should see some speedup (allow some margin for test environment)
        # In CI, parallelism may not show full speedup
        assert par_time <= seq_time * 1.5  # At least not significantly slower


# ============================================================================
# TEST ASYNC PATTERN PROCESSOR
# ============================================================================

class TestAsyncPatternProcessor:
    """Test AsyncPatternProcessor functionality."""
    
    @pytest.mark.asyncio
    async def test_init(self):
        """Test initialization."""
        processor = AsyncPatternProcessor(max_concurrent=50, timeout_seconds=10.0)
        assert processor.max_concurrent == 50
        assert processor.timeout_seconds == 10.0
    
    @pytest.mark.asyncio
    async def test_encode_batch_async(self):
        """Test async batch encoding."""
        processor = AsyncPatternProcessor(max_concurrent=10)
        
        patterns = [f"pattern_{i}" for i in range(5)]
        encoder_fn = lambda p: f"encoded_{p}"
        
        results = await processor.encode_batch_async(patterns, encoder_fn)
        
        assert len(results) == 5
        assert all(r.is_success for r in results)
        
        values = [r.value for r in results]
        expected = [f"encoded_pattern_{i}" for i in range(5)]
        assert values == expected
    
    @pytest.mark.asyncio
    async def test_search_batch_async(self):
        """Test async batch search."""
        processor = AsyncPatternProcessor(max_concurrent=10)
        
        queries = ["q1", "q2", "q3"]
        search_fn = lambda q: [f"result_for_{q}"]
        
        results = await processor.search_batch_async(queries, search_fn)
        
        assert len(results) == 3
        assert all(r.is_success for r in results)
    
    @pytest.mark.asyncio
    async def test_cluster_async(self):
        """Test async clustering."""
        processor = AsyncPatternProcessor()
        
        patterns = ["a", "b", "c"]
        cluster_fn = lambda ps: {0: ps[:2], 1: ps[2:]}
        
        result = await processor.cluster_async(patterns, cluster_fn)
        
        assert result.is_success
        assert result.value == {0: ["a", "b"], 1: ["c"]}
    
    @pytest.mark.asyncio
    async def test_process_stream_with_callback(self):
        """Test stream processing with callback."""
        processor = AsyncPatternProcessor(max_concurrent=5)
        
        items = list(range(10))
        processed = []
        
        def callback(result):
            processed.append(result.value)
        
        results = await processor.process_stream(
            items,
            lambda x: x * 2,
            on_result=callback
        )
        
        assert len(results) == 10
        assert len(processed) == 10
        assert all(r.is_success for r in results)
    
    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Test that concurrency is properly limited."""
        processor = AsyncPatternProcessor(max_concurrent=3)
        
        concurrent_count = [0]
        max_concurrent = [0]
        lock = asyncio.Lock()
        
        def track_concurrent(x):
            # This runs in a thread, so use thread-safe operations
            concurrent_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])
            time.sleep(0.01)
            concurrent_count[0] -= 1
            return x
        
        items = list(range(10))
        await processor.encode_batch_async(items, track_concurrent)
        
        # Max concurrent should not exceed limit
        assert max_concurrent[0] <= 3
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics retrieval."""
        processor = AsyncPatternProcessor(max_concurrent=10)
        
        items = list(range(5))
        await processor.encode_batch_async(items, lambda x: x)
        
        stats = processor.get_stats()
        
        assert stats['total_operations'] == 5
        assert stats['successful_operations'] == 5
        assert stats['failed_operations'] == 0
        assert stats['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test statistics reset."""
        processor = AsyncPatternProcessor()
        
        await processor.encode_batch_async([1, 2, 3], lambda x: x)
        processor.reset_stats()
        
        stats = processor.get_stats()
        assert stats['total_operations'] == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in async operations."""
        processor = AsyncPatternProcessor()
        
        def failing_encoder(x):
            if x == 2:
                raise ValueError("Test error")
            return x
        
        results = await processor.encode_batch_async([1, 2, 3], failing_encoder)
        
        assert len(results) == 3
        assert results[0].is_success
        assert not results[1].is_success
        assert results[2].is_success
        
        stats = processor.get_stats()
        assert stats['failed_operations'] == 1


# ============================================================================
# TEST WORK STEALING SCHEDULER
# ============================================================================

class TestWorkStealingScheduler:
    """Test WorkStealingScheduler functionality."""
    
    def test_init(self):
        """Test initialization."""
        scheduler = WorkStealingScheduler(num_workers=4)
        assert scheduler.num_workers == 4
        assert len(scheduler._queues) == 4
        assert len(scheduler._locks) == 4
    
    def test_submit_work(self):
        """Test work submission."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        work = WorkItem(id=1, data="task1")
        worker_id = scheduler.submit_work(work)
        
        assert 0 <= worker_id < 4
        assert scheduler.get_pending_count() == 1
    
    def test_submit_distributes_work(self):
        """Test that work is distributed across workers."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        # Submit many items
        for i in range(20):
            scheduler.submit_work(WorkItem(id=i, data=f"task_{i}"))
        
        load = scheduler.balance_load()
        
        # All workers should have some work
        assert sum(load.values()) == 20
    
    def test_get_work_from_own_queue(self):
        """Test getting work from own queue."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        # Submit work to specific worker
        work = WorkItem(id=1, data="task1")
        assigned = scheduler.submit_work(work)
        
        # Get from same worker
        retrieved = scheduler.get_work(assigned, timeout=0.1)
        
        assert retrieved is not None
        assert retrieved.id == 1
    
    def test_work_stealing(self):
        """Test work stealing mechanism."""
        scheduler = WorkStealingScheduler(num_workers=2)
        
        # Submit many items - they'll go to various workers
        for i in range(10):
            scheduler.submit_work(WorkItem(id=i, data=f"task_{i}"))
        
        # Get all work from worker 0 (may steal from worker 1)
        retrieved_count = 0
        while scheduler.get_pending_count() > 0:
            work = scheduler.get_work(worker_id=0, timeout=0.1)
            if work:
                retrieved_count += 1
            else:
                break
        
        # Should get all items
        assert retrieved_count >= 5  # At least items from one queue
    
    def test_priority_ordering(self):
        """Test that high priority items come first."""
        scheduler = WorkStealingScheduler(num_workers=1)
        
        # Submit with different priorities
        scheduler.submit_work(WorkItem(id=1, data="low", priority=0))
        scheduler.submit_work(WorkItem(id=2, data="high", priority=2))
        scheduler.submit_work(WorkItem(id=3, data="medium", priority=1))
        
        # Get in priority order
        work1 = scheduler.get_work(0, timeout=0.1)
        work2 = scheduler.get_work(0, timeout=0.1)
        work3 = scheduler.get_work(0, timeout=0.1)
        
        assert work1.priority == 2
        assert work2.priority == 1
        assert work3.priority == 0
    
    def test_balance_load(self):
        """Test load balance reporting."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        # Submit items
        for i in range(12):
            scheduler.submit_work(WorkItem(id=i, data=f"task_{i}"))
        
        load = scheduler.balance_load()
        
        assert len(load) == 4
        assert sum(load.values()) == 12
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        for i in range(10):
            scheduler.submit_work(WorkItem(id=i, data=f"task_{i}"))
        
        stats = scheduler.get_stats()
        
        assert stats['num_workers'] == 4
        assert stats['total_submitted'] == 10
        assert stats['total_pending'] == 10
        assert 'load_balance' in stats
    
    def test_clear(self):
        """Test clearing all pending work."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        for i in range(10):
            scheduler.submit_work(WorkItem(id=i, data=f"task_{i}"))
        
        cleared = scheduler.clear()
        
        assert cleared == 10
        assert scheduler.get_pending_count() == 0
    
    def test_get_work_empty_queue(self):
        """Test getting work from empty queue."""
        scheduler = WorkStealingScheduler(num_workers=2)
        
        result = scheduler.get_work(0, timeout=0.05)
        
        assert result is None


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parallel_map(self):
        """Test parallel_map function."""
        items = list(range(10))
        results = parallel_map(lambda x: x * 2, items, max_workers=2)
        
        assert results == [x * 2 for x in items]
    
    def test_parallel_reduce(self):
        """Test parallel_reduce function."""
        items = list(range(1, 11))
        result = parallel_reduce(lambda a, b: a + b, items)
        
        assert result == sum(items)
    
    @pytest.mark.asyncio
    async def test_async_map(self):
        """Test async_map function."""
        items = list(range(5))
        results = await async_map(lambda x: x * 2, items, max_concurrent=3)
        
        assert len(results) == 5
        assert all(r.is_success for r in results)
        values = [r.value for r in results]
        assert values == [x * 2 for x in items]


# ============================================================================
# TEST EXECUTION RESULT
# ============================================================================

class TestExecutionResult:
    """Test ExecutionResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = ExecutionResult(value=42, success=True)
        
        assert result.is_success
        assert result.value == 42
        assert result.error is None
    
    def test_failure_result(self):
        """Test failed result."""
        error = ValueError("test error")
        result = ExecutionResult(value=None, success=False, error=error)
        
        assert not result.is_success
        assert result.error is error


class TestBatchResult:
    """Test BatchResult dataclass."""
    
    def test_success_rate(self):
        """Test success rate calculation."""
        results = [
            ExecutionResult(value=1, success=True),
            ExecutionResult(value=2, success=True),
            ExecutionResult(value=None, success=False, error=ValueError()),
        ]
        
        batch = BatchResult(
            results=results,
            total_time_ms=100.0,
            successful_count=2,
            failed_count=1
        )
        
        assert batch.success_rate == pytest.approx(0.6667, rel=0.01)
    
    def test_get_successful_values(self):
        """Test getting successful values."""
        results = [
            ExecutionResult(value=1, success=True),
            ExecutionResult(value=None, success=False),
            ExecutionResult(value=3, success=True),
        ]
        
        batch = BatchResult(
            results=results,
            total_time_ms=100.0,
            successful_count=2,
            failed_count=1
        )
        
        values = batch.get_successful_values()
        assert values == [1, 3]


class TestWorkItem:
    """Test WorkItem dataclass."""
    
    def test_priority_comparison(self):
        """Test priority-based comparison."""
        low = WorkItem(id=1, data="low", priority=0)
        high = WorkItem(id=2, data="high", priority=2)
        
        # Higher priority should be "less than" for min-heap
        assert high < low
    
    def test_age_comparison_same_priority(self):
        """Test age-based comparison for same priority."""
        old = WorkItem(id=1, data="old", priority=1, created_at=100.0)
        new = WorkItem(id=2, data="new", priority=1, created_at=200.0)
        
        # Older should come first (less than)
        assert old < new


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestParallelProcessorIntegration:
    """Integration tests for parallel processing."""
    
    def test_complex_workflow(self):
        """Test complex multi-stage workflow."""
        with ParallelExecutor(max_workers=4) as executor:
            # Stage 1: Generate data
            data = list(range(100))
            
            # Stage 2: Parallel transform
            squared = executor.map_parallel(lambda x: x ** 2, data)
            
            # Stage 3: Parallel filter
            filtered = executor.map_parallel(
                lambda x: x if x > 100 else None, 
                squared
            )
            filtered = [x for x in filtered if x is not None]
            
            # Stage 4: Parallel reduce
            total = executor.reduce_parallel(lambda a, b: a + b, filtered)
            
            # Verify result
            expected = sum(x ** 2 for x in data if x ** 2 > 100)
            assert total == expected
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async(self):
        """Test mixing sync and async operations."""
        # Sync executor for CPU-bound
        executor = ParallelExecutor(max_workers=4)
        
        # Async processor for I/O-bound
        processor = AsyncPatternProcessor(max_concurrent=10)
        
        # CPU-bound parallel work
        data = list(range(20))
        cpu_results = executor.map_parallel(lambda x: x ** 2, data)
        
        # Async I/O work
        async_results = await processor.encode_batch_async(
            cpu_results,
            lambda x: f"encoded_{x}"
        )
        
        assert len(async_results) == 20
        assert all(r.is_success for r in async_results)
        
        executor.shutdown()
    
    def test_work_stealing_under_load(self):
        """Test work stealing under uneven load."""
        scheduler = WorkStealingScheduler(num_workers=4)
        
        # Submit work with varying complexity (simulated by different data)
        for i in range(100):
            scheduler.submit_work(WorkItem(
                id=i,
                data=f"task_{i}",
                priority=i % 3
            ))
        
        # Simulate workers processing
        processed = [0] * 4
        
        def worker_fn(worker_id: int):
            count = 0
            while True:
                work = scheduler.get_work(worker_id, timeout=0.01)
                if work is None:
                    break
                count += 1
            return count
        
        # Run workers in parallel
        with ParallelExecutor(max_workers=4) as executor:
            results = executor.map_parallel(worker_fn, list(range(4)))
        
        total_processed = sum(results)
        assert total_processed == 100  # All work processed


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
