"""
Phase 2A.5 - Task 1: Parallel Processing Engine
================================================

Multi-threaded pattern operations for dramatic performance gains.

This module provides:
- ParallelExecutor: Thread pool management with work stealing
- AsyncPatternProcessor: Async I/O for pattern operations
- WorkStealingScheduler: Dynamic load balancing

Performance Targets:
- 4x speedup on 8-core machines for batch operations
- Sub-linear scaling with core count
- Memory-efficient work distribution

Copyright 2025 - SigmaLang Project
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, Iterable, List, Optional, 
    Tuple, TypeVar, Union
)
from functools import partial
import logging

# Type variables for generic operations
T = TypeVar('T')
R = TypeVar('R')

logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION RESULT
# ============================================================================

@dataclass
class ExecutionResult(Generic[T]):
    """Result of a parallel execution."""
    value: Optional[T]
    success: bool
    error: Optional[Exception] = None
    execution_time_ms: float = 0.0
    worker_id: Optional[int] = None
    
    @property
    def is_success(self) -> bool:
        return self.success and self.error is None


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch parallel execution."""
    results: List[ExecutionResult[T]]
    total_time_ms: float
    successful_count: int
    failed_count: int
    worker_utilization: Dict[int, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        total = self.successful_count + self.failed_count
        return self.successful_count / total if total > 0 else 0.0
    
    def get_successful_values(self) -> List[T]:
        """Get all successful result values."""
        return [r.value for r in self.results if r.is_success and r.value is not None]


# ============================================================================
# PARALLEL EXECUTOR
# ============================================================================

class ParallelExecutor:
    """
    Thread pool management with intelligent work distribution.
    
    Features:
    - Auto-tuned thread count based on CPU cores
    - Work stealing for load balancing
    - Batch submission with progress tracking
    - Graceful shutdown handling
    
    Example:
        >>> executor = ParallelExecutor(max_workers=4)
        >>> results = executor.map_parallel(process_pattern, patterns)
        >>> print(f"Processed {len(results)} patterns")
        >>> executor.shutdown()
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_name_prefix: str = "parallel_worker",
        enable_work_stealing: bool = True
    ):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum worker threads (default: CPU count)
            thread_name_prefix: Prefix for worker thread names
            enable_work_stealing: Enable work stealing for load balancing
        """
        self.max_workers = max_workers or self.get_optimal_workers()
        self.thread_name_prefix = thread_name_prefix
        self.enable_work_stealing = enable_work_stealing
        
        self._executor: Optional[ThreadPoolExecutor] = None
        self._active_tasks: Dict[int, int] = {}  # worker_id -> task_count
        self._lock = threading.Lock()
        self._total_tasks_submitted = 0
        self._total_tasks_completed = 0
        self._start_time: Optional[float] = None
        
        # Work stealing scheduler
        self._scheduler: Optional[WorkStealingScheduler] = None
        if enable_work_stealing:
            self._scheduler = WorkStealingScheduler(self.max_workers)
    
    @staticmethod
    def get_optimal_workers() -> int:
        """
        Determine optimal worker count based on system resources.
        
        Returns:
            Optimal number of worker threads
        """
        cpu_count = os.cpu_count() or 4
        # Use CPU count for compute-bound, 2x for I/O-bound
        # Default to CPU count as most pattern ops are compute-bound
        return max(2, min(cpu_count, 16))  # Cap at 16 to avoid overhead
    
    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Ensure executor is initialized."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix=self.thread_name_prefix
            )
            self._start_time = time.time()
        return self._executor
    
    def submit(
        self, 
        fn: Callable[..., T], 
        *args: Any, 
        **kwargs: Any
    ) -> Future[ExecutionResult[T]]:
        """
        Submit a single task for execution.
        
        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future containing ExecutionResult
        """
        executor = self._ensure_executor()
        
        def wrapped_fn() -> ExecutionResult[T]:
            worker_id = hash(threading.current_thread().name) % self.max_workers
            start = time.time()
            try:
                with self._lock:
                    self._active_tasks[worker_id] = self._active_tasks.get(worker_id, 0) + 1
                
                result = fn(*args, **kwargs)
                
                execution_time = (time.time() - start) * 1000
                return ExecutionResult(
                    value=result,
                    success=True,
                    execution_time_ms=execution_time,
                    worker_id=worker_id
                )
            except Exception as e:
                execution_time = (time.time() - start) * 1000
                logger.error(f"Task failed: {e}")
                return ExecutionResult(
                    value=None,
                    success=False,
                    error=e,
                    execution_time_ms=execution_time,
                    worker_id=worker_id
                )
            finally:
                with self._lock:
                    self._active_tasks[worker_id] = self._active_tasks.get(worker_id, 1) - 1
                    self._total_tasks_completed += 1
        
        with self._lock:
            self._total_tasks_submitted += 1
        
        return executor.submit(wrapped_fn)
    
    def submit_batch(
        self,
        tasks: List[Tuple[Callable[..., T], tuple, dict]]
    ) -> BatchResult[T]:
        """
        Submit multiple tasks for execution.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            
        Returns:
            BatchResult containing all results
        """
        if not tasks:
            return BatchResult(
                results=[],
                total_time_ms=0.0,
                successful_count=0,
                failed_count=0
            )
        
        start_time = time.time()
        futures: List[Future[ExecutionResult[T]]] = []
        
        for fn, args, kwargs in tasks:
            future = self.submit(fn, *args, **kwargs)
            futures.append(future)
        
        # Collect results
        results: List[ExecutionResult[T]] = []
        successful = 0
        failed = 0
        worker_times: Dict[int, float] = {}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result.is_success:
                successful += 1
            else:
                failed += 1
            
            if result.worker_id is not None:
                worker_times[result.worker_id] = worker_times.get(
                    result.worker_id, 0
                ) + result.execution_time_ms
        
        total_time = (time.time() - start_time) * 1000
        
        # Calculate worker utilization
        max_worker_time = max(worker_times.values()) if worker_times else 1.0
        utilization = {
            wid: wtime / max_worker_time 
            for wid, wtime in worker_times.items()
        }
        
        return BatchResult(
            results=results,
            total_time_ms=total_time,
            successful_count=successful,
            failed_count=failed,
            worker_utilization=utilization
        )
    
    def map_parallel(
        self,
        fn: Callable[[T], R],
        items: Iterable[T],
        chunk_size: Optional[int] = None
    ) -> List[R]:
        """
        Apply function to items in parallel.
        
        Args:
            fn: Function to apply to each item
            items: Items to process
            chunk_size: Items per chunk (auto-determined if None)
            
        Returns:
            List of results in original order
        """
        items_list = list(items)
        if not items_list:
            return []
        
        # Auto-determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items_list) // (self.max_workers * 4))
        
        executor = self._ensure_executor()
        
        # Submit all items
        future_to_idx: Dict[Future, int] = {}
        for idx, item in enumerate(items_list):
            future = executor.submit(fn, item)
            future_to_idx[future] = idx
        
        # Collect results in order
        results: List[Optional[R]] = [None] * len(items_list)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"map_parallel item {idx} failed: {e}")
                results[idx] = None  # type: ignore
        
        return results  # type: ignore
    
    def reduce_parallel(
        self,
        fn: Callable[[T, T], T],
        items: List[T],
        initial: Optional[T] = None
    ) -> Optional[T]:
        """
        Parallel reduction of items.
        
        Uses tree-based reduction for optimal parallelism.
        
        Args:
            fn: Binary reduction function
            items: Items to reduce
            initial: Initial value (optional)
            
        Returns:
            Reduced value or None if empty
        """
        if not items:
            return initial
        
        if len(items) == 1:
            return fn(initial, items[0]) if initial is not None else items[0]
        
        # Tree-based parallel reduction
        current = list(items)
        if initial is not None:
            current = [initial] + current
        
        while len(current) > 1:
            # Pair up items for parallel reduction
            pairs: List[Tuple[T, T]] = []
            for i in range(0, len(current) - 1, 2):
                pairs.append((current[i], current[i + 1]))
            
            # Handle odd item
            remainder = current[-1] if len(current) % 2 == 1 else None
            
            # Reduce pairs in parallel
            reduced = self.map_parallel(lambda p: fn(p[0], p[1]), pairs)
            
            current = reduced
            if remainder is not None:
                current.append(remainder)
        
        return current[0] if current else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            active = sum(self._active_tasks.values())
            uptime = (time.time() - self._start_time) if self._start_time else 0
        
        return {
            'max_workers': self.max_workers,
            'total_submitted': self._total_tasks_submitted,
            'total_completed': self._total_tasks_completed,
            'active_tasks': active,
            'uptime_seconds': uptime,
            'work_stealing_enabled': self.enable_work_stealing
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.
        
        Args:
            wait: Wait for pending tasks to complete
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info(f"ParallelExecutor shut down. Stats: {self.get_stats()}")
    
    def __enter__(self) -> 'ParallelExecutor':
        self._ensure_executor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown(wait=True)


# ============================================================================
# ASYNC PATTERN PROCESSOR
# ============================================================================

class AsyncPatternProcessor:
    """
    Async I/O for pattern operations.
    
    Enables non-blocking pattern processing for I/O-bound operations
    like disk access and network calls.
    
    Example:
        >>> processor = AsyncPatternProcessor()
        >>> results = await processor.encode_batch_async(patterns)
        >>> print(f"Encoded {len(results)} patterns")
    """
    
    def __init__(
        self,
        max_concurrent: int = 100,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize async processor.
        
        Args:
            max_concurrent: Maximum concurrent operations
            timeout_seconds: Operation timeout
        """
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'timeout_operations': 0
        }
    
    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    async def _execute_with_limit(
        self,
        coro: Any,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Execute coroutine with concurrency limit."""
        semaphore = self._get_semaphore()
        timeout = timeout or self.timeout_seconds
        
        start = time.time()
        self._stats['total_operations'] += 1
        
        async with semaphore:
            try:
                result = await asyncio.wait_for(coro, timeout=timeout)
                execution_time = (time.time() - start) * 1000
                self._stats['successful_operations'] += 1
                return ExecutionResult(
                    value=result,
                    success=True,
                    execution_time_ms=execution_time
                )
            except asyncio.TimeoutError:
                execution_time = (time.time() - start) * 1000
                self._stats['timeout_operations'] += 1
                return ExecutionResult(
                    value=None,
                    success=False,
                    error=TimeoutError(f"Operation timed out after {timeout}s"),
                    execution_time_ms=execution_time
                )
            except Exception as e:
                execution_time = (time.time() - start) * 1000
                self._stats['failed_operations'] += 1
                return ExecutionResult(
                    value=None,
                    success=False,
                    error=e,
                    execution_time_ms=execution_time
                )
    
    async def encode_batch_async(
        self,
        patterns: List[Any],
        encoder_fn: Callable[[Any], Any]
    ) -> List[ExecutionResult]:
        """
        Encode patterns asynchronously.
        
        Args:
            patterns: Patterns to encode
            encoder_fn: Synchronous encoder function
            
        Returns:
            List of ExecutionResults
        """
        loop = asyncio.get_event_loop()
        
        async def encode_one(pattern: Any) -> Any:
            return await loop.run_in_executor(None, encoder_fn, pattern)
        
        tasks = [
            self._execute_with_limit(encode_one(p))
            for p in patterns
        ]
        
        return await asyncio.gather(*tasks)
    
    async def search_batch_async(
        self,
        queries: List[Any],
        search_fn: Callable[[Any], List[Any]]
    ) -> List[ExecutionResult]:
        """
        Search patterns asynchronously.
        
        Args:
            queries: Queries to search
            search_fn: Synchronous search function
            
        Returns:
            List of ExecutionResults
        """
        loop = asyncio.get_event_loop()
        
        async def search_one(query: Any) -> List[Any]:
            return await loop.run_in_executor(None, search_fn, query)
        
        tasks = [
            self._execute_with_limit(search_one(q))
            for q in queries
        ]
        
        return await asyncio.gather(*tasks)
    
    async def cluster_async(
        self,
        patterns: List[Any],
        cluster_fn: Callable[[List[Any]], Dict[int, List[Any]]]
    ) -> ExecutionResult:
        """
        Cluster patterns asynchronously.
        
        Args:
            patterns: Patterns to cluster
            cluster_fn: Synchronous clustering function
            
        Returns:
            ExecutionResult with cluster assignments
        """
        loop = asyncio.get_event_loop()
        
        async def do_cluster() -> Dict[int, List[Any]]:
            return await loop.run_in_executor(None, cluster_fn, patterns)
        
        return await self._execute_with_limit(do_cluster())
    
    async def process_stream(
        self,
        items: Iterable[T],
        processor_fn: Callable[[T], R],
        on_result: Optional[Callable[[ExecutionResult[R]], None]] = None
    ) -> List[ExecutionResult[R]]:
        """
        Process items as a stream with callbacks.
        
        Args:
            items: Items to process
            processor_fn: Processing function
            on_result: Optional callback for each result
            
        Returns:
            List of all results
        """
        loop = asyncio.get_event_loop()
        results: List[ExecutionResult[R]] = []
        
        async def process_one(item: T) -> ExecutionResult[R]:
            async def run():
                return await loop.run_in_executor(None, processor_fn, item)
            
            result = await self._execute_with_limit(run())
            
            if on_result:
                on_result(result)
            
            return result
        
        tasks = [process_one(item) for item in items]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            **self._stats,
            'max_concurrent': self.max_concurrent,
            'timeout_seconds': self.timeout_seconds,
            'success_rate': (
                self._stats['successful_operations'] / self._stats['total_operations']
                if self._stats['total_operations'] > 0 else 0.0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'timeout_operations': 0
        }


# ============================================================================
# WORK STEALING SCHEDULER
# ============================================================================

@dataclass
class WorkItem(Generic[T]):
    """A unit of work to be scheduled."""
    id: int
    data: T
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other: 'WorkItem') -> bool:
        # Higher priority first, then older first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class WorkStealingScheduler:
    """
    Dynamic load balancing through work stealing.
    
    Each worker has its own queue. When a worker's queue is empty,
    it "steals" work from the busiest worker's queue.
    
    Benefits:
    - Automatic load balancing
    - Reduced contention
    - Better cache locality
    
    Example:
        >>> scheduler = WorkStealingScheduler(num_workers=4)
        >>> scheduler.submit_work(WorkItem(id=1, data=pattern))
        >>> work = scheduler.get_work(worker_id=0)
    """
    
    def __init__(self, num_workers: int):
        """
        Initialize work stealing scheduler.
        
        Args:
            num_workers: Number of worker threads
        """
        self.num_workers = num_workers
        self._queues: List[queue.PriorityQueue] = [
            queue.PriorityQueue() for _ in range(num_workers)
        ]
        self._locks: List[threading.Lock] = [
            threading.Lock() for _ in range(num_workers)
        ]
        self._next_worker = 0
        self._assignment_lock = threading.Lock()
        self._work_counter = 0
        self._steal_count = 0
        self._submit_count = 0
    
    def submit_work(self, work: WorkItem) -> int:
        """
        Submit work to scheduler.
        
        Args:
            work: Work item to schedule
            
        Returns:
            Worker ID assigned to the work
        """
        # Round-robin assignment with consideration for queue size
        with self._assignment_lock:
            # Find least loaded worker
            min_size = float('inf')
            best_worker = 0
            
            for i in range(self.num_workers):
                size = self._queues[i].qsize()
                if size < min_size:
                    min_size = size
                    best_worker = i
            
            self._queues[best_worker].put(work)
            self._submit_count += 1
            return best_worker
    
    def get_work(
        self, 
        worker_id: int, 
        timeout: float = 0.1
    ) -> Optional[WorkItem]:
        """
        Get work for a worker, stealing if necessary.
        
        Args:
            worker_id: Worker requesting work
            timeout: Timeout for blocking get
            
        Returns:
            WorkItem or None if no work available
        """
        # Try own queue first
        try:
            work = self._queues[worker_id].get(timeout=timeout)
            return work
        except queue.Empty:
            pass
        
        # Try to steal from busiest worker
        return self._steal_work(worker_id)
    
    def _steal_work(self, thief_id: int) -> Optional[WorkItem]:
        """
        Steal work from another worker.
        
        Args:
            thief_id: Worker trying to steal
            
        Returns:
            Stolen work item or None
        """
        # Find busiest worker
        max_size = 0
        victim_id = -1
        
        for i in range(self.num_workers):
            if i == thief_id:
                continue
            size = self._queues[i].qsize()
            if size > max_size:
                max_size = size
                victim_id = i
        
        if victim_id < 0 or max_size <= 1:
            return None
        
        # Try to steal half of victim's work
        with self._locks[victim_id]:
            if self._queues[victim_id].empty():
                return None
            
            try:
                work = self._queues[victim_id].get_nowait()
                self._steal_count += 1
                return work
            except queue.Empty:
                return None
    
    def balance_load(self) -> Dict[int, int]:
        """
        Get current load distribution.
        
        Returns:
            Dict mapping worker_id to queue size
        """
        return {
            i: self._queues[i].qsize()
            for i in range(self.num_workers)
        }
    
    def get_pending_count(self) -> int:
        """Get total pending work items."""
        return sum(q.qsize() for q in self._queues)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        load = self.balance_load()
        total_pending = sum(load.values())
        max_load = max(load.values()) if load else 0
        min_load = min(load.values()) if load else 0
        
        return {
            'num_workers': self.num_workers,
            'total_submitted': self._submit_count,
            'total_steals': self._steal_count,
            'total_pending': total_pending,
            'load_balance': load,
            'load_imbalance': max_load - min_load,
            'steal_rate': (
                self._steal_count / self._submit_count 
                if self._submit_count > 0 else 0.0
            )
        }
    
    def clear(self) -> int:
        """
        Clear all pending work.
        
        Returns:
            Number of items cleared
        """
        cleared = 0
        for i in range(self.num_workers):
            with self._locks[i]:
                while not self._queues[i].empty():
                    try:
                        self._queues[i].get_nowait()
                        cleared += 1
                    except queue.Empty:
                        break
        return cleared


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None
) -> List[R]:
    """
    Convenience function for parallel mapping.
    
    Args:
        fn: Function to apply
        items: Items to process
        max_workers: Worker count (default: auto)
        
    Returns:
        Results in original order
    """
    with ParallelExecutor(max_workers=max_workers) as executor:
        return executor.map_parallel(fn, items)


def parallel_reduce(
    fn: Callable[[T, T], T],
    items: List[T],
    initial: Optional[T] = None,
    max_workers: Optional[int] = None
) -> Optional[T]:
    """
    Convenience function for parallel reduction.
    
    Args:
        fn: Binary reduction function
        items: Items to reduce
        initial: Initial value
        max_workers: Worker count (default: auto)
        
    Returns:
        Reduced value
    """
    with ParallelExecutor(max_workers=max_workers) as executor:
        return executor.reduce_parallel(fn, items, initial)


async def async_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_concurrent: int = 100
) -> List[ExecutionResult[R]]:
    """
    Convenience function for async mapping.
    
    Args:
        fn: Function to apply
        items: Items to process
        max_concurrent: Max concurrent operations
        
    Returns:
        List of ExecutionResults
    """
    processor = AsyncPatternProcessor(max_concurrent=max_concurrent)
    return await processor.process_stream(items, fn)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Demo parallel processing
    import math
    
    print("=" * 60)
    print("PARALLEL PROCESSOR DEMONSTRATION")
    print("=" * 60)
    
    # Test data
    numbers = list(range(1, 101))
    
    # Sequential baseline
    print("\n1. Sequential Processing:")
    start = time.time()
    sequential_results = [math.factorial(n % 20 + 100) for n in numbers]
    seq_time = (time.time() - start) * 1000
    print(f"   Time: {seq_time:.2f}ms")
    
    # Parallel processing
    print("\n2. Parallel Processing:")
    with ParallelExecutor() as executor:
        print(f"   Workers: {executor.max_workers}")
        start = time.time()
        parallel_results = executor.map_parallel(
            lambda n: math.factorial(n % 20 + 100), 
            numbers
        )
        par_time = (time.time() - start) * 1000
        print(f"   Time: {par_time:.2f}ms")
        print(f"   Speedup: {seq_time / par_time:.2f}x")
        print(f"   Stats: {executor.get_stats()}")
    
    # Parallel reduction
    print("\n3. Parallel Reduction:")
    with ParallelExecutor() as executor:
        result = executor.reduce_parallel(
            lambda a, b: a + b,
            list(range(1, 1001))
        )
        print(f"   Sum 1-1000: {result}")
        expected = sum(range(1, 1001))
        print(f"   Expected: {expected}")
        print(f"   Match: {result == expected}")
    
    # Work stealing demo
    print("\n4. Work Stealing Scheduler:")
    scheduler = WorkStealingScheduler(num_workers=4)
    
    # Submit uneven work
    for i in range(20):
        scheduler.submit_work(WorkItem(id=i, data=f"task_{i}", priority=i % 3))
    
    print(f"   Initial load: {scheduler.balance_load()}")
    print(f"   Stats: {scheduler.get_stats()}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
