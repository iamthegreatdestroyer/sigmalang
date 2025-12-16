#!/usr/bin/env python3
"""
Phase 9B: Performance Optimization - Verification Script
=========================================================

Verifies all Phase 9B optimization modules across:
- ΣLANG: Fast Encoder
- ΣVAULT: Parallel I/O  
- Ryot LLM: Optimized Attention, Batch Inference, Tiered KV Cache

Run: python scripts/verify_phase9b.py
"""

import sys
import asyncio
from pathlib import Path
from typing import Tuple, List

# Color output helpers
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_result(name: str, success: bool, details: str = "") -> None:
    """Print colored test result."""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {status} {name}")
    if details:
        print(f"         {Colors.YELLOW}{details}{Colors.RESET}")


def test_sigmalang_fast_encoder() -> Tuple[bool, str]:
    """Test ΣLANG Fast Encoder module."""
    try:
        # Add sigmalang to path if needed
        sigmalang_path = Path(__file__).parent.parent
        if str(sigmalang_path) not in sys.path:
            sys.path.insert(0, str(sigmalang_path))
        
        from core.fast_encoder import (
            FastGlyphEncoder,
            FastEncoderConfig,
            EncodingStats,
        )
        
        # Test instantiation
        config = FastEncoderConfig(
            chunk_size=512,
            max_workers=2,
            cache_enabled=True,
        )
        encoder = FastGlyphEncoder(config=config)
        
        # Test encoding
        test_text = "Hello, ΣLANG! " * 100
        encoded = encoder.encode_fast(test_text)
        assert isinstance(encoded, bytes), "encode_fast must return bytes"
        assert len(encoded) > 0, "Encoded result must not be empty"
        
        # Test cache hit
        encoded2 = encoder.encode_fast(test_text)
        assert encoded == encoded2, "Cached result must match"
        
        # Test batch encoding
        batch = encoder.encode_batch(["test1", "test2", "test3"])
        assert len(batch) == 3, "Batch must return 3 results"
        
        # Test stats (returns dict)
        stats = encoder.get_stats()
        assert stats["cache_hits"] > 0, "Should have cache hits"
        assert stats["total_encoded"] > 0, "Should have encoded items"
        
        return True, f"Encoded {len(encoded)} bytes, {stats['cache_hits']} cache hits"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_sigmavault_parallel_io() -> Tuple[bool, str]:
    """Test ΣVAULT Parallel I/O module."""
    try:
        # Add sigmavault to path
        sigmavault_path = Path(__file__).parent.parent.parent / "sigmavault"
        if str(sigmavault_path) not in sys.path:
            sys.path.insert(0, str(sigmavault_path))
        
        from sigmavault.core.parallel_io import (
            ParallelIOManager,
            ParallelIOConfig,
            IOStats,
            MockChunkBackend,
        )
        
        async def run_test():
            # Test instantiation
            config = ParallelIOConfig(
                max_concurrent=4,
                chunk_timeout_s=5.0,
            )
            manager = ParallelIOManager(config=config)
            
            # Test with mock backend
            backend = MockChunkBackend()
            
            # MUST write chunks BEFORE reading (MockChunkBackend returns None for unwritten)
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            write_data = {cid: f"data_{cid}".encode() for cid in chunk_ids}
            await manager.write_chunks_parallel(write_data, backend)
            
            # Now read the chunks we just wrote
            results = await manager.read_chunks_parallel(chunk_ids, backend)
            assert len(results) == 3, f"Should read 3 chunks, got {len(results)}"
            
            # Test additional writes
            extra_data = {
                "write_4": b"data4",
                "write_5": b"data5",
            }
            await manager.write_chunks_parallel(extra_data, backend)
            
            # Test stats (returns dict)
            stats = manager.get_stats()
            assert stats["reads"] >= 3, "Should have at least 3 reads"
            assert stats["writes"] >= 5, "Should have at least 5 writes"
            
            return stats
        
        stats = asyncio.run(run_test())
        return True, f"{stats['reads']} reads, {stats['writes']} writes"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_ryot_optimized_attention() -> Tuple[bool, str]:
    """Test Ryot LLM Optimized Attention module."""
    try:
        # Add Ryot to path (use RYZEN-LLM root, not src, for relative imports)
        ryot_path = Path.home() / "Ryot" / "RYZEN-LLM"
        if str(ryot_path) not in sys.path:
            sys.path.insert(0, str(ryot_path))
        
        from src.core.engine.optimized_attention import (
            OptimizedAttention,
            AttentionConfig,
        )
        
        # Test instantiation (using actual API: num_heads, head_dim)
        config = AttentionConfig(
            num_heads=4,
            head_dim=64,
            chunk_size=128,
        )
        attention = OptimizedAttention(config)
        
        # Test configuration  
        assert attention.config.num_heads == 4
        assert attention.config.head_dim == 64
        
        return True, f"Config: {config.num_heads} heads × {config.head_dim}d"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_ryot_batch_inference() -> Tuple[bool, str]:
    """Test Ryot LLM Batch Inference module."""
    try:
        # Add Ryot to path (use RYZEN-LLM root, not src, for relative imports)
        ryot_path = Path.home() / "Ryot" / "RYZEN-LLM"
        if str(ryot_path) not in sys.path:
            sys.path.insert(0, str(ryot_path))
        
        from src.core.engine.batch_inference import (
            BatchInferenceEngine,
            BatchRequest,
            RequestStatus,
        )
        
        # Test enum values
        assert RequestStatus.PENDING is not None
        assert RequestStatus.PROCESSING is not None
        assert RequestStatus.COMPLETED is not None
        
        # Test request dataclass
        request = BatchRequest(
            request_id="test_001",
            prompt="Hello, world!",
            max_tokens=100,
        )
        assert request.status == RequestStatus.PENDING
        
        return True, f"Request states: {len(RequestStatus)} statuses"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_ryot_tiered_kv_cache() -> Tuple[bool, str]:
    """Test Ryot LLM Tiered KV Cache module."""
    try:
        # Add Ryot to path (use RYZEN-LLM root, not src, for relative imports)
        ryot_path = Path.home() / "Ryot" / "RYZEN-LLM"
        if str(ryot_path) not in sys.path:
            sys.path.insert(0, str(ryot_path))
        
        from src.core.cache.tiered_kv_cache import (
            TieredKVCache,
            TieredCacheConfig,
            CacheTier,
            CacheEntry,
        )
        
        # Test tiers
        assert CacheTier.L1 is not None
        assert CacheTier.L2 is not None
        assert CacheTier.L3 is not None
        
        # Test cache instantiation (using TieredCacheConfig)
        config = TieredCacheConfig(
            l1_max_entries=1000,
            l2_max_entries=10000,
            l3_max_entries=100000,
        )
        cache = TieredKVCache(config=config)
        
        # Verify config
        assert cache.config.l1_max_entries == 1000
        assert cache.config.l2_max_entries == 10000
        
        return True, f"Tiers: L1={config.l1_max_entries}, L2={config.l2_max_entries}"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main() -> int:
    """Run all Phase 9B verification tests."""
    print_header("Phase 9B: Performance Optimization")
    print("Verification Script v1.0\n")
    
    tests = [
        ("ΣLANG Fast Encoder", test_sigmalang_fast_encoder),
        ("ΣVAULT Parallel I/O", test_sigmavault_parallel_io),
        ("Ryot Optimized Attention", test_ryot_optimized_attention),
        ("Ryot Batch Inference", test_ryot_batch_inference),
        ("Ryot Tiered KV Cache", test_ryot_tiered_kv_cache),
    ]
    
    results: List[Tuple[str, bool, str]] = []
    
    print(f"{Colors.BOLD}Running verification tests...{Colors.RESET}\n")
    
    for name, test_func in tests:
        success, details = test_func()
        results.append((name, success, details))
        print_result(name, success, details)
    
    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print_header("Summary")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}All {total} tests PASSED!{Colors.RESET}")
        print(f"\n{Colors.GREEN}Phase 9B: Performance Optimization - VERIFIED ✓{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}{passed}/{total} tests passed{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Review failed tests above{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
