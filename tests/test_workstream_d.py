#!/usr/bin/env python
"""
Workstream D Adaptive Compression - Quick Validation Test
Validates pattern detection, entropy analysis, and strategy selection
"""

import sys
import time
sys.path.insert(0, '.')

# Test adaptive compression selector
from sigmalang.core.adaptive_compression import (
    AdaptiveCompressionSelector,
    analyze_data_patterns
)

def main():
    print("=" * 80)
    print("ADAPTIVE COMPRESSION SELECTOR VALIDATION")
    print("=" * 80)
    
    # Test data
    test_cases = {
        "repetitive": b"AAAAAABBBBBBCCCCCCDDDDDD" * 100,
        "random": bytes((i * 7 + j * 13) % 256 for i in range(256) for j in range(16)),
        "text": b"The quick brown fox " * 50,
        "small": b"hi",
    }
    
    selector = AdaptiveCompressionSelector(enable_tracking=True)
    
    print("\nTesting strategy selection...")
    for name, data in test_cases.items():
        start = time.time()
        decision = selector.select(data)
        elapsed = (time.time() - start) * 1000
        
        print(f"\n{name:15} ({len(data):6,} bytes)")
        print(f"  Strategy:       {decision.strategy.name}")
        print(f"  Data type:      {decision.characteristics.data_type}")
        print(f"  Confidence:     {decision.confidence:.0%}")
        print(f"  Entropy:        {decision.characteristics.entropy:.2f}/8.0")
        print(f"  Detection time: {decision.decision_time_ms:.3f} ms")
        print(f"  Total time:     {elapsed:.3f} ms")
    
    # Test analysis function
    print("\n" + "=" * 80)
    print("DATA PATTERN ANALYSIS")
    print("=" * 80)
    
    data = b"PATTERN" * 1000
    analysis = analyze_data_patterns(data)
    
    print(f"\nData: b'PATTERN' * 1000 ({len(data)} bytes)")
    print(f"  Data type:      {analysis['data_type']}")
    print(f"  Entropy:        {analysis['entropy']:.3f}/8.0")
    print(f"  Repetition:     {analysis['repetition_ratio']:.1%}")
    print(f"  Patterns found: {analysis['has_patterns']}")
    print(f"  Max run length: {analysis['max_run_length']}")
    print(f"  Strategy:       {analysis['recommended_strategy']}")
    print(f"  Detection time: {analysis['detection_time_ms']:.3f} ms")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks = [
        ("Repetitive data selected PATTERN", 
         test_cases["repetitive"] and selector.select(test_cases["repetitive"]).strategy.name == "PATTERN"),
        
        ("Random data selected RAW or REFERENCE",
         selector.select(test_cases["random"]).strategy.name in ["RAW", "REFERENCE"]),
        
        ("Text data detected as text_like or mixed",
         selector.select(test_cases["text"]).characteristics.data_type in ["text_like", "mixed_structured"]),
        
        ("Small data selected RAW",
         selector.select(test_cases["small"]).strategy.name == "RAW"),
        
        ("Detection time < 10ms",
         selector.select(test_cases["repetitive"]).decision_time_ms < 10.0),
        
        ("Entropy range valid (0-8)",
         all(0 <= selector.select(d).characteristics.entropy <= 8 for d in test_cases.values())),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ VALIDATION COMPLETE - ALL TESTS PASSED")
    else:
        print("✗ VALIDATION FAILED - SOME TESTS FAILED")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
