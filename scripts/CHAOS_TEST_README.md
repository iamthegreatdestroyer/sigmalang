# Chaos Testing Framework

Chaos engineering framework for testing SigmaLang system resilience under adverse conditions.

## Overview

The chaos testing framework validates that the system handles edge cases, failures, and stress conditions gracefully without crashes or data corruption.

## Installation

No additional dependencies required - uses standard library and existing SigmaLang components.

## Usage

### Run All Scenarios

```bash
python scripts/chaos_test.py --scenarios all
```

### Run Specific Scenarios

```bash
# Single scenario
python scripts/chaos_test.py --scenarios input-fuzzing

# Multiple scenarios
python scripts/chaos_test.py --scenarios input-fuzzing,circuit-breaker,memory-pressure
```

### Verbose Mode

```bash
python scripts/chaos_test.py --scenarios all --verbose
```

## Available Scenarios

### 1. Input Fuzzing (`input-fuzzing`)

Tests system resilience against malformed and edge-case inputs:

- **Empty/whitespace**: `""`, `" "`, `"\n"`, `"\t"`
- **Very long strings**: 100K-1M characters
- **Binary data**: Non-UTF-8 bytes
- **Unicode edge cases**: Emojis, multi-byte characters, null bytes
- **Injection attempts**: SQL injection, XSS, path traversal
- **Special characters**: Control chars, escape sequences

**Expected Behavior**: System should either:
- Process input successfully, OR
- Raise ValueError/UnicodeDecodeError gracefully (no crashes)

**Test Count**: 28 malformed inputs

### 2. Circuit Breaker (`circuit-breaker`)

Simulates backend failures to test circuit breaker resilience:

- **Random failures**: 30% of requests fail with ConnectionError
- **Graceful degradation**: System continues processing valid requests
- **No cascading failures**: Isolated failures don't crash the system

**Expected Behavior**:
- Accept failures gracefully
- Continue processing after failures

**Test Count**: 100 requests with random failures

### 3. Rate Limiter (`rate-limiter`)

Tests system behavior under burst traffic:

- **Rapid requests**: 200 requests as fast as possible
- **Throughput measurement**: Tracks requests/second
- **Rate limit enforcement**: Verifies rate limiting (if implemented)

**Expected Behavior**:
- Handle burst traffic without crashes
- Return 429 errors if rate-limited (acceptable)
- Maintain system stability

**Test Count**: 200 rapid-fire requests

### 4. Graceful Shutdown (`graceful-shutdown`)

Validates clean shutdown during active operations:

- **Background workers**: 3 concurrent threads processing requests
- **Shutdown signal**: Triggered during active work
- **Request completion**: All in-flight requests should finish

**Expected Behavior**:
- Active requests complete before shutdown
- No orphaned requests after shutdown
- `active_requests == 0` at shutdown

**Test Count**: 1 shutdown scenario

### 5. Memory Pressure (`memory-pressure`)

Tests system behavior under memory stress:

- **Large inputs**: 200KB-500KB text inputs
- **Repeated processing**: 50 iterations
- **Memory handling**: MemoryError handling

**Expected Behavior**:
- Process large inputs successfully
- Handle MemoryError gracefully (if out of memory)
- No memory leaks

**Test Count**: 50 large input encodings

### 6. Slow Network (`slow-network`)

⚠️ **Not Implemented** - Requires HTTP layer

Simulates network latency and timeouts.

### 7. Pod Kill (`pod-kill`)

⚠️ **Not Implemented** - Requires containerization

Simulates sudden process termination.

## Test Results

### Success Criteria

- ✅ **100% Pass Rate**: All tests should pass
- ✅ **No Crashes**: System handles all edge cases gracefully
- ✅ **Graceful Errors**: Invalid inputs raise appropriate exceptions

### Sample Output

```
======================================================================
[SUMMARY] CHAOS TESTING SUMMARY
======================================================================
Total Scenarios: 5
Total Tests: 379
Passed: 379
Failed: 0
Overall Success Rate: 100.0%
Total Duration: 6.14s

[PASS] ALL CHAOS TESTS PASSED - System is resilient!
======================================================================
```

### Performance Metrics

- **Input Fuzzing**: ~87 malformed inputs/second
- **Circuit Breaker**: ~5000 requests/second with failures
- **Rate Limiter**: ~2200 requests/second burst traffic
- **Memory Pressure**: ~12 large inputs/second

## CI/CD Integration

Add to GitHub Actions workflow:

```yaml
- name: Run Chaos Tests
  run: |
    python scripts/chaos_test.py --scenarios all
  continue-on-error: false
```

## Architecture

### Components

1. **ChaosScenario**: Enum of available scenarios
2. **ChaosTestResult**: Test result data structure
3. **ChaosTester**: Main orchestrator
4. **Scenario Classes**:
   - `InputFuzzingChaos`
   - `CircuitBreakerChaos`
   - `RateLimiterChaos`
   - `GracefulShutdownChaos`
   - `MemoryPressureChaos`

### Design Principles

- **Fail-Safe**: Tests validate graceful failure handling
- **Isolation**: Each scenario is independent
- **Observability**: Detailed error reporting and metrics
- **Windows Compatible**: Safe console output (no emoji crashes)

## Extending

### Adding New Scenarios

1. Create new scenario class:

```python
class NewChaos:
    def __init__(self, api):
        self.api = api
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()

    def run(self) -> ChaosTestResult:
        # Your test logic
        return ChaosTestResult(...)
```

2. Add to `ChaosScenario` enum
3. Add to `run_scenario()` switch
4. Update `parse_scenarios()` for "all"

## Troubleshooting

### Windows Console Encoding Errors

The framework includes `safe_print()` to handle Windows console encoding:
- Replaces emojis with `[PASS]`, `[FAIL]`, etc.
- Falls back to ASCII if needed

### Parser/Encoder Errors

The chaos test uses the internal parser + encoder pipeline:
```python
tree = parser.parse(text)
encoded = encoder.encode(tree, original_text=text)
```

This bypasses the incomplete API service layer.

## References

- [Chaos Engineering Principles](https://principlesofchaos.org/)
- [Netflix Chaos Monkey](https://netflix.github.io/chaosmonkey/)
- [Google SRE Book - Testing for Reliability](https://sre.google/sre-book/testing-reliability/)
