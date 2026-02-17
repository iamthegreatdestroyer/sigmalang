# Benchmark Regression Testing

Continuous benchmark regression testing to detect performance regressions.

## Overview

The benchmark regression tool runs performance tests and compares results against a baseline to detect regressions > 10% (configurable).

## Features

- ✅ Runs all benchmark tests via pytest-benchmark
- ✅ Saves results to timestamped JSON files
- ✅ Automatic baseline comparison
- ✅ Configurable regression threshold
- ✅ CI/CD ready (exit code 1 on regression)
- ✅ Detailed performance reports

## Installation

Requires `pytest-benchmark`:

```bash
pip install pytest pytest-benchmark
```

## Usage

### Run Benchmarks and Compare

```bash
# Run benchmarks and auto-compare against latest baseline
python scripts/benchmark_regression.py --run

# Run with custom alert threshold (default: 10%)
python scripts/benchmark_regression.py --run --alert-threshold 15

# Run with additional pytest args
python scripts/benchmark_regression.py --run --pytest-args="--verbose --timeout=600"
```

### Compare Specific Baseline

```bash
# Compare latest results against specific baseline
python scripts/benchmark_regression.py --compare benchmark_results/baseline.json
```

### Options

- `--run`: Run benchmarks and save results
- `--compare BASELINE`: Compare against specific baseline file
- `--alert-threshold PERCENT`: Regression alert threshold (default: 10.0)
- `--output-dir DIR`: Output directory (default: benchmark_results)
- `--pytest-args "ARGS"`: Additional pytest arguments

## Output

### Benchmark Results

Results are saved to `benchmark_results/benchmark_YYYYMMDD_HHMMSS.json` in pytest-benchmark JSON format:

```json
{
  "benchmarks": [
    {
      "name": "test_cache_lookup_vs_linear_search",
      "stats": {
        "mean": 0.000105,
        "min": 0.000091,
        "max": 0.000967,
        "stddev": 0.000025,
        "iterations": 11026
      }
    }
  ]
}
```

### Regression Report

```
======================================================================
[REPORT] BENCHMARK REGRESSION REPORT
======================================================================

Total Benchmarks: 2
Regressions Found: 0
Alert Threshold: ±10.0%

[PASS] No significant regressions detected

Benchmark                                          Mean         Min          Max
--------------------------------------------------------------------------------------
test_cache_lookup_vs_linear_search                       0.11ms       0.09ms       0.97ms
test_iterative_vs_recursive_traversal                    3.73ms       2.68ms      11.64ms
======================================================================
```

### With Regressions

```
Benchmark                                          Baseline      Current       Change
----------------------------------------------------------------------------------------
test_slow_operation                                    1.50ms        2.10ms    [REG] +40.0%
test_fast_operation                                    0.10ms        0.08ms    [IMP] -20.0%

[FAIL] Performance regressions detected!
```

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/ci.yml`:

```yaml
benchmark:
  runs-on: ubuntu-latest
  needs: test
  steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        pip install pytest pytest-benchmark numpy
        pip install -e .

    - name: Run benchmark regression
      run: |
        python scripts/benchmark_regression.py --run --alert-threshold 10
      continue-on-error: false  # Fail on regression
```

### GitLab CI

```yaml
benchmark:
  stage: test
  script:
    - pip install pytest pytest-benchmark numpy
    - pip install -e .
    - python scripts/benchmark_regression.py --run --alert-threshold 10
  allow_failure: false
```

## Available Benchmarks

### Current Benchmarks

1. **test_cache_lookup_vs_linear_search** (`test_optimizations.py`)
   - Compares FastPrimitiveCache vs linear search
   - Target: < 0.2ms mean time

2. **test_iterative_vs_recursive_traversal** (`test_optimizations.py`)
   - Compares iterative vs recursive tree traversal
   - Target: < 5ms mean time

### Benchmark Markers

To run only benchmark tests:

```bash
pytest tests/ -k benchmark --benchmark-only
```

To run tests WITHOUT benchmarks:

```bash
pytest tests/ -k "not benchmark"
```

## Performance Targets

| Benchmark | Target | Threshold |
|-----------|--------|-----------|
| Cache lookup | < 0.2ms | 10% |
| Tree traversal | < 5ms | 10% |
| Encoding | < 10ms | 15% |
| Decoding | < 10ms | 15% |

## Troubleshooting

### No Benchmarks Found

If no benchmarks are found, check:

1. Tests are marked with `@pytest.mark.benchmark`
2. Tests use the `benchmark` fixture
3. File names match `test_*.py` or `*_benchmark.py`

Example benchmark test:

```python
import pytest

def test_my_operation(benchmark):
    """Benchmark my operation."""
    result = benchmark(my_function, arg1, arg2)
    assert result is not None
```

### Skipped Benchmarks

Some benchmarks may be skipped in the analogy engine tests. This is expected for tests that require specific data or longer setup.

### Excluding Slow Tests

Exclude very slow benchmarks during development:

```bash
python scripts/benchmark_regression.py --run \
  --pytest-args="--ignore=tests/test_hd_vs_lsh_benchmark.py"
```

## Baseline Management

### Creating Initial Baseline

First run creates the baseline automatically:

```bash
python scripts/benchmark_regression.py --run
# Output: [INFO]  No baseline found for comparison
#         This will be used as the new baseline
```

### Updating Baseline

Run benchmarks again to create a new data point:

```bash
python scripts/benchmark_regression.py --run
# Compares against previous run automatically
```

### Manual Baseline Selection

Set a specific file as baseline for comparison:

```bash
python scripts/benchmark_regression.py --compare benchmark_results/baseline_20260101.json
```

## Best Practices

1. **Run on same hardware**: Benchmark results vary by hardware
2. **Consistent conditions**: Same Python version, dependencies, OS
3. **Warm-up**: Benchmarks include warm-up by default (`--benchmark-warmup=on`)
4. **Multiple iterations**: More iterations = more reliable results
5. **Disable GC**: Use `--benchmark-disable-gc` for consistency
6. **Track history**: Keep all benchmark JSON files for trend analysis

## Exit Codes

- `0`: No regressions detected
- `1`: Performance regressions found OR error occurred

## Example Workflow

```bash
# 1. Run initial baseline
python scripts/benchmark_regression.py --run

# 2. Make code changes
# ... edit code ...

# 3. Run benchmarks again to check for regressions
python scripts/benchmark_regression.py --run --alert-threshold 10

# 4. If regressions found, investigate
# [FAIL] Performance regressions detected!

# 5. Fix performance issues and re-run
python scripts/benchmark_regression.py --run
# [PASS] No performance regressions
```

## References

- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [Performance Testing Best Practices](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Google SRE Book - Performance](https://sre.google/sre-book/monitoring-distributed-systems/)
