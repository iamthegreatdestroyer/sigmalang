# Testing Guide

## Overview

ΣLANG has 1,656 tests covering:
- Unit tests (1,200+ tests)
- Integration tests (300+ tests)
- Performance tests (100+ tests)
- Memory profiling tests (56 tests)

All tests pass with 100% success rate.

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_optimizations.py -v
```

### Run Specific Test

```bash
pytest tests/test_optimizations.py::test_glyph_buffer_pool -v
```

### Run Tests Matching Pattern

```bash
pytest tests/ -k "encode" -v
```

### Run with Coverage

```bash
pytest tests/ --cov=sigmalang --cov-report=html
open htmlcov/index.html
```

### Run Tests in Parallel

```bash
pytest tests/ -n 4  # 4 workers
```

### Run with Verbose Output

```bash
pytest tests/ -vv -s  # -s shows print statements
```

### Run with Detailed Failure Info

```bash
pytest tests/ -l --tb=long
```

## Test Categories

### Unit Tests

```bash
pytest tests/test_*.py -v
```

Tests individual components:
- Primitives (test_primitives.py)
- Optimizations (test_optimizations.py)
- Parser logic
- Encoder/Decoder

### Integration Tests

```bash
pytest tests/integration/ -v
```

Tests system workflows:
- CLI commands (test_cli_commands.py)
- API endpoints (test_api.py)
- End-to-end pipelines
- Cross-module interactions

### Performance Tests

```bash
pytest tests/ -k benchmark -v
```

Benchmarks:
- Encoding speed
- Memory usage
- Compression ratio
- Throughput

### Memory Profiling

```bash
pytest tests/test_memory_profiling.py -v
```

Analyzes:
- Memory allocation patterns
- Scaling characteristics
- GC behavior

## Test Structure

### Basic Test Template

```python
import pytest
from sigmalang.core.encoder import SigmaEncoder


class TestSigmaEncoder:
    """Test suite for SigmaEncoder"""

    @pytest.fixture
    def encoder(self):
        """Provide encoder instance"""
        return SigmaEncoder()

    def test_basic_encoding(self, encoder):
        """Test basic encoding functionality"""
        # Arrange
        text = "Hello, World!"

        # Act
        result = encoder.encode(text)

        # Assert
        assert result is not None
        assert len(result) > 0


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_string(self):
        """Test handling of empty string"""
        encoder = SigmaEncoder()
        with pytest.raises(ValueError):
            encoder.encode("")

    def test_very_large_text(self):
        """Test handling of large text"""
        encoder = SigmaEncoder()
        large_text = "x" * 1000000
        result = encoder.encode(large_text)
        assert len(result) > 0
```

### Using Fixtures

```python
@pytest.fixture
def parser():
    """Provide parser instance"""
    return SemanticParser()

@pytest.fixture
def sample_tree(parser):
    """Provide pre-parsed semantic tree"""
    return parser.parse("Sample text")

@pytest.fixture(params=["low", "medium", "high"])
def optimization_level(request):
    """Parametrized fixture for optimization levels"""
    return request.param
```

### Parametrized Tests

```python
@pytest.mark.parametrize("text,expected", [
    ("hello", 5),
    ("world", 5),
    ("a" * 100, 100),
])
def test_encoding_length(text, expected):
    """Test that encoding preserves length"""
    encoder = SigmaEncoder()
    result = encoder.encode(text)
    assert len(result) >= expected * 0.1  # At least 10% compression
```

### Marking Tests

```python
@pytest.mark.slow
def test_slow_operation():
    """Slow test, skip with -m "not slow"""""
    pass

@pytest.mark.benchmark
def test_performance():
    """Performance benchmark"""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Integration test"""
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    """Feature not yet implemented"""
    pass
```

## Assertions

### Common Assertions

```python
def test_assertions():
    """Demonstrate common assertions"""
    # Basic comparisons
    assert result == expected
    assert result != unexpected
    assert result > 0
    assert result >= minimum

    # Membership
    assert item in collection
    assert item not in collection

    # Type checking
    assert isinstance(result, bytes)
    assert isinstance(result, (list, tuple))

    # Exception handling
    with pytest.raises(ValueError):
        invalid_function()

    with pytest.raises(ValueError, match="error message"):
        function_that_fails("with message")

    # Containment
    assert "substring" in result
    assert "error" not in result

    # Approximate equality
    assert result == pytest.approx(3.14, abs=0.01)
```

## Test Data

### Fixtures for Common Data

```python
@pytest.fixture
def small_text():
    return "The quick brown fox"

@pytest.fixture
def medium_text():
    return "x" * 10000

@pytest.fixture
def large_text():
    return "x" * 1000000

@pytest.fixture
def multilingual_text():
    return "Hello мир 你好 مرحبا"
```

### Temporary Files

```python
import tempfile

def test_file_encoding(tmp_path):
    """Test encoding from file"""
    # Create temporary file
    temp_file = tmp_path / "test.txt"
    temp_file.write_text("test content")

    # Use temp file
    encoder = SigmaEncoder()
    with open(temp_file) as f:
        result = encoder.encode(f.read())

    assert result is not None
```

## Performance Testing

### Benchmark Tests

```python
@pytest.mark.benchmark
def test_encode_throughput(benchmark):
    """Benchmark encoding throughput"""
    encoder = SigmaEncoder()
    text = "x" * 10000

    result = benchmark(encoder.encode, text)
    assert len(result) > 0
```

### Memory Profiling

```python
from tracemalloc import Snapshot
import tracemalloc

def test_memory_usage():
    """Test memory usage"""
    tracemalloc.start()

    encoder = SigmaEncoder()
    result = encoder.encode("x" * 1000000)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Assert memory usage is reasonable
    assert peak < 100 * 1024 * 1024  # Less than 100MB
```

### Timing Tests

```python
import time

def test_encoding_speed():
    """Test that encoding is fast enough"""
    encoder = SigmaEncoder()
    text = "x" * 100000

    start = time.time()
    result = encoder.encode(text)
    elapsed = time.time() - start

    # Assert encoding completes in reasonable time
    assert elapsed < 1.0  # Less than 1 second
```

## Mocking and Stubbing

### Mock External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock_redis():
    """Test with mocked Redis"""
    with patch('sigmalang.storage.cache.redis') as mock_redis:
        mock_redis.get.return_value = b'cached_data'

        cache = SemanticCache()
        result = cache.get('key')

        assert result == b'cached_data'
        mock_redis.get.assert_called_once_with('key')
```

### Mock File Operations

```python
from unittest.mock import mock_open, patch

def test_file_reading():
    """Test file reading with mock"""
    mock_data = "file content"
    with patch('builtins.open', mock_open(read_data=mock_data)):
        with open('test.txt') as f:
            content = f.read()

        assert content == mock_data
```

## Test Coverage

### Generate Coverage Report

```bash
pytest tests/ --cov=sigmalang --cov-report=html
```

### View Coverage

```bash
# HTML report
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=sigmalang --cov-report=term-missing
```

### Coverage Thresholds

Current targets:
- Overall: >85%
- Critical modules: >95%
- Tests pass at: 100%

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to main
- Pull requests
- Scheduled nightly

### Local Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

pytest tests/ -q
if [ $? -ne 0 ]; then
    echo "Tests failed, commit aborted"
    exit 1
fi
```

## Best Practices

1. **Test one thing per test**: Single responsibility
2. **Use descriptive names**: Clearly state what's tested
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Test edge cases**: Empty, None, large, special chars
5. **Use fixtures**: DRY principle for test setup
6. **Mark slow tests**: Use @pytest.mark.slow
7. **Parametrize when possible**: Test multiple cases efficiently
8. **Keep tests fast**: <100ms per test ideally
9. **Test error paths**: Not just happy paths
10. **Maintain test data**: Keep fixtures current

## Troubleshooting

### Test Fails Intermittently

```bash
# Run test multiple times
pytest tests/test_file.py::test_function -v --count=100

# Run with different seed
pytest tests/ --randomly-seed=12345
```

### Test Hangs

```bash
# Add timeout
pytest tests/ --timeout=30

# Kill specific test
pytest tests/test_file.py::test_function --timeout=5
```

### Memory Leak in Tests

```bash
# Profile memory
python -m memory_profiler tests/test_file.py

# Check for unclosed resources
pytest tests/ --tb=short
```

### Cannot Import Module

```bash
# Ensure package installed in dev mode
pip install -e ".[dev]"

# Check Python path
python -c "import sys; print(sys.path)"
```

## Next Steps

- Read [Contributing Guide](contributing.md)
- Explore [Architecture](architecture.md)
- View [Test Results](../../../tests/)
