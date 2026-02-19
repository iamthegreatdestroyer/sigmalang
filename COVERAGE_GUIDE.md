# Coverage Reporting Guide

## Overview

ΣLANG uses pytest-cov for code coverage reporting. This guide explains how to generate and interpret coverage reports.

## Quick Start

### Fast Coverage (Unit Tests - 5 minutes)

```bash
# Option 1: Using script
./run_coverage.sh --fast

# Option 2: Direct command
pytest tests/ \
  --cov=sigmalang \
  --cov-report=html \
  --timeout=120 \
  -k "not slow"
```

### Full Coverage (All Tests - 10 minutes)

```bash
# Option 1: Using script
./run_coverage.sh --full

# Option 2: Direct command
pytest tests/ \
  --cov=sigmalang \
  --cov-report=html \
  --cov-report=xml \
  --timeout=300
```

### View Results

```bash
# Open HTML report in browser
open htmlcov/index.html
```

## Configuration

### .coveragerc File

The `.coveragerc` file controls coverage behavior:

```ini
[run]
branch = True          # Track branch coverage
parallel = True        # Support parallel test execution
omit =
    */tests/*          # Exclude test files
    */venv/*           # Exclude virtual env

[report]
precision = 2          # 2 decimal places
show_missing = True    # Show uncovered lines
skip_covered = False   # Show all files

[html]
directory = htmlcov    # Output directory
```

### pytest.toml Configuration

```toml
[tool.pytest.ini_options]
# Coverage must use thread-based timeouts to work with coverage
timeout_method = "thread"
addopts = "--timeout=300"
```

## Understanding Coverage Reports

### Coverage Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| Line Coverage | % of lines executed | >85% |
| Branch Coverage | % of if/else branches taken | >75% |
| Missing | Lines/branches not covered | Minimize |

### Coverage Levels

```
90-100% : Excellent
80-89%  : Good
70-79%  : Acceptable
60-69%  : Poor
<60%    : Critical
```

## Interpreting the Report

### HTML Report Navigation

1. **Summary Page** - Overall coverage stats
2. **File View** - Drill down into specific files
3. **Line View** - See which lines are covered
   - Green: Covered (executed)
   - Red: Uncovered (not executed)
   - Orange: Partial coverage (branch not taken)

### Finding Uncovered Code

```bash
# View uncovered lines in terminal
pytest tests/ --cov=sigmalang --cov-report=term-missing

# Sample output:
# sigmalang/core/encoder.py  125  def encode()
#   Line 127 (if condition branch)  UNCOVERED
```

## Troubleshooting

### Coverage is Hanging

**Symptoms:**
```
Running pytest...
[hangs for several minutes]
^C to interrupt
```

**Causes:**
- Incorrect timeout_method configuration
- Test with infinite loops or deadlock
- Too small global timeout

**Solutions:**

```bash
# 1. Use thread-based timeout (in pyproject.toml)
[tool.pytest.ini_options]
timeout_method = "thread"

# 2. Increase individual test timeout
pytest tests/ --timeout=300

# 3. Skip slow/problematic tests
pytest tests/ -k "not slow"

# 4. Run fast coverage instead
./run_coverage.sh --fast
```

### Coverage Not Generated

**Check prerequisites:**
```bash
pip install pytest-cov
pip install coverage
pytest --version
```

**Run with verbose output:**
```bash
pytest tests/ --cov=sigmalang -v
```

**Clean and retry:**
```bash
rm -rf .coverage htmlcov/ .pytest_cache/
pytest tests/ --cov=sigmalang --cov-report=html
```

### Coverage Percentage Too Low

**Investigate uncovered code:**
```bash
# See which files have lowest coverage
pytest tests/ --cov=sigmalang --cov-report=term-missing | grep "UNCOVERED"

# Focus on specific file
pytest tests/test_encoder.py --cov=sigmalang/core/encoder --cov-report=term-missing
```

**Write additional tests:**
```python
def test_edge_case():
    """Test currently uncovered branch"""
    encoder = SigmaEncoder()
    result = encoder.encode("")  # Test edge case
    assert result is not None
```

## Continuous Monitoring

### GitHub Actions Integration

```yaml
# .github/workflows/coverage.yml
name: Coverage Report

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -e '.[dev]'
      - run: pytest --cov=sigmalang --cov-report=xml
      - uses: codecov/codecov-action@v2
        with:
          files: ./coverage.xml
```

### Coverage Badges

Add to README.md:

```markdown
[![Coverage Status](https://img.shields.io/codecov/c/github/iamthegreatdestroyer/sigmalang)](https://codecov.io/gh/iamthegreatdestroyer/sigmalang)
```

## Best Practices

### 1. Run Coverage Regularly

```bash
# Before committing
./run_coverage.sh --fast

# Before pushing
./run_coverage.sh --full
```

### 2. Maintain Coverage Levels

- Set minimum: 85% overall
- Maintain: 75% branch coverage
- Monitor: Declining coverage trends

### 3. Write Testable Code

```python
# Bad: Hard to test
def process():
    if random() > 0.5:
        do_something()

# Good: Testable
def process(random_func=None):
    if random_func is None:
        random_func = random
    if random_func() > 0.5:
        do_something()
```

### 4. Test Error Paths

```python
# Include error handling
def test_invalid_input():
    with pytest.raises(ValueError):
        encoder.encode(None)

def test_resource_cleanup():
    # Ensure cleanup happens
    encoder = SigmaEncoder()
    result = encoder.encode("text")
    # Verify state
```

## Performance Optimization

### Parallel Coverage

```bash
pytest tests/ \
  --cov=sigmalang \
  -n 4  \  # 4 workers
  --cov-report=html
```

### Selective Coverage

```bash
# Only report on sigmalang package
pytest tests/ \
  --cov=sigmalang \
  --cov-report=html \
  --cov-report=term:skip-covered
```

## Production Setup

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Checking coverage..."
COVERAGE=$(pytest tests/ --cov=sigmalang -q | tail -1)

if [[ $COVERAGE < 85 ]]; then
    echo "Coverage below 85%: $COVERAGE"
    echo "Please add more tests"
    exit 1
fi
```

### CI/CD Requirements

```yaml
# Fail if coverage below threshold
- run: pytest tests/ --cov=sigmalang --cov-fail-under=85
```

## Resources

- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Coverage Best Practices](https://coverage.readthedocs.io/en/latest/guide.html)

---

**Last Updated:** February 19, 2026
**Status:** Production Ready
