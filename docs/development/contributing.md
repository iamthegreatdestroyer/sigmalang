# Contributing Guide

## Welcome!

We love contributions! Whether you're fixing bugs, adding features, or improving documentation, your help is welcome.

## Getting Started

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/test_optimizations.py -v
```

### Build Documentation

```bash
pip install mkdocs mkdocs-material
mkdocs serve
# Access at http://localhost:8000
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### 2. Write Tests First

Create tests in `tests/test_your_feature.py`:

```python
def test_my_new_feature():
    """Test description"""
    # Setup
    input_data = "test"

    # Execute
    result = my_function(input_data)

    # Assert
    assert result == expected_output
```

Run tests:

```bash
# Single test
pytest tests/test_your_feature.py::test_my_new_feature -v

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=sigmalang --cov-report=html
```

### 3. Implement Feature

Add your code to appropriate module in `sigmalang/`:

```python
# sigmalang/core/my_feature.py
def my_function(text):
    """Function description"""
    return processed_text
```

### 4. Verify Tests Pass

```bash
pytest tests/ -v
```

All tests must pass. If tests fail:
1. Debug the issue
2. Update implementation
3. Re-run tests

### 5. Code Review

Ensure code quality:

```bash
# Type checking
mypy sigmalang/

# Linting
flake8 sigmalang/ --max-line-length=100

# Code formatting
black sigmalang/
```

### 6. Commit Changes

```bash
git add .
git commit -m "feat: Add my new feature

- Description of what changed
- Why it matters
- Any breaking changes"
```

Commit message format:
```
type: subject

Body explaining the change...

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub.

## Code Style

### Python Style Guide

We follow PEP 8 with some customizations:

```python
# Imports: stdlib, third-party, local
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from fastapi import FastAPI

from sigmalang.core.encoder import SigmaEncoder


# Classes
class MyClass:
    """Class description."""

    def __init__(self):
        """Initialize."""
        self.value = None

    def my_method(self, param: str) -> str:
        """Method description."""
        return param.upper()


# Functions
def my_function(text: str, length: int = 100) -> str:
    """Function description.

    Args:
        text: Input text
        length: Maximum length

    Returns:
        Processed text
    """
    return text[:length]
```

### Type Hints

Always include type hints:

```python
from typing import List, Dict, Optional

def process_texts(texts: List[str]) -> Dict[str, int]:
    """Process multiple texts."""
    return {text: len(text) for text in texts}

def optional_param(value: Optional[str] = None) -> str:
    """Handle optional parameters."""
    return value or "default"
```

### Docstrings

Use docstring format:

```python
def my_function(param: str) -> str:
    """One-line summary.

    Longer description if needed. Explain the purpose,
    behavior, and any important details.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid

    Examples:
        >>> my_function("test")
        "TEST"
    """
    pass
```

## Testing Guidelines

### Test Structure

```python
import pytest
from sigmalang.core.my_module import MyClass


class TestMyClass:
    """Test suite for MyClass."""

    @pytest.fixture
    def instance(self):
        """Provide test instance."""
        return MyClass()

    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result is not None

    def test_error_handling(self, instance):
        """Test error handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_param)
```

### Test Coverage

Aim for >80% coverage:

```bash
pytest tests/ --cov=sigmalang --cov-report=html
# Open htmlcov/index.html to view coverage
```

### Performance Tests

Mark performance tests:

```python
@pytest.mark.benchmark
def test_encoding_performance():
    """Benchmark encoding performance."""
    import time

    start = time.time()
    result = encoder.encode(large_text)
    elapsed = time.time() - start

    # Assert performance target
    assert elapsed < 0.1  # 100ms
```

## Documentation Guidelines

### Update Docs

Update relevant documentation when adding features:

1. **API docs** - Add to `docs/api/`
2. **Concepts** - Add to `docs/concepts/`
3. **Examples** - Add to `docs/getting-started/`
4. **README** - Update project README

### Write Examples

Include practical examples:

```python
"""
Example: Encoding text

Usage:
    from sigmalang.core.encoder import SigmaEncoder

    encoder = SigmaEncoder()
    encoded = encoder.encode(tree)
"""
```

## Pull Request Process

### Before Submitting

- [ ] Tests pass: `pytest tests/`
- [ ] Code formatted: `black sigmalang/`
- [ ] Type check passes: `mypy sigmalang/`
- [ ] Coverage maintained: `>80%`
- [ ] Documentation updated
- [ ] Commit messages clear

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No new warnings

## Screenshots (if applicable)
Include screenshots for UI changes
```

## Common Tasks

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_my_feature.py

# Specific test
pytest tests/test_my_feature.py::test_specific

# With verbosity
pytest -v

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Parallel execution
pytest -n 4
```

### Building Docs

```bash
# Install mkdocs
pip install mkdocs mkdocs-material

# Preview
mkdocs serve

# Build
mkdocs build

# Deploy
mkdocs gh-deploy
```

### Checking Code Quality

```bash
# Format code
black sigmalang/

# Check style
flake8 sigmalang/

# Type checking
mypy sigmalang/

# Security check
bandit -r sigmalang/
```

## Reporting Issues

### Report a Bug

Create an issue with:
1. Clear title
2. Description of the bug
3. Steps to reproduce
4. Expected vs actual behavior
5. Environment (OS, Python version, etc.)
6. Relevant code/logs

### Suggest a Feature

Create an issue with:
1. Feature title
2. Description and motivation
3. Example use case
4. Proposed implementation (if applicable)
5. Alternative approaches

## Code Review Standards

Reviewers look for:
- ✅ Tests pass
- ✅ Code quality
- ✅ Documentation
- ✅ Performance impact
- ✅ Security considerations
- ✅ Backwards compatibility

## Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **GitHub Wikis**: Community documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Check [existing issues](https://github.com/iamthegreatdestroyer/sigmalang/issues)
- Read [documentation](https://sigmalang.io)
- Open a new issue for help

Thank you for contributing! 🎉
