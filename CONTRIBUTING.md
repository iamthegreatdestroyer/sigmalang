# Contributing to ΣLANG

Thank you for your interest in contributing to ΣLANG! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something innovative.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sigmalang.git
   cd sigmalang
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/sigmalang.git
   ```

## Development Setup

### Prerequisites
- Python 3.9 or higher
- NumPy

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Verify Setup
```bash
# Run tests
python -m pytest tests/ -v

# Run demo
python tests/test_sigmalang.py --demo
```

## Making Changes

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `perf/description` - Performance improvements

### Workflow
1. Create a branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes, following the [Style Guidelines](#style-guidelines)

3. Write or update tests as needed

4. Run the test suite:
   ```bash
   python -m pytest tests/ -v
   ```

5. Run the compression demo to verify performance:
   ```bash
   python tests/test_sigmalang.py --demo
   ```

6. Commit your changes:
   ```bash
   git commit -m "feat: add new compression strategy for code patterns"
   ```

## Testing

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_sigmalang.py -v

# With coverage
python -m pytest tests/ -v --cov=sigmalang
```

### Writing Tests
- Place tests in `tests/` directory
- Use descriptive test names: `test_compression_achieves_target_ratio`
- Include both positive and edge case tests
- Test compression ratios for new features

### Test Structure
```python
class TestMyFeature(unittest.TestCase):
    def setUp(self):
        # Setup code
        pass
    
    def test_feature_basic_functionality(self):
        """Test that the feature works in the basic case."""
        # Test code
        pass
    
    def test_feature_edge_case(self):
        """Test feature behavior with edge cases."""
        # Test code
        pass
```

## Submitting Changes

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add LSH-based pattern matching for improved compression
fix: resolve index error in glyph decoder
docs: update README with new API examples
perf: optimize semantic hash computation by 40%
```

### Pull Request Process
1. Update documentation if needed
2. Ensure all tests pass
3. Update CHANGELOG.md with your changes
4. Submit PR against `main` branch
5. Fill out the PR template completely
6. Wait for review

### Review Process
- PRs require at least one approval
- Address all review comments
- Keep PRs focused and reasonably sized
- Large features should be broken into smaller PRs

## Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints where practical
- Maximum line length: 100 characters
- Use docstrings for public functions/classes

### Documentation
```python
def encode_input(self, text: str) -> EncodingResult:
    """
    Encode human text input to ΣLANG format.
    
    Args:
        text: Human natural language input
        
    Returns:
        EncodingResult with compressed bytes and metadata
        
    Raises:
        ValueError: If text is empty
        
    Example:
        >>> result = pipeline.encode_input("Create a function")
        >>> print(result.compression_ratio)
        3.5
    """
```

### Code Organization
- Keep functions focused and small
- Use meaningful variable names
- Group related functionality in classes
- Separate concerns between modules

## Areas for Contribution

### High Priority
- [ ] Improve compression ratios for natural language
- [ ] Fix decoder round-trip edge cases
- [ ] Add more Σ-primitive definitions for new domains
- [ ] Optimize LSH index performance
- [ ] Add benchmarking suite

### Medium Priority
- [ ] Support for additional pattern types
- [ ] Visualization tools for compression analysis
- [ ] Integration examples for other LLM frameworks
- [ ] Expand test coverage

### Documentation
- [ ] API reference documentation
- [ ] Tutorial notebooks
- [ ] Architecture deep-dive
- [ ] Performance tuning guide

## Questions?

Open an issue with the `question` label or start a discussion.

---

Thank you for contributing to ΣLANG! 🚀
