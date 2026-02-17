# ΣLANG Documentation

Welcome to the official documentation for **ΣLANG** (Sigma Language) - the Sub-Linear Algorithmic Neural Glyph Language designed for extreme LLM compression.

## What is ΣLANG?

ΣLANG achieves **10-50x compression** by encoding semantic meaning directly, bypassing the verbose scaffolding of human natural language. It represents concepts as compact glyphs that preserve meaning while dramatically reducing token costs.

```
Human: "Create a Python function that sorts a list in descending order"
Tokens: ~12 tokens (~48 bytes)

ΣLANG: Σ[CREATE] → Σ[FUNCTION](lang=python) → Σ[SORT](target=list, order=desc)
Bytes: 9 bytes
Compression: 5.3x
```

## Key Features

- ✅ **Semantic Encoding**: Transform natural language into compact glyph representations
- ✅ **Analogy Engine**: Solve word analogies (king:queen :: man:?) with high accuracy
- ✅ **Semantic Search**: Find similar content using sub-linear algorithms
- ✅ **Entity Extraction**: Extract entities and relationships from text
- ✅ **REST API**: Production-ready HTTP API with OpenAPI documentation
- ✅ **CLI Interface**: Full command-line interface for all operations
- ✅ **Docker Ready**: Containerized deployment with health checks
- ✅ **Monitoring**: Built-in Prometheus metrics and health endpoints

## Quick Start

### Installation

```bash
pip install sigmalang
```

### Basic Usage

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser

# Initialize
parser = SemanticParser()
encoder = SigmaEncoder()

# Encode text
text = "Machine learning transforms data into insights"
tree = parser.parse(text)
encoded = encoder.encode(tree)

print(f"Original: {len(text)} bytes")
print(f"Encoded: {len(encoded)} bytes")
print(f"Compression: {len(text)/len(encoded):.1f}x")
```

## Architecture

ΣLANG operates on a three-tier primitive system:

### Tier 0: Existential Primitives (Σ₀₀₀ - Σ₀₁₅)
Universal concepts: ENTITY, ACTION, RELATION, ATTRIBUTE, QUANTITY, TEMPORAL, SPATIAL, CAUSAL

### Tier 1: Domain Primitives (Σ₀₁₆ - Σ₁₂₇)
Specialized encodings for code, math, logic, entities, actions, communication, and data structures

### Tier 2: Learned Primitives (Σ₁₂₈ - Σ₂₅₅)
Dynamically allocated based on usage patterns

## Navigation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting Started](getting-started/installation.md)**

    Install ΣLANG and run your first compression

- :material-book-open-variant: **[Concepts](concepts/overview.md)**

    Understand semantic primitives and compression

- :material-api: **[API Reference](api/overview.md)**

    Explore REST API, Python API, and CLI

- :material-kubernetes: **[Deployment](deployment/kubernetes.md)**

    Deploy to Kubernetes with Helm

</div>

## Resources

- [GitHub Repository](https://github.com/iamthegreatdestroyer/sigmalang)
- [PyPI Package](https://pypi.org/project/sigmalang/)
- [Docker Hub](https://hub.docker.com/r/sigmalang/sigmalang)
- [API Documentation](api/rest-api.md)

## License

ΣLANG is released under the MIT License. See [License](about/license.md) for details.
