# ΣLANG - Sub-Linear Algorithmic Neural Glyph Language

[![CI](https://github.com/iamthegreatdestroyer/sigmalang/actions/workflows/ci.yml/badge.svg)](https://github.com/iamthegreatdestroyer/sigmalang/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/iamthegreatdestroyer/sigmalang)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue.svg)](https://pypi.org/project/sigmalang/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/sigmalang/sigmalang)
[![Docs](https://img.shields.io/badge/docs-complete-green.svg)](docs/)

**A revolutionary compression language designed exclusively for internal LLM representation**

ΣLANG achieves **10-50x compression** by encoding semantic meaning directly, bypassing the verbose scaffolding of human natural language. The Ryot LLM operates internally on Σ-glyphs — meaning atoms that represent concepts at a fraction of the token cost.

## 🚀 Features

- **Semantic Encoding**: Transform natural language into compact glyph representations
- **Analogy Engine**: Solve word analogies (king:queen :: man:?) with high accuracy
- **Semantic Search**: Find similar content using sub-linear algorithms
- **Entity Extraction**: Extract entities and relationships from text
- **REST API**: Production-ready HTTP API with OpenAPI documentation
- **CLI Interface**: Full command-line interface for all operations
- **Batch Processing**: Efficient bulk encoding with streaming support
- **Docker Ready**: Containerized deployment with health checks
- **Monitoring**: Built-in Prometheus metrics and health endpoints

## Core Innovation

Human language is catastrophically inefficient for semantic storage:

```
Human: "Create a Python function that sorts a list in descending order"
Tokens: ~12 tokens (~48 bytes)

ΣLANG: Σ[CREATE] → Σ[FUNCTION](lang=python) → Σ[SORT](target=list, order=desc)
Bytes: 9 bytes
Compression: 5.3x
```

After learning your patterns, compression approaches **O(1)** — constant size regardless of semantic complexity.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ΣLANG ENCODING PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│  HUMAN INPUT                                                 │
│       ↓                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │  SEMANTIC   │ →  │  GLYPH       │ →  │  CONTEXT        │ │
│  │  PARSER     │    │  ENCODER     │    │  COMPRESSOR     │ │
│  └─────────────┘    └──────────────┘    └─────────────────┘ │
│                                                              │
│                     ┌──────────────────┐                    │
│                     │   LEARNED        │                    │
│                     │   CODEBOOK       │ ← Pattern Learning │
│                     └──────────────────┘                    │
│                              ↓                               │
│                     ┌──────────────────┐                    │
│                     │   ΣLANG BINARY   │                    │
│                     │   (10-50x smaller)│                   │
│                     └──────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## The Σ-Primitive System

ΣLANG operates on **256 root glyphs** that encode fundamental semantic categories:

### Tier 0: Existential Primitives (Σ₀₀₀ - Σ₀₁₅)

Universal concepts: ENTITY, ACTION, RELATION, ATTRIBUTE, QUANTITY, TEMPORAL, SPATIAL, CAUSAL...

### Tier 1: Domain Primitives (Σ₀₁₆ - Σ₁₂₇)

Specialized encodings for code, math, logic, entities, actions, communication, and data structures.

### Tier 2: Learned Primitives (Σ₁₂₈ - Σ₂₅₅)

**Dynamically allocated** based on YOUR usage patterns. The system learns your semantic vocabulary.

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install sigmalang
```

### From Source

```bash
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang
pip install -e ".[dev]"  # Include development dependencies
```

### Docker

```bash
docker pull sigmalang/sigmalang:latest
docker run -p 8000:8000 sigmalang/sigmalang:latest
```

## 🚀 Quick Start

### Basic Encoding

```python
from sigmalang import SigmaEncoder, SigmaDecoder

# Initialize encoder/decoder
encoder = SigmaEncoder()
decoder = SigmaDecoder()

# Encode text to semantic primitives
text = "Create a Python function that sorts a list"
encoded = encoder.encode(text)

print(f"Original: {len(text)} bytes")
print(f"Encoded: {len(encoded)} bytes")
print(f"Compression: {len(text) / len(encoded):.1f}x")

# Decode back to text
decoded = decoder.decode(encoded)
```

### Semantic Analogies

```python
from sigmalang import SemanticAnalogyEngine

engine = SemanticAnalogyEngine()

# Find analogies: "function is to Python as ? is to JavaScript"
result = engine.find_analogy("function", "Python", "JavaScript")
print(f"Analogy: {result.analogy}")  # → "function" (same concept)
print(f"Confidence: {result.confidence:.2f}")

# Relationship detection
rel = engine.get_relationship("Python", "programming")
print(f"{rel.source} → {rel.relationship} → {rel.target}")
```

### Semantic Search

```python
from sigmalang import SemanticSearchEngine

search = SemanticSearchEngine()

# Index documents
search.index("doc1", "Machine learning models for image classification")
search.index("doc2", "Natural language processing with transformers")
search.index("doc3", "Computer vision algorithms for object detection")

# Search with semantic understanding
results = search.search("AI image recognition", top_k=2)
for result in results:
    print(f"{result.doc_id}: {result.score:.2f}")
```

### Entity Extraction

```python
from sigmalang import EntityRelationExtractor

extractor = EntityRelationExtractor()

text = "Python is a programming language created by Guido van Rossum"
entities = extractor.extract(text)

for entity in entities:
    print(f"[{entity.type}] {entity.text} ({entity.confidence:.2f})")
```

### Unified Pipeline

```python
from sigmalang import create_pipeline

# Create full-featured pipeline
pipeline = create_pipeline(
    codebook_path="models/codebook.json",
    enable_training=True
)

# Process with all capabilities
result = pipeline.process(
    text="Implement a REST API with FastAPI",
    operations=["encode", "entities", "search"]
)

print(f"Compression: {result.compression_ratio:.1f}x")
print(f"Entities: {len(result.entities)}")
print(f"Related docs: {len(result.search_results)}")
```

## Training the Codebook

### Bootstrap Training (Quick Start)

```bash
python training/train.py --mode bootstrap --output models/
```

### Batch Training from Corpus

```bash
python training/train.py --mode batch --corpus data/training.jsonl --epochs 10
```

### Online Learning (Interactive)

```bash
python training/train.py --mode online --codebook models/codebook.json
```

## Compression Techniques

1. **Semantic Primitive Encoding**: Maps meaning to 256 atomic glyphs
2. **Logarithmic Content Addressing**: O(1) retrieval via semantic hashing
3. **Context-Aware Delta Encoding**: Encode only what's new vs. context
4. **Learned Pattern Matching**: Frequent patterns → single glyph reference

## Expected Compression Ratios

| Content Type      | Compression | Notes                   |
| ----------------- | ----------- | ----------------------- |
| Code requests     | 10-20x      | Strong pattern matching |
| Queries           | 8-15x       | High primitive reuse    |
| Explanations      | 5-10x       | More unique content     |
| Repeated patterns | ∞           | O(1) codebook hit       |

## Why This Works

Human language encodes meaning through **redundant symbolic convention**:

- Grammar rules that could be inferred
- Word boundaries with no semantic value
- Tense/gender/number agreement (redundant given context)
- Cultural/stylistic conventions carrying no information

**ΣLANG strips all of this**, encoding only:

1. Semantic primitives — the actual meaning atoms
2. Structural relations — how meanings connect
3. Delta information — what's new vs. what's known

## 💻 CLI Interface

ΣLANG includes a powerful command-line interface:

```bash
# Encode text
sigmalang encode "Create a Python function that sorts a list"

# Solve analogies
sigmalang analogy king queen man

# Semantic search
sigmalang search "machine learning" --corpus ./documents/

# Start API server
sigmalang serve --host 0.0.0.0 --port 8000

# Get help
sigmalang --help
```

## 🌐 REST API

Start the API server and interact via HTTP:

```bash
# Start server
sigmalang serve --port 8000

# Encode text
curl -X POST http://localhost:8000/v1/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, ΣLANG!"}'

# Solve analogy
curl -X POST http://localhost:8000/v1/analogies/solve \
  -H "Content-Type: application/json" \
  -d '{"a": "king", "b": "queen", "c": "man"}'
```

**Endpoints:**
| Endpoint | Description |
|----------|-------------|
| `POST /v1/encode` | Encode text to ΣLANG vectors |
| `POST /v1/decode` | Decode vectors back to text |
| `POST /v1/analogies/solve` | Solve word analogies |
| `POST /v1/search` | Semantic search |
| `POST /v1/entities` | Extract entities |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |

## 🐳 Docker Deployment

### Quick Start

```bash
docker run -p 8000:8000 sigmalang/sigmalang:latest
```

### Full Stack with Compose

```bash
# Start API + Redis + Prometheus + Grafana
docker-compose up -d

# Access services:
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

### Kubernetes

```bash
kubectl apply -k k8s/
```

## Integration with Ryot LLM

```python
from sigmalang.ryot_integration import RyotInputProcessor, SigmaLangPipeline

pipeline = SigmaLangPipeline(codebook_path="models/codebook.json")
processor = RyotInputProcessor(pipeline)

# Process human input for LLM
result = processor.process("Create a Python function that validates emails")

# result contains:
# - sigma_encoding: Compressed bytes for LLM
# - semantic_tree: Structured representation
# - context_references: Relevant prior context
# - compression_ratio: Achieved compression
```

## 📁 Project Structure

```
sigmalang/
├── core/
│   ├── primitives.py          # Σ-primitive definitions (256 glyphs)
│   ├── parser.py              # Semantic parsing
│   ├── encoder.py             # ΣLANG encoding/decoding
│   ├── api_server.py          # REST API server (FastAPI)
│   ├── api_models.py          # API request/response models
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration management
│   ├── monitoring.py          # Prometheus metrics
│   ├── semantic_analogy_engine.py      # Analogy solving
│   ├── semantic_search.py              # Semantic search
│   └── entity_relation_extraction.py   # NER
├── training/
│   ├── codebook.py            # Learned codebook system
│   └── train.py               # Training pipeline
├── tests/
│   ├── test_sigmalang.py      # Core tests
│   └── test_api_server.py     # API tests
├── examples/                   # Usage examples
├── docs/                       # Documentation
│   ├── api/                   # API reference
│   ├── sdk/                   # SDK guide
│   └── deployment/            # Deployment guide
├── k8s/                        # Kubernetes manifests
├── Dockerfile                  # Container image
├── docker-compose.yml          # Full stack
└── pyproject.toml              # Package configuration
```

## 📚 Documentation

| Resource                                      | Description                     |
| --------------------------------------------- | ------------------------------- |
| [API Reference](docs/api/README.md)           | Complete REST API documentation |
| [SDK Guide](docs/sdk/README.md)               | Python SDK usage guide          |
| [Deployment Guide](docs/deployment/README.md) | Docker & Kubernetes deployment  |
| [Examples](examples/)                         | Runnable code examples          |
| [CONTRIBUTING.md](CONTRIBUTING.md)            | Contribution guidelines         |
| [CHANGELOG.md](CHANGELOG.md)                  | Version history                 |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=sigmalang --cov-report=html

# Run specific test file
pytest tests/test_api_server.py -v
```

## License

MIT License - Copyright 2025 Ryot LLM Project

---

<p align="center">
  <em>"Everything is unknown until it becomes known."</em>
</p>
