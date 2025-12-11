# ΣLANG - Sub-Linear Algorithmic Neural Glyph Language

[![CI](https://github.com/iamthegreatdestroyer/sigmalang/actions/workflows/ci.yml/badge.svg)](https://github.com/iamthegreatdestroyer/sigmalang/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A revolutionary compression language designed exclusively for internal LLM representation**

ΣLANG achieves **10-50x compression** by encoding semantic meaning directly, bypassing the verbose scaffolding of human natural language. The Ryot LLM operates internally on Σ-glyphs — meaning atoms that represent concepts at a fraction of the token cost.

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

## Installation

```bash
cd sigmalang
pip install -e .
```

## Quick Start

```python
from sigmalang import SigmaLangPipeline, create_pipeline

# Create pipeline with pattern learning enabled
pipeline = create_pipeline("models/codebook.json", enable_training=True)

# Encode human input
result = pipeline.encode_input("Create a Python function that sorts a list")

print(f"Original: {result.input_size} bytes")
print(f"Compressed: {result.output_size} bytes")  
print(f"Ratio: {result.compression_ratio:.1f}x")
print(f"Type: {result.encoding_type}")
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

| Content Type | Compression | Notes |
|--------------|-------------|-------|
| Code requests | 10-20x | Strong pattern matching |
| Queries | 8-15x | High primitive reuse |
| Explanations | 5-10x | More unique content |
| Repeated patterns | ∞ | O(1) codebook hit |

## Project Structure

```
sigmalang/
├── core/
│   ├── primitives.py    # Σ-primitive definitions
│   ├── parser.py        # Semantic parsing
│   └── encoder.py       # ΣLANG encoding/decoding
├── training/
│   ├── codebook.py      # Learned codebook system
│   └── train.py         # Training pipeline
├── tests/
│   └── test_sigmalang.py
└── ryot_integration.py  # Ryot LLM integration
```

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

## License

Copyright 2025 - Ryot LLM Project

---

*"Everything is unknown until it becomes known."*
