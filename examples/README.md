# ΣLANG Examples

This directory contains runnable examples demonstrating ΣLANG capabilities.

## Quick Start

```bash
# Install ΣLANG
pip install -e ..

# Run any example
python 01_basic_encoding.py
```

## Examples

| File                                                 | Description                      |
| ---------------------------------------------------- | -------------------------------- |
| [01_basic_encoding.py](01_basic_encoding.py)         | Basic text encoding/decoding     |
| [02_analogies.py](02_analogies.py)                   | Solving word analogies           |
| [03_semantic_search.py](03_semantic_search.py)       | Semantic search across documents |
| [04_entity_extraction.py](04_entity_extraction.py)   | Named entity recognition         |
| [05_cli_usage.py](05_cli_usage.py)                   | CLI interface examples           |
| [06_api_client.py](06_api_client.py)                 | REST API client examples         |
| [07_batch_processing.py](07_batch_processing.py)     | Batch encoding operations        |
| [08_advanced_analogies.py](08_advanced_analogies.py) | Advanced analogy patterns        |

## Running Examples

Each example is self-contained and can be run directly:

```bash
python 01_basic_encoding.py
```

## Requirements

- Python 3.9+
- ΣLANG installed (`pip install -e ..`)
- For API examples: running API server (`sigmalang serve`)
