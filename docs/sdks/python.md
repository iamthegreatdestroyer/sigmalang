# Python SDK

The Python SDK provides direct access to ΣLANG's encoding and compression capabilities.

## Installation

```bash
pip install sigmalang
```

## Quick Start

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser

parser = SemanticParser()
encoder = SigmaEncoder()

text = "Machine learning transforms data into insights"
tree = parser.parse(text)
encoded = encoder.encode(tree)

print(f"Compression: {len(text) / len(encoded):.1f}x")
```

## API Reference

See [Python API Documentation](../api/python-api.md) for complete reference.

## Examples

### Encoding

```python
from sigmalang.core.encoder import SigmaEncoder

encoder = SigmaEncoder()
encoded = encoder.encode(tree)
```

### Batch Processing

```python
from sigmalang.core.processor import BatchProcessor

processor = BatchProcessor(batch_size=100)
results = processor.encode_batch(texts)
```

### Analogy Solving

```python
from sigmalang.core.analogy import AnalogyEngine

engine = AnalogyEngine()
result = engine.solve("king", "queen", "man")
```

## Configuration

```python
from sigmalang.core.config import SigmaConfig
from sigmalang.core.encoder import SigmaEncoder

config = SigmaConfig(
    optimization_level="high",
    cache_enabled=True
)

encoder = SigmaEncoder(config=config)
```

## Documentation

- [Python API Reference](../api/python-api.md)
- [Basic Usage](../getting-started/basic-usage.md)
- [Architecture](../development/architecture.md)

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/iamthegreatdestroyer/sigmalang)
- Check [FAQ](../getting-started/basic-usage.md)
