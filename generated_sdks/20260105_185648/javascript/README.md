# SigmaLang Javascript SDK

Enterprise-grade SDK for SigmaLang semantic compression.

## Installation

```bash
# Installation instructions for javascript
```

## Usage

```python
from sigmalang_sdk import SigmaLang

client = SigmaLang("your-api-key")
compressed = client.compress("Hello world")
text = client.decompress(compressed)
```

## API Reference

- `compress(text, options)` - Compress text
- `decompress(compressed)` - Decompress data
- `analyze(text)` - Analyze semantic structure

## License

MIT License
