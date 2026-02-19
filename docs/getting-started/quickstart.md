# Quick Start (5 Minutes)

## Start the API Server

### Option 1: Docker (Recommended)

```bash
docker compose up -d
curl http://localhost:26080/health
```

### Option 2: Local Python

```bash
pip install -e ".[dev]"
sigmalang-server
# In another terminal:
curl http://localhost:26080/health
```

## Encode Text

### Using CLI

```bash
sigmalang encode "Machine learning transforms data into insights"
```

Output:
```
Original bytes: 52
Encoded bytes: 18
Compression: 2.9x
```

### Using cURL

```bash
curl -X POST http://localhost:26080/api/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning transforms data into insights"}'
```

Response:
```json
{
  "text": "Machine learning transforms data into insights",
  "encoded_bytes": 18,
  "compression_ratio": 2.9,
  "tokens": ["MACHINE", "LEARNING", "TRANSFORMS", "DATA", "INSIGHTS"]
}
```

## Extract Entities

```bash
sigmalang entities "Apple Inc is headquartered in Cupertino, California"
```

Output:
```
Entities found:
- ORGANIZATION: Apple Inc
- LOCATION: Cupertino, California
```

## Solve Analogies

```bash
sigmalang analogy --word1 king --word2 queen --word3 man
```

Output:
```
Analogy: king is to queen as man is to woman
Confidence: 0.94
```

## Python API

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser

# Initialize
parser = SemanticParser()
encoder = SigmaEncoder()

# Parse and encode text
text = "The quick brown fox jumps over the lazy dog"
tree = parser.parse(text)
encoded = encoder.encode(tree)

print(f"Original: {len(text)} bytes")
print(f"Encoded: {len(encoded)} bytes")
print(f"Compression: {len(text)/len(encoded):.1f}x")
```

## Next Steps

- Read [Basic Usage](basic-usage.md) for more examples
- Explore the [API Reference](../api/overview.md)
- Check [Deployment](../deployment/docker.md) options
- See [Monitoring](../operations/monitoring.md) setup

## API Documentation

View interactive API docs at:
- **Swagger UI**: http://localhost:26080/docs
- **ReDoc**: http://localhost:26080/redoc
