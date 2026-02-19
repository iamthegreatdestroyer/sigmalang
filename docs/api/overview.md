# API Reference Overview

ΣLANG provides three interfaces for interacting with the system:

1. **REST API** - HTTP endpoints for all operations
2. **Python API** - Direct library usage in Python code
3. **CLI** - Command-line interface for shell scripting

## Quick Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/detailed` | GET | Detailed health status |
| `/metrics` | GET | Prometheus metrics |
| `/api/encode` | POST | Encode text |
| `/api/decode` | POST | Decode data |
| `/api/entities` | POST | Extract entities |
| `/api/analogy` | POST | Solve word analogies |
| `/api/search` | POST | Semantic search |
| `/docs` | GET | OpenAPI documentation |
| `/redoc` | GET | ReDoc documentation |

### Python API Modules

| Module | Purpose |
|--------|---------|
| `sigmalang.core.encoder` | Text encoding |
| `sigmalang.core.decoder` | Data decoding |
| `sigmalang.core.parser` | Semantic parsing |
| `sigmalang.core.analogy` | Analogy engine |
| `sigmalang.core.primitives` | Semantic primitives |
| `sigmalang.core.processor` | Batch processing |

### CLI Commands

| Command | Description |
|---------|-------------|
| `sigmalang encode` | Encode text |
| `sigmalang decode` | Decode data |
| `sigmalang entities` | Extract entities |
| `sigmalang analogy` | Solve analogies |
| `sigmalang search` | Semantic search |

## Getting Started

### REST API

```bash
# Start API server
docker compose up sigmalang

# Make request
curl -X POST http://localhost:26080/api/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, World!"}'
```

### Python API

```python
from sigmalang.core.encoder import SigmaEncoder

encoder = SigmaEncoder()
encoded = encoder.encode(text)
```

### CLI

```bash
sigmalang encode "Hello, World!"
```

## Authentication

The REST API currently does not require authentication. For production deployments, implement API key or OAuth2 authentication as needed.

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 404 | Not found |
| 500 | Server error |
| 503 | Service unavailable |

### Error Response Format

```json
{
  "error": "invalid_text",
  "message": "Text must be non-empty",
  "code": "INVALID_INPUT"
}
```

### Python Exceptions

```python
from sigmalang.core.exceptions import (
    EncodingError,
    DecodingError,
    ParsingError,
    InvalidConfigError
)
```

## Rate Limiting

REST API has no rate limiting by default. Configure limits via:

```bash
export SIGMALANG_RATE_LIMIT=1000/minute
```

## Documentation

- **Interactive**: http://localhost:26080/docs (Swagger UI)
- **Alternative**: http://localhost:26080/redoc (ReDoc)
- **OpenAPI Schema**: http://localhost:26080/openapi.json

## Performance Considerations

- Batch processing is faster than individual requests
- Enable caching for repeated operations
- Use appropriate optimization level for your use case
- Monitor metrics at http://localhost:26900

## Next Steps

- Read [REST API](rest-api.md) documentation
- Explore [Python API](python-api.md) reference
- Learn [CLI](cli.md) commands
