# REST API Reference

## Base URL

```
http://localhost:26080/api
```

## Authentication

Currently no authentication required. Implement as needed for production.

## Content Types

- **Request**: `application/json`
- **Response**: `application/json`

## Endpoints

### Health Check

#### GET /health

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-19T12:00:00Z"
}
```

#### GET /health/detailed

Get detailed health status including dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-19T12:00:00Z",
  "components": {
    "redis": "connected",
    "database": "connected",
    "memory": "normal"
  },
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

### Encoding

#### POST /api/encode

Encode text to ΣLANG format.

**Request:**
```json
{
  "text": "Machine learning transforms data",
  "optimization": "medium"
}
```

**Parameters:**
- `text` (required): Text to encode
- `optimization` (optional): `low`, `medium`, or `high` (default: `medium`)

**Response:**
```json
{
  "success": true,
  "text": "Machine learning transforms data",
  "encoded_bytes": 24,
  "original_bytes": 32,
  "compression_ratio": 1.33,
  "processing_time_ms": 2.5
}
```

### Decoding

#### POST /api/decode

Decode ΣLANG format back to text.

**Request:**
```json
{
  "encoded_data": "base64_encoded_data_here"
}
```

**Response:**
```json
{
  "success": true,
  "text": "Machine learning transforms data",
  "decoding_time_ms": 1.2
}
```

### Entity Extraction

#### POST /api/entities

Extract entities and relationships from text.

**Request:**
```json
{
  "text": "Apple Inc is located in Cupertino, California",
  "include_relationships": true
}
```

**Response:**
```json
{
  "entities": [
    {
      "text": "Apple Inc",
      "type": "ORGANIZATION",
      "confidence": 0.98
    },
    {
      "text": "Cupertino",
      "type": "LOCATION",
      "confidence": 0.95
    },
    {
      "text": "California",
      "type": "LOCATION",
      "confidence": 0.97
    }
  ],
  "relationships": [
    {
      "source": "Apple Inc",
      "relation": "located_in",
      "target": "Cupertino"
    }
  ]
}
```

### Analogy Solving

#### POST /api/analogy

Solve word analogies.

**Request:**
```json
{
  "word1": "king",
  "word2": "queen",
  "word3": "man",
  "top_k": 5
}
```

**Response:**
```json
{
  "word1": "king",
  "word2": "queen",
  "word3": "man",
  "answer": "woman",
  "confidence": 0.94,
  "top_candidates": [
    {"word": "woman", "confidence": 0.94},
    {"word": "lady", "confidence": 0.89},
    {"word": "female", "confidence": 0.82}
  ]
}
```

### Semantic Search

#### POST /api/search

Search through indexed data using semantic similarity.

**Request:**
```json
{
  "query": "technology companies",
  "limit": 10,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "query": "technology companies",
  "results": [
    {
      "id": "doc1",
      "text": "Apple Inc manufactures consumer electronics",
      "similarity": 0.92
    },
    {
      "id": "doc2",
      "text": "Microsoft develops software solutions",
      "similarity": 0.88
    }
  ],
  "total_results": 2
}
```

### Metrics

#### GET /metrics

Get Prometheus metrics.

**Response:**
```
# HELP sigmalang_encode_duration_ms Encoding operation duration
# TYPE sigmalang_encode_duration_ms histogram
sigmalang_encode_duration_ms_bucket{le="1"} 42
sigmalang_encode_duration_ms_bucket{le="5"} 156
...
```

## Error Responses

### 400 Bad Request

```json
{
  "error": "invalid_request",
  "message": "Text must be non-empty",
  "code": "INVALID_TEXT"
}
```

### 500 Internal Server Error

```json
{
  "error": "internal_error",
  "message": "Failed to process request",
  "code": "PROCESSING_ERROR"
}
```

## Rate Limiting

No default rate limiting. Configure via environment:

```bash
SIGMALANG_RATE_LIMIT=1000/minute
```

## Batch Operations

### Batch Encoding

```bash
curl -X POST http://localhost:26080/api/encode/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First text",
      "Second text",
      "Third text"
    ],
    "optimization": "medium"
  }'
```

## Response Headers

```
Content-Type: application/json
X-Request-ID: uuid-here
X-Processing-Time-Ms: 2.5
```

## Pagination

For list endpoints, use `limit` and `offset`:

```
GET /api/search?query=test&limit=10&offset=0
```

## API Documentation

- **Interactive**: http://localhost:26080/docs
- **Alternative**: http://localhost:26080/redoc
- **OpenAPI Schema**: http://localhost:26080/openapi.json

## Examples

### Python with requests

```python
import requests

response = requests.post(
    "http://localhost:26080/api/encode",
    json={"text": "Hello, World!"}
)
result = response.json()
print(f"Compression: {result['compression_ratio']}x")
```

### JavaScript with fetch

```javascript
const response = await fetch('http://localhost:26080/api/encode', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello, World!' })
});
const result = await response.json();
console.log(`Compression: ${result.compression_ratio}x`);
```

### cURL

```bash
curl -X POST http://localhost:26080/api/encode \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, World!"}'
```

## Next Steps

- Explore [Python API](python-api.md)
- Learn [CLI](cli.md) commands
- Try [Basic Usage](../getting-started/basic-usage.md) examples
