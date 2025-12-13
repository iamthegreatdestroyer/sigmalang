# ΣLANG API Reference

Complete reference documentation for the ΣLANG REST API.

## Base URL

```
https://api.sigmalang.io/v1  # Production
http://localhost:8000/v1     # Local development
```

## Authentication

All API requests require authentication via API key:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.sigmalang.io/v1/encode
```

Or via header:

```bash
curl -H "X-API-Key: YOUR_API_KEY" https://api.sigmalang.io/v1/encode
```

## Rate Limiting

| Plan       | Requests/Minute | Burst Size |
| ---------- | --------------- | ---------- |
| Free       | 60              | 10         |
| Pro        | 600             | 50         |
| Enterprise | Unlimited       | Unlimited  |

Rate limit headers in response:

- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Seconds until window reset

---

## Endpoints

### Core Operations

| Method | Endpoint     | Description                  |
| ------ | ------------ | ---------------------------- |
| POST   | `/v1/encode` | Encode text to ΣLANG vectors |
| POST   | `/v1/decode` | Decode vectors back to text  |
| POST   | `/v1/batch`  | Batch encoding/decoding      |

### Analogy Operations

| Method | Endpoint                | Description                   |
| ------ | ----------------------- | ----------------------------- |
| POST   | `/v1/analogies/solve`   | Solve A:B::C:? analogies      |
| POST   | `/v1/analogies/explain` | Explain analogy relationships |

### Search & Discovery

| Method | Endpoint         | Description                   |
| ------ | ---------------- | ----------------------------- |
| POST   | `/v1/search`     | Semantic search across corpus |
| POST   | `/v1/similarity` | Calculate text similarity     |

### Entity Extraction

| Method | Endpoint       | Description                    |
| ------ | -------------- | ------------------------------ |
| POST   | `/v1/entities` | Extract entities and relations |

### Management

| Method | Endpoint   | Description        |
| ------ | ---------- | ------------------ |
| GET    | `/health`  | Health check       |
| GET    | `/ready`   | Readiness probe    |
| GET    | `/metrics` | Prometheus metrics |
| GET    | `/v1/info` | System information |

---

## Encoding

### POST /v1/encode

Encode text into high-dimensional semantic vectors.

**Request Body:**

```json
{
  "text": "Create a Python function that sorts a list",
  "normalize": true,
  "output_format": "json",
  "include_metadata": true,
  "dimensions": 512
}
```

| Field              | Type     | Default      | Description                           |
| ------------------ | -------- | ------------ | ------------------------------------- |
| `text`             | string   | **required** | Text to encode                        |
| `texts`            | string[] | null         | Multiple texts (batch)                |
| `normalize`        | boolean  | true         | Normalize vector to unit length       |
| `output_format`    | enum     | "json"       | One of: json, compact, binary, base64 |
| `include_metadata` | boolean  | false        | Include encoding metadata             |
| `dimensions`       | integer  | null         | Override vector dimensions            |

**Response:**

```json
{
  "success": true,
  "request_id": "a1b2c3d4e5f6",
  "timestamp": "2024-12-13T10:30:00Z",
  "processing_time_ms": 12.5,
  "vector": [0.123, -0.456, 0.789, ...],
  "dimensions": 512,
  "token_count": 8,
  "metadata": {
    "encoding_type": "semantic",
    "compression_ratio": 5.3
  }
}
```

**cURL Example:**

```bash
curl -X POST https://api.sigmalang.io/v1/encode \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "Hello, ΣLANG!",
    "normalize": true
  }'
```

---

### POST /v1/decode

Decode semantic vectors back to natural language text.

**Request Body:**

```json
{
  "vector": [0.123, -0.456, 0.789, ...],
  "max_length": 512,
  "temperature": 1.0
}
```

| Field         | Type      | Default      | Description              |
| ------------- | --------- | ------------ | ------------------------ |
| `vector`      | float[]   | **required** | Vector to decode         |
| `vectors`     | float[][] | null         | Multiple vectors (batch) |
| `max_length`  | integer   | 512          | Maximum output length    |
| `temperature` | float     | 1.0          | Sampling temperature     |

**Response:**

```json
{
  "success": true,
  "text": "Hello, ΣLANG!",
  "confidence": 0.95,
  "alternatives": [{ "text": "Hello, Sigma Language!", "confidence": 0.82 }]
}
```

---

## Analogies

### POST /v1/analogies/solve

Solve analogical reasoning problems: A is to B as C is to ?

**Request Body:**

```json
{
  "a": "king",
  "b": "queen",
  "c": "man",
  "top_k": 5,
  "analogy_type": "semantic",
  "include_explanation": true
}
```

| Field                 | Type    | Default      | Description                                                  |
| --------------------- | ------- | ------------ | ------------------------------------------------------------ |
| `a`                   | string  | **required** | First term                                                   |
| `b`                   | string  | **required** | Second term (related to a)                                   |
| `c`                   | string  | **required** | Third term                                                   |
| `d`                   | string  | null         | Optional: verify if d is correct                             |
| `top_k`               | integer | 5            | Number of solutions to return                                |
| `analogy_type`        | enum    | "semantic"   | Type: semantic, structural, proportional, causal, functional |
| `include_explanation` | boolean | false        | Include detailed explanation                                 |

**Response:**

```json
{
  "success": true,
  "solutions": [
    {
      "answer": "woman",
      "confidence": 0.94,
      "relation": "gender_counterpart",
      "reasoning": "The relationship male→female is preserved"
    },
    {
      "answer": "female",
      "confidence": 0.78,
      "relation": "gender_attribute",
      "reasoning": "Direct gender mapping"
    }
  ],
  "best_answer": "woman",
  "confidence": 0.94,
  "explanation": "The analogy king:queen::man:woman holds because...",
  "relation_type": "semantic_parallel"
}
```

**cURL Example:**

```bash
curl -X POST https://api.sigmalang.io/v1/analogies/solve \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "a": "king",
    "b": "queen",
    "c": "man",
    "top_k": 3
  }'
```

---

### POST /v1/analogies/explain

Explain the relationship between analogy components.

**Request Body:**

```json
{
  "a": "Paris",
  "b": "France",
  "c": "Tokyo",
  "d": "Japan",
  "depth": 2
}
```

**Response:**

```json
{
  "success": true,
  "explanation": "Paris is the capital of France, just as Tokyo is the capital of Japan.",
  "relation_ab": "capital_of",
  "relation_cd": "capital_of",
  "similarity_score": 0.97,
  "structural_analysis": {
    "pattern": "X is to Y as X' is to Y'",
    "mapping_type": "geographic_relation"
  },
  "semantic_analysis": {
    "shared_attributes": ["city", "capital", "cultural_center"],
    "relation_strength": 0.95
  }
}
```

---

## Search

### POST /v1/search

Perform semantic search across a corpus of documents.

**Request Body:**

```json
{
  "query": "machine learning for beginners",
  "corpus": [
    "Introduction to ML concepts",
    "Advanced deep learning techniques",
    "ML fundamentals for newcomers",
    "Database optimization strategies"
  ],
  "top_k": 10,
  "mode": "semantic",
  "threshold": 0.5
}
```

| Field       | Type     | Default      | Description                                 |
| ----------- | -------- | ------------ | ------------------------------------------- |
| `query`     | string   | **required** | Search query                                |
| `corpus`    | string[] | **required** | Documents to search                         |
| `top_k`     | integer  | 10           | Maximum results                             |
| `mode`      | enum     | "semantic"   | Search mode: exact, semantic, hybrid, fuzzy |
| `threshold` | float    | 0.0          | Minimum similarity score                    |

**Response:**

```json
{
  "success": true,
  "results": [
    {
      "text": "ML fundamentals for newcomers",
      "score": 0.89,
      "index": 2,
      "highlights": ["ML", "fundamentals", "newcomers"]
    },
    {
      "text": "Introduction to ML concepts",
      "score": 0.76,
      "index": 0
    }
  ],
  "total_searched": 4,
  "query_vector_dimensions": 512
}
```

---

## Entity Extraction

### POST /v1/entities

Extract named entities and relationships from text.

**Request Body:**

```json
{
  "text": "Apple CEO Tim Cook announced the new iPhone at WWDC 2024 in Cupertino.",
  "include_relations": true,
  "confidence_threshold": 0.5
}
```

**Response:**

```json
{
  "success": true,
  "entities": [
    {
      "text": "Apple",
      "entity_type": "organization",
      "confidence": 0.98,
      "start": 0,
      "end": 5
    },
    {
      "text": "Tim Cook",
      "entity_type": "person",
      "confidence": 0.96,
      "start": 10,
      "end": 18
    },
    {
      "text": "iPhone",
      "entity_type": "product",
      "confidence": 0.94
    },
    {
      "text": "WWDC 2024",
      "entity_type": "event",
      "confidence": 0.91
    },
    {
      "text": "Cupertino",
      "entity_type": "location",
      "confidence": 0.95
    }
  ],
  "relations": [
    {
      "source": "Tim Cook",
      "target": "Apple",
      "relation_type": "works_for",
      "confidence": 0.93
    },
    {
      "source": "WWDC 2024",
      "target": "Cupertino",
      "relation_type": "located_in",
      "confidence": 0.88
    }
  ]
}
```

---

## Health & Monitoring

### GET /health

Check API health status.

```bash
curl https://api.sigmalang.io/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-12-13T10:30:00Z",
  "components": {
    "encoder": { "status": "healthy" },
    "decoder": { "status": "healthy" },
    "cache": { "status": "healthy" }
  }
}
```

### GET /ready

Kubernetes readiness probe.

```bash
curl https://api.sigmalang.io/ready
```

**Response:**

```json
{
  "ready": true,
  "dependencies": {
    "redis": "connected",
    "models": "loaded"
  }
}
```

### GET /metrics

Prometheus-compatible metrics endpoint.

```bash
curl https://api.sigmalang.io/metrics
```

**Response (text/plain):**

```
# HELP sigmalang_requests_total Total API requests
# TYPE sigmalang_requests_total counter
sigmalang_requests_total{method="POST",endpoint="/v1/encode",status="200"} 12345

# HELP sigmalang_request_duration_seconds Request latency
# TYPE sigmalang_request_duration_seconds histogram
sigmalang_request_duration_seconds_bucket{le="0.01"} 8234
sigmalang_request_duration_seconds_bucket{le="0.05"} 11892
sigmalang_request_duration_seconds_bucket{le="0.1"} 12100
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "success": false,
  "error": "Human-readable error message",
  "error_code": "VALIDATION_ERROR",
  "request_id": "a1b2c3d4",
  "timestamp": "2024-12-13T10:30:00Z",
  "details": {
    "field": "text",
    "message": "Field is required"
  }
}
```

### Error Codes

| Code                   | HTTP Status | Description                     |
| ---------------------- | ----------- | ------------------------------- |
| `VALIDATION_ERROR`     | 400         | Invalid request data            |
| `AUTHENTICATION_ERROR` | 401         | Missing or invalid API key      |
| `AUTHORIZATION_ERROR`  | 403         | Insufficient permissions        |
| `NOT_FOUND`            | 404         | Resource not found              |
| `RATE_LIMIT_EXCEEDED`  | 429         | Too many requests               |
| `INTERNAL_ERROR`       | 500         | Server error                    |
| `SERVICE_UNAVAILABLE`  | 503         | Service temporarily unavailable |

---

## SDKs

### Python

```bash
pip install sigmalang
```

```python
from sigmalang import SigmalangClient

client = SigmalangClient(api_key="YOUR_API_KEY")

# Encode
result = client.encode("Hello, ΣLANG!")
print(result.vector)

# Analogy
solution = client.analogy.solve("king", "queen", "man")
print(f"Answer: {solution.best_answer}")  # woman
```

### JavaScript/TypeScript

```bash
npm install sigmalang
```

```typescript
import { SigmalangClient } from "sigmalang";

const client = new SigmalangClient({ apiKey: "YOUR_API_KEY" });

// Encode
const result = await client.encode("Hello, ΣLANG!");
console.log(result.vector);

// Analogy
const solution = await client.analogy.solve("king", "queen", "man");
console.log(`Answer: ${solution.bestAnswer}`); // woman
```

---

## Changelog

### v1.0.0 (2024-12-13)

- Initial release
- Core encoding/decoding endpoints
- Analogy solving and explanation
- Semantic search
- Entity extraction
- Prometheus metrics integration
