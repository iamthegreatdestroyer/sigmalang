# Basic Usage

## Command-Line Interface

### Encoding Text

**Encode from stdin:**
```bash
echo "Hello, World!" | sigmalang encode
```

**Encode with string:**
```bash
sigmalang encode "The quick brown fox jumps over the lazy dog"
```

**Encode from file:**
```bash
sigmalang encode -i input.txt -o encoded.bin
```

**Encode with optimization level:**
```bash
sigmalang encode -i input.txt -o output.bin --optimization high
```

### Decoding Text

**Decode file:**
```bash
sigmalang decode -i encoded.bin -o decoded.txt
```

**Decode and view metadata:**
```bash
sigmalang decode -i encoded.bin -o output.txt --show-metadata
```

### Entity Extraction

**Extract entities from text:**
```bash
sigmalang entities "Apple Inc is located in Cupertino, California"
```

**Extract with relationship info:**
```bash
sigmalang entities "Steve Jobs founded Apple" --relationships
```

### Entity Search

**Find similar entities:**
```bash
sigmalang search-entities "technology companies"
```

### Analogy Solving

**Solve word analogy:**
```bash
sigmalang analogy --word1 king --word2 queen --word3 man
```

Output: `man:woman (confidence: 0.94)`

## Python API

### Basic Encoding

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser

parser = SemanticParser()
encoder = SigmaEncoder()

text = "Machine learning is transforming industries"
tree = parser.parse(text)
encoded = encoder.encode(tree)

print(f"Compression ratio: {len(text)/len(encoded):.1f}x")
```

### Batch Processing

```python
from sigmalang.core.processor import BatchProcessor

processor = BatchProcessor(batch_size=100)
texts = ["Text 1", "Text 2", ...]

results = processor.encode_batch(texts)
for result in results:
    print(f"{result.text}: {result.compression_ratio:.1f}x")
```

### Custom Configuration

```python
from sigmalang.core.config import SigmaConfig
from sigmalang.core.encoder import SigmaEncoder

config = SigmaConfig(
    optimization_level="high",
    cache_enabled=True,
    max_buffer_size=1024*1024,  # 1MB
)

encoder = SigmaEncoder(config=config)
```

### Error Handling

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.exceptions import EncodingError

encoder = SigmaEncoder()

try:
    encoded = encoder.encode(tree)
except EncodingError as e:
    print(f"Encoding failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## REST API

### Health Check

```bash
curl http://localhost:26080/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-19T12:00:00Z"
}
```

### Encode via REST API

```bash
curl -X POST http://localhost:26080/api/encode \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, World!",
    "optimization": "high"
  }'
```

Response:
```json
{
  "success": true,
  "text": "Hello, World!",
  "encoded_bytes": 8,
  "original_bytes": 13,
  "compression_ratio": 1.625,
  "processing_time_ms": 2.5
}
```

### Extract Entities via REST API

```bash
curl -X POST http://localhost:26080/api/entities \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc is in California"}'
```

Response:
```json
{
  "entities": [
    {
      "text": "Apple Inc",
      "type": "ORGANIZATION"
    },
    {
      "text": "California",
      "type": "LOCATION"
    }
  ]
}
```

## Performance Tips

1. **Use batch processing** for multiple texts
2. **Enable caching** for repeated operations
3. **Adjust optimization level** based on your needs:
   - `low`: Fast, minimal compression
   - `medium`: Balanced (default)
   - `high`: Maximum compression, slower
4. **Monitor metrics** via Prometheus: `http://localhost:26900`

## Common Use Cases

### Log Compression

```python
logs = [
    "User logged in from 192.168.1.1",
    "Request processed in 125ms",
    "Database query executed successfully",
]

processor = BatchProcessor()
compressed = processor.encode_batch(logs)
```

### API Request Compression

```python
# Compress request bodies for transmission
response = requests.post(
    "https://api.example.com/data",
    data=encoder.encode(tree),
    headers={"Content-Encoding": "sigmalang"}
)
```

### Large Text Processing

```python
# Process large documents efficiently
with open("large_document.txt") as f:
    processor = BatchProcessor(batch_size=50)
    for chunk in processor.chunk_text(f, chunk_size=10000):
        encoded = processor.encode_batch(chunk)
```

## Next Steps

- Explore [Concepts](../concepts/overview.md)
- Read [API Reference](../api/overview.md) for complete documentation
- Check [Deployment](../deployment/docker.md) for production setup
