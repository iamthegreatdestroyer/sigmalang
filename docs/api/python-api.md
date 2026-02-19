# Python API Reference

## Installation

```bash
pip install sigmalang
```

## Core Modules

### SigmaEncoder

Encodes text to ΣLANG format.

```python
from sigmalang.core.encoder import SigmaEncoder

encoder = SigmaEncoder()
encoded = encoder.encode(tree)
```

**Methods:**
- `encode(tree: SemanticTree) -> bytes`: Encode semantic tree
- `encode_text(text: str) -> bytes`: Encode raw text

**Parameters:**
- `config`: Optional SigmaConfig object

### SigmaDecoder

Decodes ΣLANG format back to text.

```python
from sigmalang.core.decoder import SigmaDecoder

decoder = SigmaDecoder()
text = decoder.decode(encoded_data)
```

**Methods:**
- `decode(data: bytes) -> str`: Decode bytes to text
- `decode_tree(data: bytes) -> SemanticTree`: Decode to semantic tree

### SemanticParser

Parses text into semantic trees.

```python
from sigmalang.core.parser import SemanticParser

parser = SemanticParser()
tree = parser.parse("The quick brown fox jumps")
```

**Methods:**
- `parse(text: str) -> SemanticTree`: Parse text to tree
- `parse_batch(texts: List[str]) -> List[SemanticTree]`: Parse multiple texts

### BatchProcessor

Process multiple texts efficiently.

```python
from sigmalang.core.processor import BatchProcessor

processor = BatchProcessor(batch_size=100)
results = processor.encode_batch(texts)
```

**Methods:**
- `encode_batch(texts: List[str]) -> List[EncodingResult]`
- `decode_batch(data: List[bytes]) -> List[str]`
- `process(texts: List[str]) -> List[ProcessingResult]`

### AnalogyEngine

Solve word analogies.

```python
from sigmalang.core.analogy import AnalogyEngine

engine = AnalogyEngine()
result = engine.solve(
    word1="king",
    word2="queen",
    word3="man"
)
print(f"Answer: {result.word}")
```

**Methods:**
- `solve(word1, word2, word3) -> AnalogyResult`
- `solve_top_k(word1, word2, word3, k=5) -> List[AnalogyResult]`
- `solve_batch(analogies) -> List[AnalogyResult]`

## Data Structures

### SemanticTree

Represents hierarchical semantic structure.

```python
from sigmalang.core.primitives import SemanticTree, SemanticNode

root = SemanticNode(primitive=ExistentialPrimitive.ENTITY)
tree = SemanticTree(root=root, source_text="...")
```

**Attributes:**
- `root: SemanticNode`: Root node
- `source_text: str`: Original text
- `metadata: Dict`: Additional info

### SemanticNode

Represents individual semantic unit.

```python
from sigmalang.core.primitives import SemanticNode, ExistentialPrimitive

node = SemanticNode(
    primitive=ExistentialPrimitive.ENTITY,
    value="example",
    confidence=0.95
)
```

**Attributes:**
- `primitive: ExistentialPrimitive`: Semantic type
- `value: str`: Node value
- `confidence: float`: Confidence score (0-1)
- `children: List[SemanticNode]`: Child nodes

### ExistentialPrimitive

Enumeration of semantic primitives.

```python
from sigmalang.core.primitives import ExistentialPrimitive

# Tier 0 primitives
ExistentialPrimitive.ENTITY      # 0
ExistentialPrimitive.ACTION      # 1
ExistentialPrimitive.RELATION    # 2
ExistentialPrimitive.ATTRIBUTE   # 3
ExistentialPrimitive.QUANTITY    # 4
ExistentialPrimitive.TEMPORAL    # 5
ExistentialPrimitive.SPATIAL     # 6
ExistentialPrimitive.CAUSAL      # 7
```

## Configuration

### SigmaConfig

Configure encoding behavior.

```python
from sigmalang.core.config import SigmaConfig

config = SigmaConfig(
    optimization_level="high",  # low, medium, high
    cache_enabled=True,
    max_buffer_size=1024*1024,
    semantic_indexing=True
)

encoder = SigmaEncoder(config=config)
```

**Options:**
- `optimization_level`: Compression vs speed tradeoff
- `cache_enabled`: Enable result caching
- `max_buffer_size`: Maximum buffer allocation
- `semantic_indexing`: Build searchable index
- `num_workers`: Parallel processing workers

## Exception Handling

```python
from sigmalang.core.exceptions import (
    EncodingError,
    DecodingError,
    ParsingError,
    InvalidConfigError,
    CacheError
)

try:
    encoded = encoder.encode(tree)
except EncodingError as e:
    print(f"Encoding failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

### Basic Encoding

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser

parser = SemanticParser()
encoder = SigmaEncoder()

text = "Machine learning transforms data into insights"
tree = parser.parse(text)
encoded = encoder.encode(tree)

print(f"Original: {len(text)} bytes")
print(f"Encoded: {len(encoded)} bytes")
print(f"Compression: {len(text)/len(encoded):.1f}x")
```

### Batch Processing

```python
from sigmalang.core.processor import BatchProcessor

processor = BatchProcessor(batch_size=50)

texts = [
    "First document content",
    "Second document content",
    "Third document content",
]

results = processor.encode_batch(texts)
for result in results:
    print(f"{result.text}: {result.compression_ratio:.1f}x")
```

### Analogy Engine

```python
from sigmalang.core.analogy import AnalogyEngine

engine = AnalogyEngine()

# Single analogy
result = engine.solve("king", "queen", "man")
print(f"Answer: {result.word} (confidence: {result.confidence:.2f})")

# Multiple candidates
candidates = engine.solve_top_k("king", "queen", "man", k=5)
for candidate in candidates:
    print(f"{candidate.word}: {candidate.confidence:.2f}")
```

### Entity Extraction

```python
from sigmalang.core.parser import SemanticParser

parser = SemanticParser()
tree = parser.parse("Apple Inc is located in Cupertino, California")

for node in tree.root.children:
    if node.primitive == ExistentialPrimitive.ENTITY:
        print(f"Entity: {node.value}")
```

### Custom Configuration

```python
from sigmalang.core.config import SigmaConfig
from sigmalang.core.encoder import SigmaEncoder

# Maximum compression
config = SigmaConfig(
    optimization_level="high",
    cache_enabled=True,
    semantic_indexing=True
)

encoder = SigmaEncoder(config=config)
encoded = encoder.encode(tree)

# Fast encoding
config_fast = SigmaConfig(
    optimization_level="low",
    cache_enabled=False
)

encoder_fast = SigmaEncoder(config=config_fast)
```

## Performance Tips

1. **Use BatchProcessor** for multiple texts
2. **Enable caching** for repeated operations
3. **Choose appropriate optimization level**:
   - `low`: ~10MB/s, 5-8x compression
   - `medium`: ~5MB/s, 10-20x compression
   - `high`: ~1MB/s, 20-50x compression
4. **Tune worker count** based on CPU cores
5. **Monitor metrics** via Prometheus client

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from sigmalang.core.encoder import SigmaEncoder

app = FastAPI()
encoder = SigmaEncoder()

@app.post("/encode")
async def encode_text(text: str):
    tree = parser.parse(text)
    encoded = encoder.encode(tree)
    return {"encoded": encoded.hex()}
```

### With Flask

```python
from flask import Flask, request
from sigmalang.core.encoder import SigmaEncoder

app = Flask(__name__)
encoder = SigmaEncoder()

@app.route("/encode", methods=["POST"])
def encode():
    data = request.json
    tree = parser.parse(data["text"])
    encoded = encoder.encode(tree)
    return {"encoded": encoded.hex()}
```

## Next Steps

- Explore [REST API](rest-api.md)
- Learn [CLI](cli.md) commands
- Try [Basic Usage](../getting-started/basic-usage.md) examples
