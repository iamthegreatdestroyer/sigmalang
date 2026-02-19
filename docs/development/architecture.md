# Architecture Overview

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                  (CLI / REST API / SDK)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    API Server Layer                          │
│  (FastAPI / Request Routing / Rate Limiting / Auth)         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Core Processing Layer                      │
│  ┌──────────────┬──────────────┬───────────────┐           │
│  │   Parser     │  Encoder     │   Decoder     │           │
│  └──────────────┴──────────────┴───────────────┘           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Optimization Layer                          │
│  ┌───────────┬──────────┬─────────────┬─────────┐          │
│  │  Entropy  │ Primitive│Buffer Pool  │ Caching │          │
│  └───────────┴──────────┴─────────────┴─────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  Infrastructure Layer                        │
│  ┌──────────┬─────────┬──────────┬──────────┐              │
│  │  Redis   │ Storage │ Metrics  │  Logging │              │
│  └──────────┴─────────┴──────────┴──────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Module Organization

```
sigmalang/
├── api/
│   ├── __init__.py
│   ├── interfaces.py         # Protocol definitions
│   ├── server.py             # FastAPI application
│   ├── routes.py             # API endpoints
│   └── middleware.py         # CORS, auth, etc.
│
├── core/
│   ├── __init__.py
│   ├── primitives.py         # Semantic primitives
│   ├── parser.py             # Text parsing
│   ├── encoder.py            # Encoding logic
│   ├── decoder.py            # Decoding logic
│   ├── analogy.py            # Analogy engine
│   ├── optimizations.py      # Optimizations
│   ├── processor.py          # Batch processing
│   ├── config.py             # Configuration
│   └── exceptions.py         # Exception types
│
├── compression/
│   ├── __init__.py
│   ├── entropy.py            # Entropy encoding
│   ├── semantic.py           # Semantic compression
│   ├── kv_cache.py           # KV cache compression
│   └── product_quantization.py
│
├── storage/
│   ├── __init__.py
│   ├── backends.py           # Storage backends
│   ├── cache.py              # Caching layer
│   └── index.py              # Semantic indexing
│
├── cli.py                     # CLI interface
├── __version__.py             # Version info
└── __main__.py                # Entry point
```

## Data Flow

### Encoding Pipeline

```
Input Text
    │
    ▼
┌─────────────────────┐
│   Parsing Stage     │
│  - Tokenization     │
│  - NLP analysis     │
│  - Tree building    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Semantic Stage     │
│  - Primitive lookup │
│  - Relation finding │
│  - Tree traversal   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Optimization Stage  │
│  - Entropy encoding │
│  - Buffer pooling   │
│  - Compression      │
└─────────────────────┘
    │
    ▼
Compressed Output
```

### Request Flow

```
HTTP Request
    │
    ▼
┌─────────────────────────┐
│ API Middleware          │
│ - Auth checking         │
│ - Rate limiting         │
│ - Request validation    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Route Handler           │
│ - Parameter extraction  │
│ - Type conversion       │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Business Logic          │
│ - Core processing       │
│ - Error handling        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ Response Formatter      │
│ - JSON serialization    │
│ - Metadata addition     │
└─────────────────────────┘
    │
    ▼
HTTP Response
```

## Key Components

### Parser (sigmalang/core/parser.py)

Converts text to semantic trees.

```python
class SemanticParser:
    def parse(self, text: str) -> SemanticTree:
        """Parse text into semantic tree"""
        # Tokenization
        tokens = self.tokenize(text)

        # NLP analysis
        pos_tags = self.tag_pos(tokens)
        deps = self.extract_dependencies(tokens)

        # Build tree
        tree = self.build_tree(tokens, pos_tags, deps)
        return tree
```

### Encoder (sigmalang/core/encoder.py)

Converts semantic trees to bytes.

```python
class SigmaEncoder:
    def encode(self, tree: SemanticTree) -> bytes:
        """Encode semantic tree to bytes"""
        # Traverse tree
        primitives = self.extract_primitives(tree)

        # Apply optimizations
        optimized = self.optimize(primitives)

        # Encode to bytes
        data = self.serialize(optimized)
        return data
```

### Buffer Pool (sigmalang/core/optimizations.py)

Manages memory efficiently via object pooling.

```python
class GlyphBufferPool:
    def __init__(self, pool_size=100, buffer_size=4096):
        self._pool = [bytearray(buffer_size) for _ in range(pool_size)]
        self._available_indices = list(range(pool_size))

    def acquire(self) -> bytearray:
        """Get a buffer from pool"""
        if self._available_indices:
            idx = self._available_indices.pop()
            return self._pool[idx]
        return bytearray(self._buffer_size)

    def release(self, buffer: bytearray):
        """Return buffer to pool using identity check"""
        for i, buf in enumerate(self._pool):
            if buf is buffer:  # Use 'is' not '=='
                self._available_indices.append(i)
                break
```

### Caching Layer (sigmalang/storage/cache.py)

Caches encoding results.

```python
class SemanticCache:
    def __init__(self, backend='redis', ttl=3600):
        self.backend = backend
        self.ttl = ttl

    def get(self, key: str) -> Optional[bytes]:
        """Get cached value"""
        return self.backend.get(key)

    def set(self, key: str, value: bytes):
        """Cache value with TTL"""
        self.backend.set(key, value, ex=self.ttl)
```

## Configuration System

### Config Hierarchy

```
1. Hardcoded defaults (core/config.py)
   ↓
2. Environment variables (SIGMALANG_*)
   ↓
3. Config file (.env or config.yaml)
   ↓
4. Runtime parameters (function args)
```

### Example Configuration

```python
from sigmalang.core.config import SigmaConfig

config = SigmaConfig(
    optimization_level="high",
    cache_enabled=True,
    cache_backend="redis",
    cache_ttl=3600,
    num_workers=4,
    max_buffer_size=1024*1024
)

encoder = SigmaEncoder(config=config)
```

## Exception Hierarchy

```
Exception
└── SigmaLangException (base for all ΣLANG errors)
    ├── EncodingError
    │   ├── InvalidTreeError
    │   └── BufferError
    ├── DecodingError
    │   ├── CorruptedDataError
    │   └── VersionMismatchError
    ├── ParsingError
    │   ├── TokenizationError
    │   └── DependencyExtractionError
    ├── CacheError
    │   ├── CacheConnectionError
    │   └── CacheExpirationError
    └── InvalidConfigError
```

## Testing Architecture

### Test Structure

```
tests/
├── test_optimizations.py    # Unit tests
├── test_memory_profiling.py # Performance tests
├── integration/
│   ├── test_cli_commands.py
│   ├── test_generated_*.py
│   └── test_streaming_*.py
└── conftest.py              # Fixtures
```

### Test Fixtures

```python
@pytest.fixture
def encoder():
    return SigmaEncoder()

@pytest.fixture
def sample_text():
    return "Sample text for testing"

@pytest.fixture
def sample_tree(parser, sample_text):
    return parser.parse(sample_text)
```

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Parse | O(n) | Linear in text length |
| Encode | O(n log n) | Tree traversal + sorting |
| Decode | O(n) | Linear in data length |
| Search | O(log n) | Indexed search |

### Space Complexity

| Component | Usage |
|-----------|-------|
| Buffer Pool | O(k·m) | k buffers of size m |
| Cache | Tunable | TTL-based eviction |
| Semantic Tree | O(n) | Proportional to input |

### Optimization Techniques

1. **Buffer Pooling**: Reuse memory allocations
2. **Entropy Encoding**: Variable-length codes
3. **Caching**: Result caching with TTL
4. **Streaming**: Process data in chunks
5. **Indexing**: Fast semantic search

## Extensibility

### Plugin Architecture

Add custom processors:

```python
from sigmalang.api.interfaces import CompressionEngine

class CustomCompressor(CompressionEngine):
    def compress(self, tree: SemanticTree) -> bytes:
        # Custom implementation
        pass

    def decompress(self, data: bytes) -> SemanticTree:
        # Custom implementation
        pass
```

### Custom Storage Backend

```python
from sigmalang.api.interfaces import StorageBackend

class CustomStorage(StorageBackend):
    def get(self, key: str):
        # Custom retrieval
        pass

    def set(self, key: str, value: bytes):
        # Custom storage
        pass
```

## Next Steps

- Read [Testing Guide](testing.md)
- Explore [Contributing](contributing.md)
- Review [API Reference](../api/overview.md)
