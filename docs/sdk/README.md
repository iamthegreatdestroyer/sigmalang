# ΣLANG Python SDK Guide

Complete guide to using the ΣLANG Python SDK for semantic encoding, analogies, and search.

## Installation

```bash
# Install from PyPI
pip install sigmalang

# Install with all dependencies
pip install sigmalang[all]

# Development installation
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang
pip install -e ".[dev]"
```

## Quick Start

```python
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
from sigmalang.core.semantic_analogy_engine import SemanticAnalogyEngine
from sigmalang.core.semantic_search import SemanticSearchEngine

# Initialize encoder
encoder = SigmaEncoder()

# Encode text to vector
text = "Create a Python function that sorts a list"
result = encoder.encode(text)

print(f"Input: {text}")
print(f"Vector shape: {result.vector.shape}")
print(f"Compression ratio: {result.compression_ratio:.1f}x")
```

---

## Core Components

### SigmaEncoder

The encoder transforms natural language text into high-dimensional semantic vectors.

```python
from sigmalang.core.encoder import SigmaEncoder

encoder = SigmaEncoder()

# Basic encoding
result = encoder.encode("Hello, world!")

# Access vector
vector = result.vector  # numpy array

# Access metadata
print(f"Original size: {result.input_size} bytes")
print(f"Compressed size: {result.output_size} bytes")
print(f"Compression ratio: {result.compression_ratio:.2f}x")
print(f"Encoding type: {result.encoding_type}")

# Batch encoding
texts = [
    "Machine learning fundamentals",
    "Deep neural networks",
    "Natural language processing"
]
results = encoder.encode_batch(texts)

for text, result in zip(texts, results):
    print(f"{text[:30]}... -> {result.compression_ratio:.1f}x compression")
```

### SigmaDecoder

The decoder reconstructs text from semantic vectors.

```python
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder

encoder = SigmaEncoder()
decoder = SigmaDecoder(encoder)

# Encode then decode
original = "The quick brown fox jumps over the lazy dog"
encoded = encoder.encode(original)
decoded = decoder.decode(encoded.vector)

print(f"Original: {original}")
print(f"Decoded: {decoded.text}")
print(f"Similarity: {decoded.similarity:.2%}")
```

---

## Analogy Engine

ΣLANG includes a powerful semantic analogy engine that can solve and explain analogical relationships.

### Solving Analogies

```python
from sigmalang.core.semantic_analogy_engine import SemanticAnalogyEngine

engine = SemanticAnalogyEngine()

# Register candidate words for analogy resolution
candidates = [
    "woman", "man", "king", "queen", "prince", "princess",
    "boy", "girl", "father", "mother", "son", "daughter"
]
engine.register_candidates(candidates)

# Solve: king:queen :: man:?
result = engine.solve_analogy("king", "queen", "man", top_k=3)

print("king : queen :: man : ?")
for solution in result.solutions:
    print(f"  → {solution.answer} (confidence: {solution.confidence:.2%})")
    print(f"    Relation: {solution.relation}")

# Output:
# king : queen :: man : ?
#   → woman (confidence: 94.2%)
#     Relation: gender_counterpart
#   → female (confidence: 78.1%)
#     Relation: gender_attribute
```

### Explaining Analogies

```python
# Explain an analogy relationship
explanation = engine.explain_analogy("Paris", "France", "Tokyo", "Japan")

print(f"Explanation: {explanation.explanation}")
print(f"Similarity score: {explanation.similarity_score:.2%}")
print(f"Relation A→B: {explanation.relation_ab}")
print(f"Relation C→D: {explanation.relation_cd}")
```

### Advanced Analogy Types

```python
from sigmalang.core.advanced_analogy_patterns import AdvancedAnalogyEngine
from sigmalang.core.analogy_composition import AnalogyComposer

# Compositional analogies
composer = AnalogyComposer()
result = composer.compose_analogy(
    base_analogies=[
        ("hot", "cold"),
        ("big", "small"),
    ],
    query="fast"
)
print(f"fast → {result.answer}")  # slow

# Cross-modal analogies
from sigmalang.core.cross_modal_analogies import CrossModalAnalogyEngine

cross_modal = CrossModalAnalogyEngine()
result = cross_modal.solve(
    source_modality="text",
    target_modality="concept",
    query="happiness"
)
```

---

## Semantic Search

Perform intelligent semantic search across document corpora.

```python
from sigmalang.core.semantic_search import SemanticSearchEngine

search_engine = SemanticSearchEngine()

# Prepare corpus
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing enables computers to understand text",
    "Computer vision allows machines to interpret images",
    "Reinforcement learning trains agents through reward signals"
]

# Index the corpus
search_engine.index_corpus(documents)

# Search
query = "AI systems that understand human language"
results = search_engine.search(query, top_k=3)

for result in results:
    print(f"Score: {result.score:.3f} | {result.text}")

# Output:
# Score: 0.891 | Natural language processing enables computers to understand text
# Score: 0.743 | Machine learning is a subset of artificial intelligence
# Score: 0.521 | Deep learning uses neural networks with many layers
```

### Hybrid Search

Combine semantic and keyword-based search:

```python
results = search_engine.search(
    query="neural network training",
    mode="hybrid",  # semantic + keyword
    semantic_weight=0.7,
    top_k=5
)
```

---

## Entity Extraction

Extract named entities and relationships from text.

```python
from sigmalang.core.entity_relation_extraction import EntityRelationExtractor

extractor = EntityRelationExtractor()

text = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
in April 1976. The company is headquartered in Cupertino, California.
"""

result = extractor.extract(text)

# Print entities
print("Entities:")
for entity in result.entities:
    print(f"  {entity.text} ({entity.type}) - confidence: {entity.confidence:.2%}")

# Print relations
print("\nRelations:")
for relation in result.relations:
    print(f"  {relation.source} --[{relation.type}]--> {relation.target}")

# Output:
# Entities:
#   Apple Inc. (ORGANIZATION) - confidence: 98.2%
#   Steve Jobs (PERSON) - confidence: 97.1%
#   Steve Wozniak (PERSON) - confidence: 96.8%
#   Ronald Wayne (PERSON) - confidence: 95.4%
#   April 1976 (DATE) - confidence: 99.1%
#   Cupertino, California (LOCATION) - confidence: 97.5%
#
# Relations:
#   Steve Jobs --[FOUNDED]--> Apple Inc.
#   Apple Inc. --[HEADQUARTERED_IN]--> Cupertino, California
```

---

## Text Understanding

Analyze text for deeper semantic understanding.

```python
from sigmalang.core.text_understanding import TextUnderstandingEngine

engine = TextUnderstandingEngine()

text = "The stock market crashed yesterday, causing widespread panic among investors."

analysis = engine.analyze(text)

print(f"Sentiment: {analysis.sentiment}")
print(f"Topics: {analysis.topics}")
print(f"Intent: {analysis.intent}")
print(f"Entities: {[e.text for e in analysis.entities]}")
print(f"Key phrases: {analysis.key_phrases}")
```

---

## Transformer Embeddings

Generate state-of-the-art embeddings using transformer models.

```python
from sigmalang.core.transformer_embeddings import TransformerEmbeddings

embeddings = TransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Single text embedding
text = "Machine learning is fascinating"
vector = embeddings.embed(text)
print(f"Embedding shape: {vector.shape}")  # (384,)

# Batch embeddings
texts = [
    "Deep learning fundamentals",
    "Neural network architecture",
    "Backpropagation algorithm"
]
vectors = embeddings.embed_batch(texts)
print(f"Batch shape: {vectors.shape}")  # (3, 384)

# Similarity calculation
from sigmalang.core.transformer_embeddings import cosine_similarity

sim = cosine_similarity(vectors[0], vectors[1])
print(f"Similarity between texts 1 and 2: {sim:.3f}")
```

---

## Unified Pipeline

Use the unified pipeline for end-to-end processing.

```python
from sigmalang.core.unified_pipeline import UnifiedPipeline, PipelineConfig

# Create pipeline with custom configuration
config = PipelineConfig(
    enable_caching=True,
    enable_analytics=True,
    embedding_model="all-MiniLM-L6-v2",
    max_tokens=1000
)

pipeline = UnifiedPipeline(config)

# Full analysis
text = "Create a Python function to sort a list in descending order"

result = pipeline.process(text)

print(f"Encoding: {result.encoding.compression_ratio:.1f}x")
print(f"Intent: {result.analysis.intent}")
print(f"Entities: {result.entities}")
print(f"Processing time: {result.processing_time_ms:.1f}ms")
```

---

## CLI Usage

ΣLANG includes a powerful command-line interface.

### Encoding

```bash
# Encode text
sigmalang encode "Hello, ΣLANG!"

# Encode from file
sigmalang encode --input document.txt --output encoded.npy

# Batch encoding
sigmalang batch encode corpus.txt --output vectors.npy
```

### Analogies

```bash
# Solve analogy
sigmalang analogy solve "king:queen::man:?"

# Explain analogy
sigmalang analogy explain "Paris:France::Tokyo:Japan"
```

### Search

```bash
# Search in corpus
sigmalang search "machine learning" --corpus documents.txt --top-k 5
```

### Server

```bash
# Start API server
sigmalang serve --host 0.0.0.0 --port 8000 --workers 4

# Development mode with hot reload
sigmalang serve --reload --debug
```

---

## Configuration

### Environment Variables

```bash
# Server configuration
export SIGMALANG_HOST="0.0.0.0"
export SIGMALANG_PORT="8000"
export SIGMALANG_WORKERS="4"
export SIGMALANG_DEBUG="false"

# Model configuration
export SIGMALANG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export SIGMALANG_VECTOR_DIMENSIONS="512"

# Cache configuration
export SIGMALANG_CACHE_ENABLED="true"
export SIGMALANG_REDIS_URL="redis://localhost:6379/0"

# Rate limiting
export SIGMALANG_RATE_LIMIT_ENABLED="true"
export SIGMALANG_RATE_LIMIT_RPM="60"
```

### Programmatic Configuration

```python
from sigmalang.core.config import get_config, SigmalangConfig

# Get current configuration
config = get_config()
print(f"Debug mode: {config.server.debug}")
print(f"Workers: {config.server.workers}")

# Create custom configuration
custom_config = SigmalangConfig()
custom_config.server.workers = 8
custom_config.cache.enabled = True
```

---

## Performance Tips

### 1. Use Batch Processing

```python
# ❌ Slow: Individual encoding
for text in texts:
    result = encoder.encode(text)

# ✅ Fast: Batch encoding
results = encoder.encode_batch(texts)
```

### 2. Enable Caching

```python
from sigmalang.core.unified_pipeline import UnifiedPipeline, PipelineConfig

config = PipelineConfig(enable_caching=True)
pipeline = UnifiedPipeline(config)
```

### 3. Use Appropriate Models

```python
# Faster, smaller embeddings (recommended for most use cases)
embeddings = TransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Higher quality, slower (for production quality)
embeddings = TransformerEmbeddings(model_name="all-mpnet-base-v2")
```

### 4. Stream Large Datasets

```python
from sigmalang.core.streaming_processor import StreamingProcessor

processor = StreamingProcessor(chunk_size=1000)

# Process large file in chunks
for result_chunk in processor.process_file("large_corpus.txt"):
    # Handle chunk
    pass
```

---

## Error Handling

```python
from sigmalang.core.api_models import ErrorResponse

try:
    result = encoder.encode(text)
except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Or use structured logging
from sigmalang.core.monitoring import StructuredLogger

logger = StructuredLogger("my_app")
logger.info("Processing started", count=100, mode="batch")
```

---

## Next Steps

- [API Reference](../api/README.md) - Complete REST API documentation
- [Deployment Guide](../deployment/README.md) - Docker and Kubernetes deployment
- [Examples](../../examples/) - Runnable code examples
