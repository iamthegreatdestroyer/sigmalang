# ΣLANG Ecosystem Integration Guide

**Version:** 2.0.0  
**Layer:** 4 — Storage & Inference  
**Consumers:** Ryzanstein (Ryot), sigma-compress

---

## Python API

### Basic encode/decode

```python
from sigmalang import SemanticParser, SigmaEncoder, SigmaDecoder

parser = SemanticParser()
enc    = SigmaEncoder()
dec    = SigmaDecoder(enc)          # decoder shares the encoder's sigma bank

text    = "SELECT * FROM users WHERE active = true"
tree    = parser.parse(text)        # str → SemanticTree
encoded = enc.encode(tree)          # SemanticTree → bytes
decoded = dec.decode(encoded)       # bytes → SemanticTree
```

`encoded` is compact binary (glyph stream). On second call with the same or similar input, the encoder returns a 4-byte reference instead of full primitives.

### Compression adapter (Ryzanstein protocol)

```python
from sigmalang import SigmaCompressionAdapter, create_ryot_compression_adapter

# High-level: creates adapter with a real ΣLANG engine
adapter = create_ryot_compression_adapter()

# Or explicit construction
from sigmalang import SigmaCompressionAdapter
from sigmalang.api.types import EncodingMode
adapter = SigmaCompressionAdapter(default_mode=EncodingMode.BALANCED)
```

`SigmaCompressionAdapter` implements Ryzanstein's `CompressionEngineProtocol` and produces `RyotSigmaEncodedContext` objects that carry the glyph sequence, original token count, compression ratio, and semantic hash.

### Context stack (multi-turn LLM)

```python
from sigmalang import ContextStack, SigmaEncoder

enc   = SigmaEncoder()
stack = enc.context_stack           # built-in — no separate construction needed

# Encode turn-by-turn; deltas are computed automatically
for turn in conversation:
    tree    = parser.parse(turn)
    encoded = enc.encode(tree)      # references/deltas vs prior turns
    stack.push(tree)                # maintains sliding window
```

---

## CLI Usage

```bash
# Encode
sigma encode "The quick brown fox"
echo "multi-line\ntext" | sigma encode

# Decode (pass the hex/binary output back)
sigma decode <encoded-file>

# Benchmark
sigma benchmark --samples 100 --output benchmark.json

# Train codebook on a corpus
sigma train --corpus ./training_data/ --output codebook.json

# Info / version
sigma --version
sigma info
```

---

## Ryzanstein (Ryot) Integration

Ryot consumes ΣLANG through `SigmaCompressionAdapter` as a drop-in for its `CompressionEngineProtocol`.

**Integration path:** `S:\repos\Layer-4-Storage\Ryot\RYZEN-LLM\docs\SIGMALANG_INTEGRATION.md`

The integration is activated post-MVP (after BitNet inference reaches 15 tok/s). Current hooks in place:

| Ryot Module | Hook | Status |
|-------------|------|--------|
| `src/optimization/memory/kv_cache.h` | `SigmaAnchorMetadata` struct, `register_sigma_anchors()` | Stub (no-op) |
| `src/recycler/recycler_interface.h` | `TokenRecyclerFactory::create(true, config)` | Stub |
| `src/orchestration/` | Context manager ΣLANG branch | Stub |

**Switching on:**
```cpp
// Ryot C++ side
auto recycler = TokenRecyclerFactory::create(/*sigma_enabled=*/true, "config.yaml");
```

**Python side (if calling from Ryot's Python layer):**
```python
from sigmalang import create_ryot_compression_adapter, RyotSigmaEncodedContext

adapter = create_ryot_compression_adapter()
context: RyotSigmaEncodedContext = adapter.compress(token_sequence)
# context.glyph_sequence     — compact bytes
# context.compression_ratio  — actual ratio achieved
# context.semantic_hash      — O(1) content-addressable lookup key
```

---

## sigma-compress Integration

`sigma-compress` (Layer 4, Rust) is a separate compression engine that handles binary/blob data for the NAS OS layer. It does **not** call the sigmalang Python package directly.

The conceptual relationship:
- **sigmalang** → semantic/linguistic compression for LLM token context
- **sigma-compress** → binary/structural compression (LZ4, Zstd, MinHash dedup) for file storage

sigma-compress uses Ryzanstein's `/v1/embeddings` endpoint for semantic deduplication of code blocks — this is a parallel track to sigmalang, not a dependency.

---

## Classes Reference

| Symbol | Module | Purpose |
|--------|--------|---------|
| `SemanticParser` | `sigmalang.core.parser` | NL text → `SemanticTree` |
| `SigmaEncoder` | `sigmalang.core.encoder` | `SemanticTree` → `bytes` |
| `SigmaDecoder` | `sigmalang.core.encoder` | `bytes` → `SemanticTree` |
| `SigmaHashBank` | `sigmalang.core.encoder` | Content-addressable tree store |
| `ContextStack` | `sigmalang.core.encoder` | Sliding window for delta encoding |
| `SigmaCompressionAdapter` | `sigmalang.adapters` | Ryot protocol bridge |
| `RyotSigmaEncodedContext` | `sigmalang.adapters` | Encoded context with metrics |
| `create_ryot_compression_adapter` | `sigmalang` | Factory shortcut |
| `CodebookTrainer` | `sigmalang.training.codebook` | Learn patterns from corpus |
| `LearnedCodebook` | `sigmalang.training.codebook` | Loaded codebook for encode speedup |
