# ΣLANG — Semantic Compression via Neural Glyphs

**Philosophy:** _"Compress meaning, not just bytes. Preserve semantics at sub-linear cost."_

## Architecture Overview

ΣLANG achieves **70%+ compression** with **>0.85 semantic fidelity**:

1. **Hyperdimensional Encoding** (Torchhd)
   - 3072-dimensional semantic embeddings
   - Product Quantizer: 192× reduction
   - LSH indexing for O(1) retrieval

2. **LZW Hypertokens**
   - Detect recurring patterns (3+ occurrences)
   - Assign stable reference IDs
   - Replace verbose descriptions

3. **Differential Updates**
   - Extract only new information
   - Count-Min Sketch frequency tracking
   - Store deltas, reconstruct on-demand

## Usage Example

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.decoder import SigmaDecoder

# Encode
encoder = SigmaEncoder(dim=3072, quantization_ratio=192)
compressed = encoder.encode(
    text=input_text,
    preserve_tokens=["AES-256", "ECDH"]
)

# Decode
decoder = SigmaDecoder()
reconstructed = decoder.decode(compressed.glyph_sequence)

# Validate
assert reconstructed.similarity(input_text) > 0.85
```

## Critical Rules
- Never compress: Crypto algorithm names, version numbers, API endpoints
- Compression target: 70% general text, 50% technical content
- Fallback: If compression < 20%, use original
- Always validate semantic similarity > 0.85
