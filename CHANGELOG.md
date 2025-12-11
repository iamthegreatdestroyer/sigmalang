# Changelog

All notable changes to ΣLANG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Improved compression ratios for natural language
- Decoder round-trip fixes
- Performance benchmarking suite
- Integration with Ryot LLM

## [1.0.0] - 2025-01-XX

### Added
- **Core Primitives System**
  - 256 Σ-primitives across 3 tiers (Existential, Domain, Learned)
  - Semantic tree structures for meaning representation
  - Glyph encoding/decoding with CRC-16 checksums
  - Primitive registry for lookup and validation

- **Semantic Parser**
  - Natural language to semantic tree conversion
  - Intent classification (Query, Command, Code Request, etc.)
  - Entity and relation extraction
  - Pattern matching for code, queries, and modifications

- **ΣLANG Encoder**
  - Multi-strategy encoding (pattern, reference, delta, full)
  - Sigma Hash Bank with LSH indexing
  - Context-aware delta encoding
  - LRU caching for hot paths

- **Learned Codebook**
  - Dynamic pattern learning during usage
  - Pattern signature extraction
  - Automatic promotion of frequent patterns
  - Persistent storage (JSON serialization)

- **Training Pipeline**
  - Bootstrap training with built-in examples
  - Batch training from JSONL corpora
  - Online/interactive training mode
  - Training metrics and reporting

- **Ryot LLM Integration**
  - `SigmaLangPipeline` for end-to-end encoding
  - `RyotInputProcessor` for input handling
  - `RyotOutputProcessor` for output decoding
  - Real-time compression metrics

- **Testing & Documentation**
  - 27 unit tests covering all major components
  - Compression demonstration script
  - Comprehensive README
  - API documentation in docstrings

### Performance
- Achieved 3-4x compression on first use
- Delta encoding provides additional 1.5-2x on repeated patterns
- Pattern matches achieve near O(1) encoding

### Known Issues
- Decoder round-trip has edge cases with complex payloads
- Some inputs achieve <1x compression (being addressed)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-01 | Initial release with full compression pipeline |

---

## Upgrade Guide

### From Pre-release to 1.0.0
This is the initial release. No migration needed.

---

## Links
- [GitHub Repository](https://github.com/YOUR_USERNAME/sigmalang)
- [Issue Tracker](https://github.com/YOUR_USERNAME/sigmalang/issues)
- [Discussions](https://github.com/YOUR_USERNAME/sigmalang/discussions)
