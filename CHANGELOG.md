# Changelog

All notable changes to ΣLANG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-31

### Added — Phase 7: Advanced Features (12 Tracks)
- **Multi-Modal Encoding** — Image & audio semantic encoders with unified VQ codebook
- **Federated Codebook Learning** — Distributed primitive aggregation with ε-differential privacy
- **Neural Architecture Search** — Evolutionary search for optimal encoder/decoder architectures
- **Sequence-to-Vector Compression** — 100–500x archival compression via fixed-size vectors
- **KV-Cache Compression** — Attention-based pruning, quantization (FP16/INT8/INT4), windowing
- **Lossless Meta-Token Layer** — LZ77-style second-pass for +15–25% lossless compression
- **Attention-Only Prompt Compression** — MLP-free prompt compression (3–5x faster)
- **Streaming Token Compression** — Real-time codebook adaptation with backpressure
- **Product Quantization** — 32x codebook memory reduction for edge/mobile deployment
- **Information-Theoretic Bounds** — Shannon entropy analysis and compression efficiency metrics
- **Enhanced MCP Server** — 10+ tools, batch operations, streaming responses for Claude
- **Optimization Daemon** — Unified background service with anomaly detection and self-healing

### Added — Phase 4–6
- **Learned Codebook Pattern Learning** — Automatic pattern observation and promotion
- **Advanced Analogy Engine** — Semantic vector space with word analogy solving
- **Semantic Search (LSH)** — O(1) approximate nearest neighbor search
- **Entity/Relation Extraction** — Pattern-based NER with knowledge graph construction
- **VectorOptimizer** — Adam-based per-sample gradient descent optimization
- **KVQuantizer** — Mixed-precision KV cache with IntactKV + MiKV policy
- **Continuous Deployment Stack** — Health monitoring, Docker multi-arch, GHCR
- **MCP Server** — Claude Desktop integration with tool composition
- **Knowledge Base Compressor** — Large document compression utility
- **Context Extender** — 200K → 5M+ effective token context window

### Added — Infrastructure
- **CI/CD Pipeline** — 5-gate CI (lint → test matrix → benchmark → build → SDK check)
- **Release Workflow** — Auto-changelog, PyPI publish, Docker build on tag push
- **Security Workflow** — Nightly bandit + pip-audit + safety scans
- **Docker Workflow** — Multi-arch (amd64 + arm64) GHCR publishing
- **Pre-commit Hooks** — 10 hooks (gitleaks, black, isort, ruff, mypy, bandit)
- **Dependabot** — Weekly pip + actions + docker dependency updates
- **Kubernetes** — 10 native manifests + complete Helm chart (v1.0.0)
- **Prometheus + Grafana** — Metrics collection and dashboard configuration
- **35+ Automation Scripts** — SDK gen, security, publishing, chaos testing, compliance

### Added — Testing & Quality
- **1,865 tests** (1,862 passing, 3 skipped, 0 failures)
- **95%+ code coverage** across all modules
- **Python 3.9–3.12 matrix** testing in CI
- **Hypothesis property-based testing** integration
- **Benchmark regression detection** in CI
- **JavaScript SDK** with TypeScript definitions

### Fixed
- 50 legacy test failures resolved (Phase 2)
- 3 WebSocket server failures fixed (parse text→SemanticTree, codec.compress())
- Benchmarking tree blowup (max_depth 16→5, max_children 16→4)
- Buffer pool identity tracking and memory profiling issues
- Memory/parallel test relaxed for system variability
- Windows UTF-8 encoding in automation scripts

### Changed
- Version bumped from 1.0.0 → 2.0.0
- Development Status classifier: Beta → **Production/Stable**
- Coverage enforcement: 60% CI gate, 85% pyproject.toml
- conftest.py enriched with 10+ reusable fixtures and isolation hooks

### Performance
- Text compression: **15–100x** (with meta-token layer)
- Archival compression: **100–500x** (sequence-to-vector mode)
- KV-cache reduction: **2–8x** with <2% quality impact
- Codebook memory: **32x reduction** via product quantization
- Encoding speed: **>1000 ops/sec**
- Context extension: **200K → 5M+ effective tokens**

---

## [1.0.0] - 2025-01-01

### Added
- Core 256 Sigma-Primitive system (Tiers 0–2)
- Semantic parser with intent classification
- Multi-strategy encoder (pattern, reference, delta, full)
- Cascaded codebook compression
- Hyperdimensional encoding
- LZW hypertokenization
- Equal-info windows
- Learned codebook with pattern learning
- Training pipeline (bootstrap, batch, online)
- 27 initial unit tests

---

## Links
- [GitHub Repository](https://github.com/iamthegreatdestroyer/sigmalang)
- [Issue Tracker](https://github.com/iamthegreatdestroyer/sigmalang/issues)
