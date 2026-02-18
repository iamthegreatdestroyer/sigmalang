"""
Batch update all Copilot agents with SigmaLang integration section.
Phase 5 Task 5.4 - Adds standardized ΣLANG capability awareness to all agents.
"""

import re
from pathlib import Path

AGENTS_DIR = Path(__file__).parent.parent / ".github" / "agents"

# Agent-specific SigmaLang role mappings
AGENT_ROLES = {
    "AEGIS": ("Security & Compliance", "security scanning of encoded payloads, compressed audit trails"),
    "APEX": ("Systems Architecture", "high-level architecture for compression pipelines, service design"),
    "ARBITER": ("Decision Making", "evaluate compression strategy trade-offs, codebook optimization decisions"),
    "ARCHITECT": ("Software Architecture", "design patterns for encoder/decoder services, API architecture"),
    "ATLAS": ("Knowledge Management", "compressed knowledge base navigation, semantic index design"),
    "AXIOM": ("Mathematical Foundations", "information theory for compression bounds, entropy analysis"),
    "BRIDGE": ("Integration & APIs", "MCP server integration, SDK generation, API endpoint design"),
    "CANVAS": ("UI/UX Design", "Grafana dashboard design, monitoring visualization, status pages"),
    "CIPHER": ("Cryptography", "encrypted compression, secure codebook storage, hash verification"),
    "COMMUNICATOR": ("Communication", "documentation of compression APIs, user guides, release notes"),
    "CRYPTO": ("Blockchain & Crypto", "immutable compression audit logs, distributed codebook consensus"),
    "ECLIPSE": ("Testing & QA", "chaos testing, property-based tests, compression fuzz testing"),
    "FLUX": ("Event-Driven Systems", "streaming compression, real-time encoding events, pipeline triggers"),
    "FORGE": ("DevOps & CI/CD", "Docker builds, Helm deployments, GitHub Actions for compression benchmarks"),
    "FORTRESS": ("Infrastructure Security", "secure deployment of compression services, access control"),
    "GENESIS": ("Project Scaffolding", "SigmaLang SDK project templates, starter kits"),
    "HELIX": ("Bioinformatics", "sequence compression patterns, genomic data encoding"),
    "LATTICE": ("Distributed Systems", "distributed codebook synchronization, sharded encoding"),
    "LEDGER": ("Financial Systems", "compressed transaction logs, efficient financial data encoding"),
    "LINGUA": ("NLP & Languages", "semantic tokenization, morphological analysis, stemming for primitives"),
    "MENTOR": ("Teaching & Guidance", "SigmaLang tutorials, compression concept explanations"),
    "MORPH": ("Data Transformation", "data format conversion, codec transformations, ETL compression"),
    "NEXUS": ("API Gateway", "compressed API responses, content negotiation, encoding middleware"),
    "OMNISCIENT": ("Meta-Orchestrator", "coordinate all agents for SigmaLang optimization, meta-learning"),
    "ORACLE": ("Prediction & Analytics", "compression ratio prediction, codebook usage forecasting"),
    "ORBIT": ("Cloud & Infrastructure", "Kubernetes deployment, auto-scaling compression services"),
    "PHANTOM": ("Reverse Engineering", "analyze compressed formats, decode unknown encodings"),
    "PHOTON": ("Real-Time Systems", "low-latency encoding, real-time compression streaming"),
    "PRISM": ("Data Analysis", "compression statistics analysis, benchmark trend detection"),
    "PULSE": ("Monitoring & Alerting", "Prometheus metrics, Grafana alerts, health check monitoring"),
    "QUANTUM": ("Quantum Computing", "quantum-inspired compression, superposition-based codebook search"),
    "SCRIBE": ("Documentation", "API documentation, OpenAPI specs, MkDocs site generation"),
    "SENTRY": ("Error Tracking", "compression error monitoring, decode failure analysis"),
    "STREAM": ("Data Streaming", "streaming encoder/decoder, chunked compression, backpressure"),
    "SYNAPSE": ("Workflow Orchestration", "compression pipeline orchestration, multi-stage encoding"),
    "TOKEN_RECYCLER": ("Token Optimization", "token recycling integration, context compression coordination"),
    "VANGUARD": ("Innovation Research", "research paper integration, new compression technique evaluation"),
    "VELOCITY": ("Performance", "encoder profiling, compression throughput optimization, benchmarking"),
    "VERTEX": ("Graph & Network", "semantic graph compression, knowledge graph encoding"),
}

SIGMALANG_SECTION = """
## SigmaLang Integration

### Role in ΣLANG Ecosystem

**Domain Contribution:** {role_description}

**ΣLANG-Specific Tasks:**
- {task_description}
- Leverage 256 Sigma-Primitive system (Tier 0: existential, Tier 1: domain, Tier 2: learned)
- Support compression pipeline: Parse → Encode → Compress → Store → Retrieve

### Key SigmaLang Files

| Component | Path |
|-----------|------|
| Core Encoder | `sigmalang/core/encoder.py` |
| Semantic Parser | `sigmalang/core/parser.py` |
| Primitives (256) | `sigmalang/core/primitives.py` |
| HD Encoder | `sigmalang/core/hyperdimensional_encoder.py` |
| LZW Hypertokens | `sigmalang/core/lzw_hypertoken.py` |
| Cascaded Codebook | `sigmalang/core/cascaded_codebook.py` |
| Equal-Info Windows | `sigmalang/core/equal_info_windows.py` |
| Enhanced Tokenizer | `sigmalang/core/enhanced_semantic_tokenizer.py` |
| Online Learner | `sigmalang/training/online_learner.py` |
| A/B Tester | `sigmalang/training/ab_tester.py` |
| Adaptive Pruner | `sigmalang/training/adaptive_pruner.py` |
| MCP Server | `integrations/claude_mcp_server.py` |
| KB Compressor | `tools/knowledge_base_compressor.py` |
| Context Extender | `tools/context_extender.py` |

### Compression Targets

- **Ratio:** 15-75x (text-dependent)
- **Primitive Reuse:** 85%+
- **Context Extension:** 200K → 2M+ effective tokens
- **Encoding Speed:** >1000 ops/sec
"""


def update_agent(agent_path: Path) -> bool:
    """Add SigmaLang section to an agent file if not already present."""
    content = agent_path.read_text(encoding="utf-8")

    # Skip if already has SigmaLang integration
    if "## SigmaLang Integration" in content or "ΣLANG Compression-Aware" in content or "ΣLANG Encoding Patterns" in content or "ΣLANG Hyperdimensional" in content:
        return False

    # Get agent name from filename
    agent_name = agent_path.stem.replace(".agent", "")

    if agent_name not in AGENT_ROLES:
        return False

    role_name, task_desc = AGENT_ROLES[agent_name]

    section = SIGMALANG_SECTION.format(
        role_description=f"{role_name} - {task_desc}",
        task_description=task_desc.capitalize()
    )

    # Find the right place to insert (before Token Recycling section or at end)
    insert_markers = [
        "# Token Recycling Integration Template",
        "## Token Recycling & Context Compression",
        "## VS Code 1.109 Integration",
    ]

    inserted = False
    for marker in insert_markers:
        if marker in content:
            content = content.replace(marker, section + "\n" + marker)
            inserted = True
            break

    if not inserted:
        # Append at end
        content = content.rstrip() + "\n" + section

    agent_path.write_text(content, encoding="utf-8")
    return True


def main():
    """Update all agent files."""
    agent_files = sorted(AGENTS_DIR.glob("*.agent.md"))

    print("=" * 60)
    print("SigmaLang Agent Enhancement - Phase 5 Task 5.4")
    print("=" * 60)

    updated = 0
    skipped = 0

    for agent_path in agent_files:
        agent_name = agent_path.stem.replace(".agent", "")

        if update_agent(agent_path):
            print(f"  [PASS] Updated: {agent_name}")
            updated += 1
        else:
            print(f"  [SKIP] Already updated or not mapped: {agent_name}")
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"[PASS] Updated {updated} agents, skipped {skipped}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
