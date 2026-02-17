# Agent Skills for ΣLANG Development

Production-ready [Agent Skills](https://agentskills.io) optimized for ΣLANG development.

## Available Skills

### 🔒 cipher-security
Advanced cryptography, threat modeling, OWASP/NIST compliance
- **Triggers:** `@CIPHER`, `security audit`, `crypto analysis`
- **Use for:** Cryptographic protocol design, security reviews

### ⚙️ core-systems
Low-level optimization, compiler design, assembly
- **Triggers:** `@CORE`, `optimize`, `assembly`, `performance`
- **Use for:** Performance tuning, memory optimization

### 🏗️ architect-design
Systems architecture, design patterns, scalability
- **Triggers:** `@ARCHITECT`, `architecture`, `design pattern`
- **Use for:** System design, microservices planning

### 📦 sigmalang-compression
Semantic compression via hyperdimensional encoding
- **Triggers:** `ΣLANG`, `compress`, `encode`, `decode`
- **Use for:** Compression algorithm development

## Installation

### Claude Desktop
1. Settings → Skills → Add Custom Skill
2. Navigate to `S:\sigmalang\agent-skills\[skill-name]`
3. Confirm installation

### GitHub Copilot
Copy to `.github/copilot/` directory

## Format

Skills follow [agentskills.io](https://agentskills.io) standard:
- `skill.json` - Metadata, triggers, capabilities
- `instructions.md` - Core implementation guidance

## License
AGPL-3.0 (inherits from ΣLANG core)
