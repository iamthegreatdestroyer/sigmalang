# SigmaLang Port Assignments (26000 Series)

**Purpose:** Port configuration for SigmaLang (ΣLANG) project.

**Port Series:** 26000-26999 (exclusive range)

**Last Updated:** December 18, 2025

---

## 🎯 Port Allocation Summary

| Port Range | Category | Description |
|------------|----------|-------------|
| 26000-26099 | Application | IDE, LSP, Playground |
| 26080-26099 | API | Compiler, Runtime, Debug |
| 26500-26599 | Cache | Redis |
| 26900-26999 | Observability | Prometheus, Grafana |

---

## 💻 Application & API Tier (26000-26099)

| Port | Service | Description | Config |
|------|---------|-------------|--------|
| **26080** | Compiler API | REST compilation service | `docker-compose.yml` |

---

## 🔄 Cache & Messaging (26500-26599)

| Port | Service | Internal | Description | Config |
|------|---------|----------|-------------|--------|
| **26500** | Redis | 6379 | Cache and pub/sub | `docker-compose.yml` |

---

## 📈 Observability (26900-26999)

| Port | Service | Internal | Description | Config |
|------|---------|----------|-------------|--------|
| **26900** | Prometheus | 9090 | Metrics collection | `docker-compose.yml` |
| **26910** | Grafana | 3000 | Dashboards | `docker-compose.yml` |

---

## 🌐 Quick Access

| Service | URL |
|---------|-----|
| Compiler API | http://localhost:26080 |
| Grafana | http://localhost:26910 |
| Redis | localhost:26500 |

---

## 🔗 Cross-Project Reference

For complete port allocations across all projects in the ecosystem, see the **MASTER_PORT_ASSIGNMENTS.md** in the NEURECTOMY project:
- **DOPPELGANGER-STUDIO:** 10000-10999
- **NEURECTOMY:** 16000-16999
- **SigmaLang:** 26000-26999 ✓
- **SigmaVault:** 36000-36999
- **Ryot LLM:** 46000-46999

---

## 🚀 Getting Started

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f sigmalang

# Stop services
docker-compose down
```

---

**Version:** 1.0
**Status:** Active and maintained
