# ΣLANG Next Steps Master Action Plan v7

## Objective: Maximize Autonomy & Automation

**Created:** 2026-03-23  
**Baseline:** 1862 tests passing, 0 failures, 3 skipped  
**Philosophy:** Every manual step becomes an automated step. Every guard becomes a gate. Every check becomes a hook.

---

## Current State Assessment

| Area                 | Status                            | Gap                                      |
| -------------------- | --------------------------------- | ---------------------------------------- |
| Tests                | ✅ 1862/1862 (100%)               | None                                     |
| CI/CD                | ⚠️ Basic (no coverage gate)       | No fail-on-coverage-drop, no strict lint |
| Pre-commit           | ❌ Missing                        | No `.pre-commit-config.yaml`             |
| Coverage Enforcement | ❌ Missing                        | No minimum threshold enforced            |
| Auto-test-fixer      | ❌ Missing                        | No automated test repair                 |
| Health Monitor       | ⚠️ Script exists, not wired       | `scripts/health_monitor.py` not in CI    |
| PyPI Publish         | ⚠️ Script exists, not wired       | `scripts/publish_pypi.py` not in CI      |
| Docker Publish       | ⚠️ Workflow exists, needs secrets | `.github/workflows/docker.yml`           |
| Security Scanning    | ⚠️ Basic workflow                 | Needs `pip-audit` + `bandit` in CI       |
| Benchmark Regression | ⚠️ CI step exists                 | Not gated (continue-on-error: true)      |
| SDK Generation       | ⚠️ JS done, Java pending          | No auto-generation in CI                 |
| conftest.py          | ⚠️ Basic                          | Missing auto-timeout, fixture isolation  |

---

## Execution Tiers (Autonomous — No Human Input Required)

### TIER A: Automated Quality Gates (Execute Now)

| ID  | Task                                                | Automation Level | Status      |
| --- | --------------------------------------------------- | ---------------- | ----------- |
| A1  | Pre-commit hooks (format + lint + test)             | Full             | 🔧 Building |
| A2  | CI strict mode (coverage ≥85%, lint fail=error)     | Full             | 🔧 Building |
| A3  | conftest.py hardening (timeouts, markers, fixtures) | Full             | 🔧 Building |
| A4  | Makefile new targets (auto-fix, validate, publish)  | Full             | 🔧 Building |
| A5  | Coverage enforcement in pyproject.toml              | Full             | 🔧 Building |

### TIER B: Automated Publishing Pipeline (Execute Now)

| ID  | Task                                    | Automation Level | Status      |
| --- | --------------------------------------- | ---------------- | ----------- |
| B1  | PyPI release workflow (on tag push)     | Full             | 🔧 Building |
| B2  | Docker release workflow (on tag push)   | Full             | 🔧 Building |
| B3  | SDK auto-generation (JS + Python stubs) | Full             | 🔧 Building |

### TIER C: Self-Healing & Monitoring (Execute Now)

| ID  | Task                                     | Automation Level | Status      |
| --- | ---------------------------------------- | ---------------- | ----------- |
| C1  | Auto-test-fixer script                   | Full             | 🔧 Building |
| C2  | Health check integration in CI           | Full             | 🔧 Building |
| C3  | Security scan (pip-audit + bandit) in CI | Full             | 🔧 Building |
| C4  | Benchmark regression gating              | Full             | 🔧 Building |

### TIER D: Future Automation (Queued)

| ID  | Task                                      | Depends On |
| --- | ----------------------------------------- | ---------- |
| D1  | Auto-changelog generation                 | B1         |
| D2  | Dependabot configuration                  | C3         |
| D3  | Auto-merge for passing dependency updates | D2         |
| D4  | Nightly stress test runner                | C2         |
| D5  | Auto-issue creation on test failure       | A2         |

---

## Success Criteria

| Metric                        | Target                        |
| ----------------------------- | ----------------------------- |
| Zero-touch CI pass rate       | 100%                          |
| Pre-commit blocks bad commits | Always                        |
| Coverage never drops below    | 85%                           |
| Publish on tag                | Fully automated               |
| Security scan frequency       | Every PR + weekly             |
| Test suite self-repairs       | Import errors, timeout tuning |
