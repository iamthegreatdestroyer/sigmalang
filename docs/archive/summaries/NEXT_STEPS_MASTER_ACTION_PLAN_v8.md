# ΣLANG Next Steps Master Action Plan v8

## Objective: Maximize Autonomy & Automation

**Created:** 2026-03-24
**Supersedes:** v7 (2026-03-23) — corrected accuracy of infrastructure assessment
**Baseline:** 1862 tests passing, 0 failures, 3 skipped (155.55s)
**Philosophy:** Every manual step becomes an automated step. Every guard becomes a gate. Every check becomes a hook.

---

## Accurate Infrastructure Assessment (v8 — Verified March 24, 2026)

### What Already EXISTS (Verified on Disk)

| Component         | File                                    | Status                                                                                          |
| ----------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Pre-commit hooks  | `.pre-commit-config.yaml`               | ✅ 10 hooks (gitleaks, black, isort, ruff, mypy, bandit, file hygiene, fast-test-gate, secrets) |
| CI pipeline       | `.github/workflows/ci.yml`              | ✅ 4 gates (lint, test matrix, benchmark, build)                                                |
| Release workflow  | `.github/workflows/release.yml`         | ✅ test → build → GitHub Release → PyPI → Docker                                                |
| Security workflow | `.github/workflows/security-review.yml` | ✅ bandit + pip-audit + safety + nightly health                                                 |
| Docker workflow   | `.github/workflows/docker.yml`          | ✅ Multi-arch GHCR (amd64 + arm64)                                                              |
| Dependabot        | `.github/dependabot.yml`                | ✅ pip + actions + docker (weekly, grouped)                                                     |
| Makefile          | `Makefile`                              | ✅ 25+ targets (test-strict, auto-fix, security-scan, validate, etc.)                           |
| Coverage config   | `pyproject.toml`                        | ✅ `fail_under = 85` + ruff + bandit + pytest                                                   |
| Auto-test-fixer   | `scripts/auto_fix_tests.py`             | ✅ Diagnose + auto-fix modes                                                                    |
| Health monitor    | `scripts/health_monitor.py`             | ✅ Quick + full modes                                                                           |
| 40+ scripts       | `scripts/`                              | ✅ SDK gen, security, publishing, chaos, compliance                                             |
| Test suite        | `tests/` (48 files)                     | ✅ 1862 passing, 0 failed                                                                       |

### Remaining Gaps (The ACTUAL Work)

| ID  | Gap                                                                                                                                      | Impact                                      | Priority                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ---------------------------------------- | ---------------------------------- | ------ |
| G1  | **conftest.py too basic** — no global timeout fixture, no isolation hooks, no performance baselines, no reusable encoder/parser fixtures | Tests fragile to environment drift          | HIGH                                     |
| G2  | **mypy non-blocking in CI** — `continue-on-error: true` means type errors never fail the build                                           | Type regressions go undetected              | HIGH                                     |
| G3  | **Benchmark regression not gated** — uses `                                                                                              |                                             | true`, results stored but never compared | Performance regressions undetected | MEDIUM |
| G4  | **pyproject.toml missing tool configs** — no `[tool.mypy]`, no `[tool.isort]` sections                                                   | Tools use defaults, not project conventions | MEDIUM                                   |
| G5  | **No auto-changelog generation** — release workflow has placeholder body                                                                 | Release notes are empty                     | MEDIUM                                   |
| G6  | **Health monitor not wired to nightly** — script exists but security-review.yml doesn't call it on success                               | Health issues only caught on failure        | MEDIUM                                   |
| G7  | **SDK auto-generation not in CI** — JS SDK created manually, no CI job                                                                   | SDKs drift from API changes                 | LOW                                      |
| G8  | **No Makefile `changelog` target** — manual process                                                                                      | Release prep is manual                      | LOW                                      |
| G9  | **pip-audit non-blocking in CI** — `continue-on-error: true`                                                                             | Known CVE deps can merge                    | LOW                                      |

---

## Execution Plan (Fully Autonomous — No Human Input Required)

### TIER 1: Hardened Test Infrastructure (Execute Immediately)

| Task                                                                             | Files Modified | Automation Gain                         |
| -------------------------------------------------------------------------------- | -------------- | --------------------------------------- |
| **T1.1** Enrich conftest.py with global fixtures (encoder, parser, codec, timer) | `conftest.py`  | Eliminates boilerplate in 48 test files |
| **T1.2** Add fixture isolation hook (warns on leftover state)                    | `conftest.py`  | Catches test pollution automatically    |
| **T1.3** Add performance baseline fixtures                                       | `conftest.py`  | Enables benchmark regression in tests   |

### TIER 2: Strict CI Quality Gates (Execute Immediately)

| Task                                                            | Files Modified        | Automation Gain                     |
| --------------------------------------------------------------- | --------------------- | ----------------------------------- |
| **T2.1** Remove `continue-on-error: true` from mypy step        | `ci.yml`              | Type errors now fail the build      |
| **T2.2** Gate benchmark regression (compare to baseline)        | `ci.yml`              | Performance regressions detected    |
| **T2.3** Add `[tool.mypy]` and `[tool.isort]` to pyproject.toml | `pyproject.toml`      | Consistent tool behavior everywhere |
| **T2.4** Wire health_monitor to nightly success path            | `security-review.yml` | Proactive health monitoring         |

### TIER 3: Release Automation (Execute Immediately)

| Task                                                            | Files Modified | Automation Gain                           |
| --------------------------------------------------------------- | -------------- | ----------------------------------------- |
| **T3.1** Add auto-changelog generation step to release workflow | `release.yml`  | Release notes auto-generated from commits |
| **T3.2** Add Makefile `changelog` target                        | `Makefile`     | One-command changelog preview             |
| **T3.3** Add SDK regeneration job to CI (on main push)          | `ci.yml`       | SDKs always in sync with API              |

### TIER 4: Future Automation (Queued for Next Session)

| ID  | Task                                           | Dependency                            |
| --- | ---------------------------------------------- | ------------------------------------- |
| F1  | Auto-merge passing Dependabot PRs              | Requires branch protection rules      |
| F2  | Nightly stress test (chaos_test.py)            | Requires dedicated runner or schedule |
| F3  | Auto-issue creation on persistent test failure | Requires GitHub API token             |
| F4  | Performance dashboard (Grafana/GitHub Pages)   | Requires benchmark storage            |
| F5  | Auto-rollback on release failure               | Requires canary deploy setup          |

---

## Success Criteria

| Metric             | Before v8                   | After v8            | Verification              |
| ------------------ | --------------------------- | ------------------- | ------------------------- |
| CI strictness      | mypy/pip-audit non-blocking | All gates blocking  | Push bad type → CI fails  |
| conftest fixtures  | 3 basic                     | 10+ reusable        | Tests use shared fixtures |
| Release automation | Manual changelog            | Auto-generated      | Tag push → full release   |
| Health monitoring  | Only on failure             | Always runs nightly | Check workflow runs       |
| Benchmark gating   | Results stored only         | Regression fails CI | Push slow code → CI fails |
| Tool consistency   | Defaults everywhere         | Explicit configs    | mypy/isort match CI       |

---

## Execution Order

```
T1.1 → T1.2 → T1.3  (conftest.py — single file, do together)
     ↓
T2.1 → T2.2 → T2.3 → T2.4  (CI + pyproject — parallel-safe)
     ↓
T3.1 → T3.2 → T3.3  (release automation)
     ↓
VALIDATE (full test suite)
     ↓
UPDATE automation_state.json
```
