# ΣLANG Next Steps Master Action Plan v9

## Objective: Maximize Autonomy & Automation

**Created:** 2026-03-31  
**Supersedes:** v8 (2026-03-24)  
**Baseline:** 1,865 collected, 1,862 passed, 0 failed, 3 skipped (~166s)  
**Tag Status:** `v1.0.0` → `3da6960` (72 commits stale — HEAD is `6660404`)  
**Philosophy:** Every manual step becomes an automated step. Ship what's built.

---

## Priority Matrix

```
                    ║  HIGH IMPACT
                    ║
  TIER 1            ║          TIER 2
  Retag + Publish   ║          CI Strictness
  (unblocks users)  ║          (prevents regressions)
                    ║
 LOW EFFORT ════════╬═════════ HIGH EFFORT
                    ║
  TIER 3            ║          TIER 4
  Workspace Cleanup ║          Advanced Automation
  (reduces noise)   ║          (future-proofing)
                    ║
                    ║  LOW IMPACT
```

---

## TIER 1: Release & Publication (Execute Immediately)

> **Goal:** Make SigmaLang installable and discoverable. Unblocks all adoption.

| Task                                            | Action                                                                | Files            | Verification                                                 |
| ----------------------------------------------- | --------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| **T1.1** Bump version to `2.0.0`                | Update `version = "2.0.0"` in pyproject.toml                          | `pyproject.toml` | `python -c "import sigmalang; print(sigmalang.__version__)"` |
| **T1.2** Update classifier to Production/Stable | Change `"Development Status :: 4 - Beta"` → `"5 - Production/Stable"` | `pyproject.toml` | grep classifier                                              |
| **T1.3** Clean CHANGELOG.md                     | Generate from git log since v1.0.0                                    | `CHANGELOG.md`   | File reviewed                                                |
| **T1.4** Build & verify package                 | `python -m build && twine check dist/*`                               | dist/            | Exit code 0                                                  |
| **T1.5** Delete stale v1.0.0 tag                | `git tag -d v1.0.0 && git push origin :refs/tags/v1.0.0`              | Git              | Tag removed                                                  |
| **T1.6** Create v2.0.0 tag on HEAD              | `git tag -a v2.0.0 -m "ΣLANG v2.0.0 — Production Release"`            | Git              | Tag exists at HEAD                                           |
| **T1.7** Push tag to trigger release            | `git push origin v2.0.0`                                              | CI/release.yml   | GitHub Release created                                       |

**Blocked on:** PyPI API token secret (`PYPI_API_TOKEN`) must be set in GitHub repo Settings → Secrets → Actions. If not set, the release workflow will create the GitHub Release but skip PyPI publish.

---

## TIER 2: CI Strictness & Quality Gates (Execute Immediately)

> **Goal:** Prevent regressions from merging. Every gate must block on failure.

| Task                                             | Action                                                 | Files                 | Verification                   |
| ------------------------------------------------ | ------------------------------------------------------ | --------------------- | ------------------------------ |
| **T2.1** Make mypy blocking                      | Remove `continue-on-error: true` from mypy step        | `ci.yml`              | Push type error → CI fails     |
| **T2.2** Make bandit blocking                    | Remove `continue-on-error: true` from bandit step      | `ci.yml`              | Push insecure code → CI fails  |
| **T2.3** Make pip-audit blocking                 | Remove `continue-on-error: true` from audit step       | `ci.yml`              | CVE dep → CI fails             |
| **T2.4** Make integration tests blocking         | Remove `continue-on-error: true` from integration step | `ci.yml`              | Integration failure → CI fails |
| **T2.5** Raise CI coverage to 85%                | Change `MIN_COVERAGE: 60` → `MIN_COVERAGE: 85`         | `ci.yml`              | Coverage drop → CI fails       |
| **T2.6** Fix existing mypy/bandit/audit warnings | Fix or suppress pre-existing issues                    | `sigmalang/core/*.py` | CI passes clean                |

**Note:** T2.6 must happen BEFORE T2.1–T2.4, otherwise CI breaks on push.

---

## TIER 3: Workspace Cleanup (Execute Immediately)

> **Goal:** Remove clutter, consolidate docs, clean root directory.

| Task                                         | Action                                                     | Files        |
| -------------------------------------------- | ---------------------------------------------------------- | ------------ |
| **T3.1** Remove temp files                   | Delete `_*.txt` files (20+ build/commit artifacts)         | Root         |
| **T3.2** Remove stale exec summaries         | Keep only v9, archive or delete v5/v6/v7                   | Root         |
| **T3.3** Remove stale action plans           | Keep only v9, archive v5/v7/v8                             | Root         |
| **T3.4** Move phase reports to docs/archive/ | Move `PHASE*` files to `docs/archive/`                     | Root → docs/ |
| **T3.5** Move publishing guides to docs/     | Move `PUBLISHING_*`, `PUBLICATION_*` to `docs/publishing/` | Root → docs/ |
| **T3.6** Remove duplicate benchmark files    | Clean root-level `benchmark_*.py` scripts                  | Root         |
| **T3.7** Update .gitignore                   | Add patterns for temp artifacts                            | `.gitignore` |

---

## TIER 4: Advanced Automation (Next Session)

> **Goal:** Zero-touch operation. Self-healing. Self-optimizing.

| ID      | Task                                            | Prereq                     | Impact                    |
| ------- | ----------------------------------------------- | -------------------------- | ------------------------- |
| **F1**  | Auto-merge passing Dependabot PRs               | Branch protection rules    | Eliminates manual merge   |
| **F2**  | Wire chaos_test.py to nightly schedule          | security-review.yml        | Catches edge failures     |
| **F3**  | Auto-issue on persistent test failure (>2 runs) | GitHub API token           | Silent regressions caught |
| **F4**  | Deploy performance dashboard (GitHub Pages)     | Benchmark artifact storage | Trend visibility          |
| **F5**  | Auto-rollback on release failure                | Canary deploy setup        | Incident auto-resolution  |
| **F6**  | SDK regeneration in CI on main push             | ci.yml new job             | SDKs always in sync       |
| **F7**  | Generate Go + Python client SDKs                | generate_sdks.sh           | Multi-language support    |
| **F8**  | Deploy MkDocs to GitHub Pages                   | GitHub Pages config        | Public documentation site |
| **F9**  | WebSocket authentication (JWT)                  | api_server.py              | Security hardening        |
| **F10** | API rate limiting middleware                    | FastAPI middleware         | Abuse prevention          |

---

## Execution Order

```
PHASE A: Pre-flight (no push)
  T2.6 → Fix mypy/bandit/audit warnings
  T1.1 → Bump version to 2.0.0
  T1.2 → Update classifier
  T1.3 → Generate CHANGELOG
  T3.1–T3.7 → Workspace cleanup

PHASE B: CI Hardening (push to main)
  T2.1–T2.5 → Remove continue-on-error flags
  VALIDATE → Full test suite passes
  COMMIT → Push all changes

PHASE C: Release (tag push)
  T1.4 → Build & verify
  T1.5 → Delete stale tag
  T1.6 → Create v2.0.0 tag
  T1.7 → Push tag → triggers release workflow

PHASE D: Automation (next session)
  F1–F10 → Advanced automation tasks
```

---

## Success Criteria

| Metric         | Before v9             | After v9              | Verification                               |
| -------------- | --------------------- | --------------------- | ------------------------------------------ |
| Version tag    | v1.0.0 → stale commit | v2.0.0 → HEAD         | `git log v2.0.0 -1`                        |
| PyPI package   | Not published         | Published             | `pip install sigmalang`                    |
| GitHub Release | None                  | v2.0.0 with artifacts | GitHub UI                                  |
| CI strictness  | 4 gates advisory      | All gates blocking    | Push bad code → CI fails                   |
| Coverage gate  | 60%                   | 85%                   | Coverage drop → fails                      |
| Root clutter   | 100+ files            | <50 files             | `ls *.md *.txt \| wc -l`                   |
| Classifier     | "Beta"                | "Production/Stable"   | `grep "Development Status" pyproject.toml` |

---

## Autonomous Execution Protocol

The following tasks require **zero human input** and can be executed immediately:

- T1.1, T1.2, T1.3, T1.4 (version bump, build)
- T2.6 (fix existing warnings — must analyze first)
- T3.1–T3.7 (cleanup)

The following tasks require **human action**:

- T1.5–T1.7 (tag deletion + push — destructive, needs confirmation)
- T2.1–T2.5 (CI changes — need T2.6 first)
- PyPI secret configuration (GitHub UI)

**Begin autonomous execution now with T1.1 + T1.2 + T1.3 + T3.1.**
