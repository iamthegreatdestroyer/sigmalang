# Next Steps Master Action Plan — v10

**Theme:** Maximum Autonomy & Automation
**Baseline commit:** `12ed12b` (foundation cleanup pushed to `origin/main`)
**Predecessor:** [NEXT_STEPS_MASTER_ACTION_PLAN_v9.md](NEXT_STEPS_MASTER_ACTION_PLAN_v9.md)
**Status:** Active

---

## Mission

Transform the SigmaLang repository from a *manually-maintained* state into a *self-maintaining, self-releasing, self-documenting* system. Every recurring human action becomes a workflow; every workflow is auditable; every state file updates itself.

---

## Guiding Principles

1. **Push-button or no-button.** Releases, security scans, dependency bumps, docs publishing all triggered by tags or schedules — never by hand.
2. **Single source of truth.** `automation_state.json` is the canonical state ledger; CI updates it.
3. **Fail loud, fix automatically.** Bots open PRs; humans approve.
4. **Reversibility.** Every automation change is a separate commit with a clear revert path.

---

## Tier A — Release Finalization (Quickest Wins)

| ID | Action | Owner |
|----|--------|-------|
| A1 | Delete stale local + remote tag `v1.0.0` (`3da6960`) | now |
| A2 | Create annotated tag `v2.0.0` at `12ed12b`; push to origin | now |
| A3 | Verify `PYPI_API_TOKEN` GitHub secret exists; document procedure in `PUBLISHING_CREDENTIALS_SETUP.md` | now |
| A4 | Confirm release workflow fires on `v2.0.0` tag push | observe |

**Exit criteria:** `git ls-remote --tags origin` shows `v2.0.0`, no `v1.0.0`.

---

## Required Repository Secrets

Configure once via `gh secret set <NAME>` (or GitHub UI → Settings → Secrets and variables → Actions). These secrets unblock fully autonomous release + container publish flows.

| Secret | Required By | Purpose | Acquisition |
|--------|-------------|---------|-------------|
| `PYPI_API_TOKEN` | `release.yml:142` | Publish wheels/sdist to PyPI on tag push | https://pypi.org/manage/account/token/ → scope: project `sigmalang` |
| `DOCKER_HUB_USERNAME` | `release.yml:167` | Authenticate Docker Hub push | Docker Hub account username |
| `DOCKER_HUB_TOKEN` | `release.yml:168` | Push container images on release | https://hub.docker.com/settings/security → New Access Token (Read/Write/Delete) |
| `CODECOV_TOKEN` *(optional)* | `ci.yml` | Upload coverage reports | https://app.codecov.io/gh/iamthegreatdestroyer/sigmalang/settings → Repository Upload Token |

**Setup commands (PowerShell):**

```powershell
gh secret set PYPI_API_TOKEN --body "pypi-AgEI..."
gh secret set DOCKER_HUB_USERNAME --body "iamthegreatdestroyer"
gh secret set DOCKER_HUB_TOKEN --body "dckr_pat_..."
gh secret set CODECOV_TOKEN --body "..."   # optional
```

**Verification:** `gh secret list` should show all four (or three) entries. Re-trigger `release.yml` via tag push or `workflow_dispatch` to confirm jobs no longer skip on missing credentials.

---

## Tier B — Carry-over from v9 Tier 4 (F1–F10)

Items deferred from v9 that remain valuable but were not blockers. Tracked but executed *after* Tier C autonomy infrastructure is in place so they benefit from automation.

- F1–F10: see [NEXT_STEPS_MASTER_ACTION_PLAN_v9.md](NEXT_STEPS_MASTER_ACTION_PLAN_v9.md) Tier 4.

---

## Tier C — Autonomy Maximizers (NEW Workflows)

Add 8 new workflow files under `.github/workflows/` and 2 config files.

| ID | File | Purpose | Trigger |
|----|------|---------|---------|
| C1 | `.github/dependabot.yml` | Weekly dependency PRs (pip + actions) | schedule |
| C2 | `.github/workflows/codeql.yml` | SAST via GitHub CodeQL | push, weekly |
| C3 | `.github/workflows/trivy.yml` | Container + filesystem CVE scan | push, weekly |
| C4 | `.github/workflows/pages.yml` | Build & deploy MkDocs to GitHub Pages | push to main, manual |
| C5 | `.github/workflows/benchmark.yml` | Run + publish benchmark results | push to main, manual |
| C6 | `.github/workflows/auto-merge.yml` | Auto-merge Dependabot patch/minor PRs after CI green | PR labeled |
| C7 | `.github/workflows/stale.yml` | Close stale issues/PRs (60d/90d) | daily |
| C8 | `.github/workflows/labeler.yml` | Auto-label PRs by path | pull_request |
| C9 | `.github/labeler.yml` | Path-to-label config consumed by C8 | — |
| C10 | `.github/workflows/release.yml` enhancement | Add Codecov upload + semantic-release dry-run | tag |

**Exit criteria:** All 8 workflows green on first scheduled / triggered run.

---

## Tier D — Workspace Hygiene

| ID | Action |
|----|--------|
| D1 | Archive root `test_*.txt`, `ws_test_*.txt`, `streaming_results_v2.txt`, `compression_analysis.py` outputs into `archive/` (gitignored) |
| D2 | Consolidate `PUBLISHING_*.md` (5 files) → single `docs/publishing.md`; leave stub redirects |
| D3 | Move `benchmark_*.py` from root → `benchmarks/` |
| D4 | Move root `test_*.py` → `tests/` |
| D5 | Delete `sgbil.lnk` (Windows shortcut, repo-irrelevant) |
| D6 | Add `archive/` to `.gitignore` |

**Exit criteria:** Repo root contains no transient artifacts; `git status` clean post-cleanup.

---

## Tier E — Self-Updating Automation

| ID | Action |
|----|--------|
| E1 | New workflow `.github/workflows/state-sync.yml` — on push to main, regenerates `automation_state.json` (version, last commit, workflow status) and opens PR if changed |
| E2 | New workflow `.github/workflows/badge-refresh.yml` — updates README badge URLs nightly |
| E3 | New workflow `.github/workflows/changelog.yml` — appends to `CHANGELOG.md` from conventional commits on tag push |
| E4 | Pre-commit hook → auto-update `EXECUTIVE_SUMMARY` "Last Updated" footer |

**Exit criteria:** State files require zero manual edits for routine updates.

---

## Execution Order

```
Tier A (tags)          ──► quick win, unblocks PyPI release
   │
   ▼
Tier C (workflows)     ──► foundation for everything else
   │
   ▼
Tier D (hygiene)       ──► clean substrate for Tier E
   │
   ▼
Tier E (self-update)   ──► autonomy realized
   │
   ▼
Tier B (v9 carry-over) ──► benefits from new automation
   │
   ▼
Bump automation_state.json → v5.0
```

---

## Success Metrics

- ✅ Zero manual `git tag` operations after v2.0.0
- ✅ Zero manual dependency bumps (Dependabot owns it)
- ✅ Zero manual docs deployments (Pages workflow owns it)
- ✅ `automation_state.json` accurate within 5 min of every push
- ✅ All 12 workflows in `.github/workflows/` green

---

**Last Updated:** 2025 (auto-managed post-Tier-E)
