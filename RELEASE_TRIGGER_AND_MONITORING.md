# ΣLANG v1.0.0 - Release Trigger & Monitoring Guide

**Status:** Ready to Execute
**Execution Time:** ~12 minutes total (5 min setup + 4-5 min pipeline + 2 min verification)

---

## IMMEDIATE ACTION REQUIRED - Execute These Commands

Open your terminal in the s:\sigmalang directory and run:

### Step 1: Create the Release Tag (30 seconds)

```bash
git tag v1.0.0
```

**What it does:** Creates a local git tag pointing to the current commit

**Expected output:**

```
(no output - that's normal)
```

---

### Step 2: Push to GitHub (30 seconds)

```bash
git push origin v1.0.0
```

**What it does:** Pushes the tag to GitHub, triggering the automated release workflow

**Expected output:**

```
Enumerating objects: 1, done.
Counting objects: 100% (1/1), done.
Total 1 (delta 0), reused 0 (delta 0), pack-reused 0
To github.com:iamthegreatdestroyer/sigmalang.git
 * [new tag]         v1.0.0 -> v1.0.0
```

---

## MONITORING - What Happens Next

Once you push the tag, GitHub Actions automatically starts the release workflow.

### Timeline

| Time   | Event                                | Status Indicator                            |
| ------ | ------------------------------------ | ------------------------------------------- |
| T+0s   | Tag pushed to GitHub                 | ✅ Push successful (see git output)         |
| T+10s  | Workflow triggered                   | 🔄 Check Actions tab shows workflow running |
| T+30s  | Test job starts (Python 3.11)        | 🟡 "test" job shows "In progress"           |
| T+2m   | Tests complete                       | ✅ "test" job shows "Passed"                |
| T+2m   | Build job starts                     | 🟡 "build" job shows "In progress"          |
| T+3m   | Build complete                       | ✅ "build" job shows "Passed"               |
| T+3m   | Release job starts                   | 🟡 "release" job shows "In progress"        |
| T+3.5m | Release and changelog created        | ✅ "release" job shows "Passed"             |
| T+3.5m | PyPI publish job starts              | 🟡 "publish-pypi" shows "In progress"       |
| T+3.5m | Docker publish job starts (parallel) | 🟡 "publish-docker" shows "In progress"     |
| T+4.5m | Both publishing complete             | ✅ Both jobs show "Passed"                  |
| T+5m   | All done!                            | 🟢 Workflow complete                        |

---

## REAL-TIME MONITORING

### Open This URL (Refresh every 10 seconds):

**https://github.com/iamthegreatdestroyer/sigmalang/actions**

You should see:

1. **Workflow Name:** "Release"
2. **Trigger:** "iamthegreatdestroyer pushed tag v1.0.0"
3. **Status:** Shows progress as each job completes

### What to Look For

**Healthy Run:**

```
✅ test         [████████████] Passed
✅ build        [████████████] Passed
✅ release      [████████████] Passed
✅ publish-pypi [████████████] Passed
✅ publish-docker [████████] Passed
```

**Problem Indicators:**

- ❌ Red X on any job = failure (check logs)
- ⏱️ Timeout after 10 minutes = infrastructure issue
- 🟡 Still running after 8 minutes = tests slow (wait, likely OK)

---

## DETAILED JOB BREAKDOWN

### Job 1: `test` (Expected: 1-2 minutes)

**What it does:**

- Set up Python 3.11
- Install dependencies (numpy, pytest)
- Run all 500+ tests from tests/ directory
- Publish test results

**Success indicators:**

- Output shows "500+ tests passed"
- Final line: "test . . . PASSED"

**Common issues:**

- Tests timeout (unlikely, usually <2min for full suite)
- Missing dependency (unlikely, pyproject.toml locked)

---

### Job 2: `build` (Expected: 1 minute)

**What it does:**

- Download Python 3.11
- Create wheel distribution (sigmalang-1.0.0-py3-none-any.whl)
- Create source distribution (sigmalang-1.0.0.tar.gz)
- Validate distributions with twine
- Upload artifacts for downstream jobs

**Success indicators:**

- Both artifacts created (wheel + source)
- Twine validation passes
- Output shows "build . . . PASSED"

**Common issues:**

- Version mismatch (versions checked - won't happen)
- Invalid wheel metadata (wheel builder handles - won't happen)

---

### Job 3: `release` (Expected: 30 seconds)

**What it does:**

- Downloads artifacts from build job
- Extracts version from tag ("v1.0.0" → "1.0.0")
- Generates changelog based on commits since last release
- Creates GitHub Release with auto-generated release notes
- Publishes release (visible on https://github.com/iamthegreatdestroyer/sigmalang/releases)

**Success indicators:**

- New release appears at /releases page
- Release notes automatically generated
- Output shows "release . . . PASSED"

**Common issues:**

- Changelog generation empty (OK - first release)
- Release notes blank (OK - auto-generated, might be minimal)

---

### Job 4: `publish-pypi` (Expected: 1 minute)

**What it does:**

- Depends on: `release` job (must complete first)
- Downloads wheel + source distributions from build job
- Uses `PYPI_API_TOKEN` GitHub Secret to authenticate
- Publishes to PyPI using `pypa/gh-action-pypi-publish@release/v1`
- Sets `skip-existing: true` to avoid collision if version exists

**Success indicators:**

- Output shows "Uploading distributions to PyPI..."
- Final output: "Publishing package distribution releases to PyPI"
- No error about invalid token or 401 authentication

**Common issues:**

- "Invalid credentials" (403) = PYPI_API_TOKEN not created or wrong
- "Version already exists" = Package v1.0.0 already on PyPI (skip-existing handles this)
- Token format wrong (should start with `pypi-`)

**After success:**
Package available: https://pypi.org/project/sigmalang/1.0.0/

---

### Job 5: `publish-docker` (Expected: 1-2 minutes, runs PARALLEL with publish-pypi)

**What it does:**

- Depends on: `release` job
- Set up Docker Buildx
- Log in to GHCR using auto-managed GITHUB_TOKEN
- Log in to Docker Hub using DOCKER_HUB_USERNAME + DOCKER_HUB_TOKEN
- Extract version from tag for image tagging
- Build multi-stage Docker image from Dockerfile.prod
- Push to 4 registries in parallel:
  1. `ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0`
  2. `ghcr.io/iamthegreatdestroyer/sigmalang:latest`
  3. `iamthegreatdestroyer/sigmalang:1.0.0`
  4. `iamthegreatdestroyer/sigmalang:latest`

**Success indicators:**

- Output shows "Building..." then "Pushing..."
- No authentication errors
- Both DOCKER_HUB_USERNAME and DOCKER_HUB_TOKEN referenced successfully
- Final: "publish-docker . . . PASSED"

**Common issues:**

- "Authentication failed" (401) = DOCKER_HUB_TOKEN invalid or not created
- Username in tag wrong = DOCKER_HUB_USERNAME not set correctly
- Build fails = Dockerfile.prod syntax issue (unlikely - verified)

**After success:**
Images available at:

- Docker Hub: https://hub.docker.com/r/iamthegreatdestroyer/sigmalang
- GHCR: https://github.com/iamthegreatdestroyer/sigmalang/pkgs/container/sigmalang

---

## VERIFICATION CHECKLIST (After Workflow Completes)

Once all jobs show ✅, verify publication on all 3 registries:

### ✅ PyPI Verification (3 minutes)

**URL:** https://pypi.org/project/sigmalang/1.0.0/

**What to check:**

- [ ] Version 1.0.0 listed
- [ ] Wheel distribution present
- [ ] Source distribution present
- [ ] README renders correctly
- [ ] Classifiers show python/development status

**Install locally to verify:**

```bash
pip install sigmalang==1.0.0
python -c "import sigmalang; print(sigmalang.__version__)"
```

**Expected output:**

```
1.0.0
```

---

### ✅ Docker Hub Verification (3 minutes)

**URL:** https://hub.docker.com/r/iamthegreatdestroyer/sigmalang

**What to check:**

- [ ] Repository exists and shows as "public"
- [ ] Tags tab shows: `1.0.0` and `latest`
- [ ] Image size reasonable (~400-500MB)
- [ ] Docker pull command works

**Pull and verify:**

```bash
docker pull iamthegreatdestroyer/sigmalang:1.0.0
docker run --rm iamthegreatdestroyer/sigmalang:1.0.0 --version
```

**Expected output:**

```
1.0.0
```

---

### ✅ GHCR Verification (2 minutes)

**URL:** https://github.com/iamthegreatdestroyer/sigmalang/pkgs/container/sigmalang

**What to check:**

- [ ] `ghcr.io/iamthegreatdestroyer/sigmalang` exists
- [ ] Tags show: `1.0.0` and `sha-{short-hash}`
- [ ] Package visibility set to "Public" (or internal if preferred)

**Pull and verify:**

```bash
docker pull ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0
docker run --rm ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0 --version
```

**Expected output:**

```
1.0.0
```

---

### ✅ GitHub Releases Verification (1 minute)

**URL:** https://github.com/iamthegreatdestroyer/sigmalang/releases

**What to check:**

- [ ] "v1.0.0" release listed at top
- [ ] Release date = today
- [ ] "Wheel distribution" attachment present
- [ ] "Source distribution" attachment present
- [ ] Release notes auto-generated from commits

**Release is public and immediately accessible**

---

## TROUBLESHOOTING DURING PIPELINE

### If Test Job Fails ❌

**Likely causes:**

1. Test environment issue (usually OK, been tested locally)
2. Dependency mismatch (pyproject.toml locked - shouldn't happen)
3. Timeout (tests take too long - would take >3 minutes)

**Check logs at:** https://github.com/iamthegreatdestroyer/sigmalang/actions → Click failed run → "test" job → "Run tests" step

**Resolution:** Fix issue locally, commit, push new tag (e.g., v1.0.1)

---

### If Build Job Fails ❌

**Likely causes:**

1. Invalid wheel metadata (shouldn't happen - validated earlier)
2. Version mismatch (pyproject.toml version ≠ tag - shouldn't happen)
3. Missing file in dist/ (artifacts should be there)

**Check logs at:** https://github.com/iamthegreatdestroyer/sigmalang/actions → Click failed run → "build" job

**Resolution:** Rebuild locally with `python -m build`, verify in dist/, commit, push new tag

---

### If PyPI Publish Fails ❌

**Error Messages & Solutions:**

```
"Invalid credentials (401)"
→ PYPI_API_TOKEN not created or pasted incorrectly
→ Solution: Verify token at https://pypi.org/manage/account/token/
→ Recreate GitHub Secret: https://github.com/iamthegreatdestroyer/sigmalang/settings/secrets

"Version already exists (409)"
→ Package sigmalang 1.0.0 already on PyPI
→ Solution: Workflow has skip-existing: true, should auto-pass
→ If blocked: Use v1.0.1 for next release

"Token format invalid"
→ Token doesn't look like PyPI token
→ Solution: Verify it starts with "pypi-", contains hyphens, not just random string
```

**Recovery:** Fix secret, git tag v1.0.1, git push

---

### If Docker Push Fails ❌

**Error Messages & Solutions:**

```
"Authentication failed (401) for Docker Hub"
→ DOCKER_HUB_TOKEN invalid or DOCKER_HUB_USERNAME wrong
→ Solution: Verify at https://hub.docker.com/settings/security
→ Check GitHub Secrets: both must be created

"Unauthorized: incorrect username or password"
→ Token not a personal access token (PAT)
→ Solution: Create new PAT at https://hub.docker.com/settings/security
→ Delete old secret, recreate with new token

"Image digest doesn't match expected"
→ Unlikely - indicates build inconsistency
→ Solution: Rebuild locally, push new tag v1.0.1
```

**Recovery:** Fix secret(s), git tag v1.0.1, git push

---

## WHAT HAPPENS IF YOU NEED TO RE-RELEASE

If workflow fails and you fix the issue:

**Option 1: New version tag (recommended)**

```bash
git tag v1.0.1
git push origin v1.0.1
```

Creates separate v1.0.1 release, keeps v1.0.0 as failed artifact

**Option 2: Delete and retag (not recommended)**

```bash
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
git tag v1.0.0
git push origin v1.0.0
```

Overwrites v1.0.0 - can cause confusion in release history

---

## ALTERNATIVE MONITORING: Command Line

If you prefer command-line monitoring, you can watch the workflow:

```bash
# List all workflow runs
gh run list --repo iamthegreatdestroyer/sigmalang

# Watch a specific run (if you have run ID)
gh run watch <RUN_ID> --repo iamthegreatdestroyer/sigmalang

# Check release creation
gh release view v1.0.0 --repo iamthegreatdestroyer/sigmalang
```

(Requires GitHub CLI: https://cli.github.com/)

---

## SUMMARY

### What You Do (5 minutes)

1. Run: `git tag v1.0.0`
2. Run: `git push origin v1.0.0`
3. Monitor at: https://github.com/iamthegreatdestroyer/sigmalang/actions

### What GitHub Does (4-5 minutes)

- Run 500+ tests → Build distributions → Create release → Publish to PyPI + Docker

### What You Verify (2 minutes)

- Check PyPI: pip install sigmalang==1.0.0
- Check Docker Hub: docker pull iamthegreatdestroyer/sigmalang:1.0.0
- Check GHCR: docker pull ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0
- Check Releases: Review at /releases page

### Total Time: ~12 minutes

### Result: ΣLANG v1.0.0 Published Globally ✅

---

**Ready to proceed? Execute the two git commands above and monitor the workflow!**

Generated: Release Trigger & Monitoring Guide v1.0
