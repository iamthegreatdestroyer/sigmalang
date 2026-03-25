# 🚀 ΣLANG v1.0.0 - RELEASE EXECUTION GUIDE

**Goal:** Publish ΣLANG v1.0.0 to PyPI and Docker Hub  
**Estimated Time:** 5 minutes  
**Difficulty:** ⭐ (One command)

---

## STEP 1: Verify Your Repository Status

Before proceeding, ensure your local repository is clean and synchronized:

```bash
# Check git status
git status

# Expected output:
# "On branch main" (or develop)
# "nothing to commit, working tree clean"

# Pull latest changes
git pull origin main
```

✅ **Goal:** No uncommitted changes, branch is up-to-date

---

## STEP 2: Create the Version Tag

```bash
git tag v1.0.0
```

✅ **What this does:** Creates a local tag named `v1.0.0`  
✅ **Verification:** `git tag -l` should show `v1.0.0`

---

## STEP 3: Push the Tag to GitHub

```bash
git push origin v1.0.0
```

✅ **What this does:** Sends tag to GitHub and triggers the release workflow  
✅ **Expected output:**

```
Counting objects: 1, done.
Writing objects: 100% (1/1), 131 bytes | 131.00 KiB/s, done.
Total 1 (delta 0), reused 0 (delta 0), pack-reused 0
To github.com:iamthegreatdestroyer/sigmalang.git
 * [new tag]         v1.0.0 -> v1.0.0
```

---

## STEP 4: Monitor the Release Pipeline

### Option A: GitHub Actions Dashboard (Recommended)

1. Go to: https://github.com/iamthegreatdestroyer/sigmalang/actions
2. Click the **Release** workflow
3. Watch the jobs execute in sequence:

```
test → build → release → [publish-pypi + publish-docker]
```

**Timeline:**

- ⏱️ T+0-2m: TEST job running
- ⏱️ T+2-3m: BUILD job running
- ⏱️ T+3-3.5m: RELEASE job running
- ⏱️ T+3.5-4.5m: PUBLISH jobs running (parallel)
- ✅ T+4.5m: ALL JOBS COMPLETE

### Option B: Command Line (Terminal)

```bash
# Watch the workflow status
gh run list --workflow=release.yml --limit=1

# View full details
gh run view --log [RUN_ID]
```

---

## STEP 5: Verify Publication (After ~5 minutes)

### Verify PyPI

```bash
# Visit the PyPI page
open https://pypi.org/project/sigmalang/

# Or install it
pip install sigmalang==1.0.0
```

✅ **Expected:** Package visible on PyPI

### Verify Docker Hub

```bash
# Visit Docker Hub
open https://hub.docker.com/r/iamthegreatdestroyer/sigmalang

# Or pull the image
docker pull iamthegreatdestroyer/sigmalang:1.0.0
```

✅ **Expected:** Image available on Docker Hub

### Verify GHCR

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0
```

✅ **Expected:** Image available on GHCR

### Verify GitHub Release

```bash
# Visit releases page
open https://github.com/iamthegreatdestroyer/sigmalang/releases
```

✅ **Expected:** v1.0.0 release with auto-generated changelog

---

## ⚠️ TROUBLESHOOTING

### Issue: "Workflow didn't trigger"

**Solution:**

1. Go to Actions tab
2. Check workflow is enabled
3. Verify tag matches pattern `v*`
4. Try: `git push origin v1.0.0 --force` (if needed)

### Issue: "PyPI step failed"

**Solution:**

1. Check `PYPI_API_TOKEN` secret is created
2. Verify token value is correct (check GitHub Settings)
3. Token should start with `pypi-`

### Issue: "Docker hub step failed"

**Solution:**

1. Check both Docker Hub secrets exist:
   - `DOCKER_HUB_USERNAME`
   - `DOCKER_HUB_TOKEN`
2. Verify `DOCKER_HUB_USERNAME` = `iamthegreatdestroyer`
3. Token should start with `dckr_pat_`

### Issue: "Build failed"

**Solution:**

1. Ensure all changes are committed locally
2. Run tests locally: `python -m pytest tests/`
3. Check `pyproject.toml` version matches tag

---

## ✅ SUCCESS CHECKLIST

After publication completes, verify:

- [ ] GitHub Actions workflow shows all jobs ✅
- [ ] PyPI shows v1.0.0 package
- [ ] Docker Hub shows iamthegreatdestroyer/sigmalang:1.0.0
- [ ] GHCR shows ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0
- [ ] GitHub Releases shows v1.0.0 with changelog
- [ ] `pip install sigmalang==1.0.0` works
- [ ] `docker pull iamthegreatdestroyer/sigmalang:1.0.0` works

---

## 🎉 RELEASE COMPLETE!

ΣLANG v1.0.0 is now published to:

- 📦 **PyPI** - Available to all Python developers worldwide
- 🐳 **Docker Hub** - Available to all Docker users worldwide
- 📦 **GHCR** - Available via GitHub Container Registry

---

## 📚 REFERENCE LINKS

- **PyPI Package:** https://pypi.org/project/sigmalang/
- **Docker Hub:** https://hub.docker.com/r/iamthegreatdestroyer/sigmalang
- **GHCR:** ghcr.io/iamthegreatdestroyer/sigmalang
- **GitHub Actions:** https://github.com/iamthegreatdestroyer/sigmalang/actions
- **GitHub Releases:** https://github.com/iamthegreatdestroyer/sigmalang/releases
- **GitHub Secrets:** https://github.com/iamthegreatdestroyer/sigmalang/settings/secrets/actions

---

**You're all set! The infrastructure is ready. Execute when ready.** 🚀
