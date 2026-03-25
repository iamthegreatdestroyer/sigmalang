# GitHub Secrets Setup Instructions - ΣLANG v1.0.0

**Purpose:** Configure GitHub Repository Secrets required for automated release pipeline

**Timeline:** 5 minutes

---

## Prerequisites

Before starting, you need:

1. **PyPI API Token** - Get from https://pypi.org/manage/account/token/
2. **Docker Hub Username** - Your Docker Hub account username
3. **Docker Hub Token** - Get from https://hub.docker.com/settings/security

---

## Step-by-Step: Add GitHub Secrets

### Step 1: Navigate to Repository Settings

1. Go to: https://github.com/iamthegreatdestroyer/sigmalang
2. Click **Settings** tab
3. In left sidebar, click **Secrets and variables** → **Actions**

You should see: "Repository secrets" section with any existing secrets

---

### Step 2: Add PYPI_API_TOKEN

1. Click **New repository secret** button
2. **Name:** `PYPI_API_TOKEN`
3. **Value:** Paste your PyPI token (starts with `pypi-`)
4. Click **Add secret**

✅ `PYPI_API_TOKEN` now created (encrypted, not visible)

---

### Step 3: Add DOCKER_HUB_USERNAME

1. Click **New repository secret** button
2. **Name:** `DOCKER_HUB_USERNAME`
3. **Value:** Your Docker Hub username (e.g., `iamthegreatdestroyer`)
4. Click **Add secret**

✅ `DOCKER_HUB_USERNAME` now created

---

### Step 4: Add DOCKER_HUB_TOKEN

1. Click **New repository secret** button
2. **Name:** `DOCKER_HUB_TOKEN`
3. **Value:** Your Docker Hub personal access token (starts with `dckr_pat_`)
4. Click **Add secret**

✅ `DOCKER_HUB_TOKEN` now created

---

## Verification Checklist

After adding all 3 secrets, you should see on the "Actions secrets and variables" page:

```
Repository secrets
──────────────────────────────────────────
✓ DOCKER_HUB_TOKEN       Updated just now
✓ DOCKER_HUB_USERNAME    Updated just now
✓ PYPI_API_TOKEN         Updated just now
```

(The order may vary, but all 3 should be listed)

---

## How These Secrets Are Used

| Secret Name           | Used By                     | Purpose                                      |
| --------------------- | --------------------------- | -------------------------------------------- |
| `PYPI_API_TOKEN`      | PyPI publish job (line 142) | Authenticates to PyPI.org for package upload |
| `DOCKER_HUB_USERNAME` | Docker login job (line 167) | Username for Docker Hub authentication       |
| `DOCKER_HUB_TOKEN`    | Docker login job (line 168) | Token for Docker Hub authentication          |

All three are referenced in `.github/workflows/release.yml` and will be automatically injected into the GitHub Actions environment when the workflow runs.

---

## Security Notes

⚠️ **Important:**

- Secrets are encrypted and never visible in logs
- Once created, you cannot see the value again (by design)
- If you need to update, delete the old one and create a new one
- Never commit credential values to code (GitHub blocks this automatically)

---

## Trigger Release

Once all 3 secrets are created, you can trigger the release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will automatically:

1. Detect the tag
2. Run tests
3. Build distributions
4. Publish to PyPI (using `PYPI_API_TOKEN`)
5. Publish to Docker Hub (using `DOCKER_HUB_USERNAME` + `DOCKER_HUB_TOKEN`)

Timeline: ~4-5 minutes total

---

## Troubleshooting

### Secret not being used?

- Verify the secret name exactly matches what's in the workflow file
- (Workflow expects: `PYPI_API_TOKEN`, `DOCKER_HUB_USERNAME`, `DOCKER_HUB_TOKEN`)
- GitHub is case-sensitive

### "Secret not found" error in workflow?

- Verify on the Secrets and variables page that the secret is listed
- If missing, create it using the steps above
- Workflow will automatically retry on next tag push

### Token format wrong?

- PyPI tokens start with `pypi-`
- Docker Hub PAT tokens start with `dckr_pat_`
- If wrong format, delete and recreate with correct token

---

## Next Steps

1. ✅ Gather credential values (PyPI token, Docker Hub username/token)
2. ✅ Follow steps above to add 3 GitHub Secrets
3. ✅ Verify all 3 secrets appear in GitHub Settings
4. ✅ Execute release: `git tag v1.0.0 && git push origin v1.0.0`
5. ✅ Monitor at https://github.com/iamthegreatdestroyer/sigmalang/actions

---

**All infrastructure is ready. Once these 3 secrets are added to GitHub, you can publish ΣLANG v1.0.0 immediately.**
