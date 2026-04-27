# ΣLANG Publishing Credentials Setup Guide

## Overview

Your GitHub Actions release workflow is configured to automatically publish ΣLANG v1.0.0 to both PyPI and Docker Hub when you create a version tag (e.g., `git tag v1.0.0`).

## Required GitHub Secrets

Create these secrets in your repository: **Settings → Secrets and variables → Actions → New repository secret**

### 1. PyPI API Token

- **Secret Name:** `PYPI_API_TOKEN`
- **Value:** Your PyPI API token (the token you generated from PyPI)
- **How to get it:**
  1. Go to https://pypi.org/manage/account/token/
  2. Create a new token with "scope" = "Entire account" or specific project
  3. Copy the token (starts with `pypi-`)

### 2. Docker Hub Credentials

- **Secret Name:** `DOCKER_HUB_USERNAME`
- **Value:** Your Docker Hub username (e.g., `your-username`)

- **Secret Name:** `DOCKER_HUB_TOKEN`
- **Value:** Your Docker Hub personal access token
- **How to get it:**
  1. Go to https://hub.docker.com/settings/security
  2. Create a new personal access token
  3. Copy the token

## Workflow Configuration

### File: `.github/workflows/release.yml`

**Current Status:** ✅ FULLY CONFIGURED

#### PyPI Publishing Step

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
    skip-existing: true
```

- Publishes Python package to PyPI
- Uses your `PYPI_API_TOKEN` secret
- Skips already-published versions (prevents version conflicts)

#### Docker Hub Publishing Step

```yaml
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKER_HUB_USERNAME }}
    password: ${{ secrets.DOCKER_HUB_TOKEN }}

# Inside build-push-action, tags include:
tags: |
  ghcr.io/${{ github.repository }}:${{ steps.meta.outputs.VERSION }}
  ghcr.io/${{ github.repository }}:latest
  ${{ secrets.DOCKER_HUB_USERNAME }}/sigmalang:${{ steps.meta.outputs.VERSION }}
  ${{ secrets.DOCKER_HUB_USERNAME }}/sigmalang:latest
```

- Logs in to Docker Hub using your credentials
- Builds and pushes images to both GHCR (GitHub Container Registry) and Docker Hub
- Tags with version number AND `latest` tag

## How to Trigger Publishing

Once you've created all three GitHub Secrets:

```bash
# Create a version tag
git tag v1.0.0

# Push the tag to trigger the release workflow
git push origin v1.0.0
```

This will:

1. ✅ Run all tests
2. ✅ Build Python distribution packages
3. ✅ Create GitHub Release with changelog
4. ✅ Publish to PyPI
5. ✅ Build and push Docker image to both GHCR and Docker Hub

## Verification Checklist

Before publishing:

- [ ] `PYPI_API_TOKEN` created in GitHub Secrets
- [ ] `DOCKER_HUB_USERNAME` created in GitHub Secrets
- [ ] `DOCKER_HUB_TOKEN` created in GitHub Secrets
- [ ] Workflow file `.github/workflows/release.yml` is up-to-date (✅ confirmed)
- [ ] `pyproject.toml` version matches your tag (e.g., both are `1.0.0`)
- [ ] All tests passing locally: `pytest tests/`

## Troubleshooting

### PyPI Upload Fails

- Verify `PYPI_API_TOKEN` is valid and not expired
- Check that version in `pyproject.toml` hasn't been published before
- Ensure `README.md` and `LICENSE` exist (required by PyPI)

### Docker Hub Upload Fails

- Verify `DOCKER_HUB_USERNAME` and `DOCKER_HUB_TOKEN` are correct
- Ensure Docker Hub username matches your account
- Check that repository `sigmalang` exists in your Docker Hub account (or create it)

### GitHub Actions Workflow

- Check workflow runs at: `https://github.com/iamthegreatdestroyer/sigmalang/actions`
- View detailed logs if any step fails
- Ensure you're pushing to the correct branch (`main`)

## Credentials Location

| Credential            | Stored In                       | Visibility                                |
| --------------------- | ------------------------------- | ----------------------------------------- |
| `PYPI_API_TOKEN`      | GitHub Secrets                  | ✅ Encrypted, only accessible in Actions  |
| `DOCKER_HUB_USERNAME` | GitHub Secrets                  | ✅ Encrypted, only accessible in Actions  |
| `DOCKER_HUB_TOKEN`    | GitHub Secrets                  | ✅ Encrypted, only accessible in Actions  |
| Workflow file         | `.github/workflows/release.yml` | ✅ Public (doesn't contain actual tokens) |

---

**Last Updated:** 2026-03-24
**Workflow File:** `.github/workflows/release.yml`
**Status:** Ready for Publishing ✅
