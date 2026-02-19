# Publication & Release Guide

Complete guide for publishing ΣLANG to PyPI, Docker Hub, and GitHub Releases.

## Pre-Release Checklist

### Code & Quality
- [ ] All tests passing (1,656/1,656)
- [ ] Coverage >85%
- [ ] No security issues (bandit, gitleaks)
- [ ] Code formatted (black)
- [ ] Type hints complete (mypy)
- [ ] Documentation complete
- [ ] Changelog updated

### Security
- [ ] No hardcoded credentials
- [ ] .env.example created
- [ ] .gitignore includes secrets
- [ ] Security audit completed
- [ ] Dependencies checked (pip audit)
- [ ] License headers present

### Versioning
- [ ] Version updated in:
  - `pyproject.toml`
  - `sigmalang/__version__.py`
  - `helm/Chart.yaml`
  - `docker-compose.yml`
- [ ] Changelog updated
- [ ] Git tag created

## Publication Steps

### 1. Prepare PyPI Release

#### Setup PyPI Credentials

```bash
# Create ~/.pypirc
cat > ~/.pypirc <<EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-<your-token-here>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-test-token-here>
EOF

chmod 600 ~/.pypirc
```

Get tokens from: https://pypi.org/manage/account/tokens/

#### Build Distribution

```bash
# Install build tools
pip install build twine

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build distribution
python -m build

# Verify build
twine check dist/*
```

#### Test on PyPI

```bash
# Upload to test PyPI
twine upload --repository testpypi dist/*

# Install from test
pip install --index-url https://test.pypi.org/simple/ sigmalang

# Test import
python -c "import sigmalang; print(sigmalang.__version__)"
```

#### Publish to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Verify publication
pip search sigmalang  # or check https://pypi.org/project/sigmalang/
```

### 2. Docker Hub Release

#### Setup Docker Hub Credentials

```bash
# Login to Docker Hub
docker login

# Create ~/.docker/config.json automatically via login
```

#### Build Docker Image

```bash
# Build with version tag
docker build -t ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0 .
docker build -t ghcr.io/iamthegreatdestroyer/sigmalang:latest .

# Or use Docker Compose
docker compose build --no-cache
```

#### Push to Docker Hub

```bash
# Tag image
docker tag ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0 \
           sigmalang/sigmalang:1.0.0
docker tag sigmalang/sigmalang:1.0.0 \
           sigmalang/sigmalang:latest

# Push to Docker Hub
docker push sigmalang/sigmalang:1.0.0
docker push sigmalang/sigmalang:latest

# Verify
docker pull sigmalang/sigmalang:latest
```

#### Push to GitHub Container Registry (GHCR)

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Push image
docker push ghcr.io/iamthegreatdestroyer/sigmalang:1.0.0
docker push ghcr.io/iamthegreatdestroyer/sigmalang:latest
```

### 3. GitHub Release

#### Create Release Tag

```bash
# Create annotated tag
git tag -a v1.0.0 -m "Release 1.0.0 - Production Ready"

# Push tag
git push origin v1.0.0
```

#### Create Release on GitHub

```bash
# Using GitHub CLI
gh release create v1.0.0 \
  --title "ΣLANG 1.0.0" \
  --notes "Production release with 1,656 tests passing" \
  dist/*
```

Or manually via GitHub web interface:
1. Go to https://github.com/iamthegreatdestroyer/sigmalang/releases
2. Click "Draft a new release"
3. Select version tag
4. Add release notes
5. Upload assets (dist files)
6. Publish

### 4. Helm Chart Release

#### Publish Helm Chart

```bash
# Package chart
helm package helm/

# Create index
helm repo index . --url https://charts.sigmalang.io

# Upload to S3 (or other hosting)
aws s3 sync . s3://sigmalang-charts/ \
  --include "*.tgz" \
  --include "index.yaml"
```

### 5. Documentation Site Release

#### Build Documentation

```bash
# Build MkDocs site
mkdocs build

# Push to GitHub Pages
mkdocs gh-deploy
```

Or manually:
```bash
# Build
mkdocs build

# Commit built site
git add site/
git commit -m "docs: Build documentation for 1.0.0"
git push

# Deploy via workflow
# (Automated GitHub Actions deployment)
```

## Post-Release

### Verification

```bash
# Verify PyPI package
pip install --upgrade sigmalang
python -c "import sigmalang; print(sigmalang.__version__)"

# Verify Docker image
docker run --rm sigmalang/sigmalang:latest --version

# Verify Helm chart
helm search repo sigmalang
helm install test sigmalang/sigmalang --dry-run

# Verify documentation
open https://sigmalang.io
```

### Announcements

- [ ] Update GitHub releases page
- [ ] Post announcement on social media
- [ ] Update project README
- [ ] Post in community forums
- [ ] Send email to stakeholders

### Maintenance

```bash
# Monitor package statistics
# PyPI: https://pypi.org/project/sigmalang/
# Docker Hub: https://hub.docker.com/r/sigmalang/sigmalang

# Set up automated testing for new Python versions
# Update CI/CD for new Python/Node versions
```

## Release Branches & Workflow

### Semantic Versioning

Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

Examples:
- `0.1.0` → `1.0.0`: First major release
- `1.0.0` → `1.1.0`: New feature
- `1.1.0` → `1.1.1`: Bug fix

### Release Branch Workflow

```bash
# Create release branch
git checkout -b release/1.0.0

# Update version numbers
sed -i 's/0.9.0/1.0.0/g' pyproject.toml
sed -i 's/0.9.0/1.0.0/g' helm/Chart.yaml
sed -i 's/0.9.0/1.0.0/g' docker-compose.yml

# Update CHANGELOG
# Add release notes

# Commit version bump
git add pyproject.toml helm/Chart.yaml docker-compose.yml CHANGELOG.md
git commit -m "chore: Bump version to 1.0.0"

# Push release branch
git push origin release/1.0.0

# Create Pull Request for review

# After approval, merge to main
git checkout main
git merge release/1.0.0

# Tag the release
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin main v1.0.0

# Delete release branch
git push origin --delete release/1.0.0
```

## Automated Release CI/CD

### GitHub Actions Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Build distribution
        run: |
          pip install build twine
          python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

      - name: Build Docker image
        run: docker build -t sigmalang/sigmalang:${{ github.ref_name }} .

      - name: Push Docker image
        run: docker push sigmalang/sigmalang:${{ github.ref_name }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Troubleshooting

### PyPI Upload Issues

```bash
# Invalid credentials
# Solution: Check ~/.pypirc and token validity

# Package already exists
# Solution: Increment version number

# Invalid metadata
# Solution: Run twine check and fix issues
```

### Docker Push Issues

```bash
# Authentication failed
# Solution: Run docker login and retry

# Image too large
# Solution: Optimize Dockerfile, reduce layers

# Tag not found
# Solution: Build image first, then tag
```

### Helm Chart Issues

```bash
# Chart validation failed
# Solution: Run helm lint and fix issues

# Upload failed
# Solution: Check S3/storage credentials
```

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Helm Docs](https://helm.sh/docs/)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated**: February 19, 2026
**Version**: 1.0.0
**Status**: Production Release
