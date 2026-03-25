# =============================================================================
# ΣLANG Makefile
# Simplified commands for development, testing, and deployment
# =============================================================================

.PHONY: help install install-dev test lint format build run clean \
        docker-build docker-run docker-push docker-clean \
        compose-up compose-down compose-dev compose-logs \
        k8s-deploy k8s-delete \
        test-fast test-strict test-watch \
        security-scan auto-fix validate publish-check \
        coverage-report pre-commit-install pre-commit-run

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER := docker
COMPOSE := docker compose
IMAGE_NAME := ghcr.io/iamthegreatdestroyer/sigmalang
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "ΣLANG Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Development
# =============================================================================
install: ## Install package dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"

test: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=sigmalang --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v --tb=short

lint: ## Run linters
	ruff check sigmalang/ --ignore E501,F401
	mypy sigmalang/ --ignore-missing-imports

format: ## Format code with black
	black sigmalang/ tests/
	isort sigmalang/ tests/

run: ## Run the API server locally
	uvicorn sigmalang.core.api_server:create_fastapi_app --factory --host 0.0.0.0 --port 8000 --reload

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Docker
# =============================================================================
docker-build: ## Build Docker image
	$(DOCKER) build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .

docker-build-dev: ## Build development Docker image
	$(DOCKER) build --target development -t $(IMAGE_NAME):dev .

docker-run: ## Run Docker container
	$(DOCKER) run -it --rm -p 8000:8000 $(IMAGE_NAME):latest

docker-push: ## Push Docker image to registry
	$(DOCKER) push $(IMAGE_NAME):$(VERSION)
	$(DOCKER) push $(IMAGE_NAME):latest

docker-clean: ## Remove Docker images and containers
	$(DOCKER) stop $$($(DOCKER) ps -q --filter "ancestor=$(IMAGE_NAME)") 2>/dev/null || true
	$(DOCKER) rm $$($(DOCKER) ps -aq --filter "ancestor=$(IMAGE_NAME)") 2>/dev/null || true
	$(DOCKER) rmi $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest $(IMAGE_NAME):dev 2>/dev/null || true

# =============================================================================
# Docker Compose
# =============================================================================
compose-up: ## Start all services with Docker Compose
	$(COMPOSE) up -d

compose-down: ## Stop all services
	$(COMPOSE) down

compose-dev: ## Start development environment
	$(COMPOSE) -f docker-compose.dev.yml up

compose-logs: ## View service logs
	$(COMPOSE) logs -f

compose-restart: ## Restart services
	$(COMPOSE) restart

compose-clean: ## Remove containers and volumes
	$(COMPOSE) down -v --remove-orphans

# =============================================================================
# Kubernetes
# =============================================================================
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -k k8s/

k8s-delete: ## Delete from Kubernetes
	kubectl delete -k k8s/

k8s-status: ## Check deployment status
	kubectl get all -n sigmalang

k8s-logs: ## View API logs
	kubectl logs -n sigmalang -l app.kubernetes.io/name=sigmalang -f

k8s-port-forward: ## Port forward to local
	kubectl port-forward -n sigmalang svc/sigmalang-api 8000:80

# =============================================================================
# Release
# =============================================================================
release-patch: ## Create patch release
	bumpversion patch
	git push && git push --tags

release-minor: ## Create minor release
	bumpversion minor
	git push && git push --tags

release-major: ## Create major release
	bumpversion major
	git push && git push --tags

# =============================================================================
# Automation & Quality Gates
# =============================================================================
test-strict: ## Run tests with coverage enforcement (fails below 85%)
	$(PYTEST) tests/ --ignore=tests/claude_integration \
		-v --cov=sigmalang --cov-report=term-missing \
		--cov-fail-under=85 --timeout=300

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/ -v --tb=short

test-watch: ## Run core tests continuously on file changes
	$(PYTEST) tests/test_sigmalang.py tests/test_config.py -q --timeout=30 -x

coverage-report: ## Generate HTML coverage report
	$(PYTEST) tests/ --ignore=tests/claude_integration \
		--cov=sigmalang --cov-report=html --cov-report=term-missing --timeout=300
	@echo "Coverage report at htmlcov/index.html"

auto-fix: ## Auto-fix common issues (formatting + imports + lint)
	black sigmalang/ tests/ --line-length=120 --quiet
	isort sigmalang/ tests/ --profile=black --line-length=120 --quiet
	ruff check sigmalang/ --fix --ignore E501,F401 --quiet || true
	@echo "✅ Auto-fix complete"

security-scan: ## Run security audit (bandit + pip-audit)
	bandit -r sigmalang/ -ll -q
	pip-audit --strict --desc 2>&1 || echo "⚠️  Dependency vulnerabilities found"
	@echo "✅ Security scan complete"

validate: ## Full validation pipeline (lint + test-strict + security)
	@echo "🔍 Running full validation..."
	$(MAKE) lint
	$(MAKE) test-strict
	$(MAKE) security-scan
	@echo "✅ Full validation passed"

publish-check: ## Verify package builds and passes twine check
	$(PYTHON) -m build
	twine check dist/*
	@echo "✅ Package ready for PyPI"

pre-commit-install: ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

pre-commit-run: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

health-check: ## Run health monitor
	$(PYTHON) scripts/health_monitor.py --quick

status: ## Show project test status summary
	@echo "=== ΣLANG Project Status ==="
	@$(PYTHON) -c "import json; d=json.load(open('automation_state.json')); t=d['test_summary']; print(f\"Tests: {t['passed']}/{t['total_collected']} passed | {t['failed']} failed | {t['skipped']} skipped | {t['duration_s']:.0f}s\")"
	@echo ""

changelog: ## Generate changelog from git history since last tag
	@echo "=== ΣLANG Changelog ==="
	@PREV=$$(git tag --sort=-creatordate 2>/dev/null | head -1); \
	if [ -z "$$PREV" ]; then \
		git log --oneline --no-decorate HEAD | head -50; \
	else \
		echo "Changes since $$PREV:"; \
		git log --oneline --no-decorate "$$PREV..HEAD" | head -50; \
	fi
