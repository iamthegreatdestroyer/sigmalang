# ΣLANG Docker Image - Multi-stage optimized build
# Target: <500MB image size, <5s startup time

# =============================================================================
# STAGE 1: Builder - Install dependencies with compilation tools
# =============================================================================
FROM python:3.12-slim AS builder

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build

# Copy only dependency files first (for better layer caching)
COPY pyproject.toml README.md ./
COPY sigmalang/__init__.py sigmalang/__init__.py

# Install numpy first (required for other scientific packages)
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy

# Install the package with all dependencies
COPY . .
RUN pip install -e ".[dev]"

# =============================================================================
# STAGE 2: Production - Minimal runtime image
# =============================================================================
FROM python:3.12-slim AS production

# Labels for container metadata
LABEL org.opencontainers.image.title="ΣLANG" \
    org.opencontainers.image.description="Neural-inspired semantic compression framework" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.vendor="ΣLANG Project" \
    org.opencontainers.image.source="https://github.com/iamthegreatdestroyer/sigmalang" \
    org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # ΣLANG configuration
    SIGMALANG_API_HOST=0.0.0.0 \
    SIGMALANG_API_PORT=8000 \
    SIGMALANG_LOG_LEVEL=INFO \
    SIGMALANG_LOG_FORMAT=json \
    SIGMALANG_METRICS_ENABLED=true \
    # Virtual environment
    PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 sigmalang && \
    useradd --uid 1000 --gid sigmalang --shell /bin/bash --create-home sigmalang

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY --chown=sigmalang:sigmalang . .

# Switch to non-root user
USER sigmalang

# Expose API port
EXPOSE 8000

# Health check - uses the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command - run the API server
CMD ["sigmalang-server"]

# =============================================================================
# STAGE 3: Development - Includes dev tools and hot reload
# =============================================================================
FROM production AS development

# Switch back to root to install dev tools
USER root

# Install development dependencies
RUN pip install watchdog pytest pytest-cov pytest-asyncio httpx

# Create volume mount points
VOLUME ["/app"]

# Switch back to non-root user
USER sigmalang

# Development command with hot reload
CMD ["uvicorn", "sigmalang.core.api_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# STAGE 4: CLI - Lightweight image for CLI usage
# =============================================================================
FROM production AS cli

# Override entrypoint for CLI usage
ENTRYPOINT ["sigmalang"]
CMD ["--help"]
