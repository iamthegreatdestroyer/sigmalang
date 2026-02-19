# ΣLANG Local Setup & Testing Guide

**Last Updated:** February 19, 2026
**Status:** Production-Ready (1,656/1656 tests passing)

## Quick Start (5 minutes)

### Prerequisites
- Docker & Docker Compose installed
- Python 3.9+ (optional, for CLI testing)
- Git

### Option 1: Docker Compose (Recommended for Testing)

```bash
# Clone the repository
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang

# Start all services (API, Redis, Prometheus, Grafana)
docker compose up -d

# Verify services are running
docker compose ps

# View API logs
docker compose logs -f sigmalang
```

**Services Available:**
- API Server: http://localhost:26080
- Prometheus: http://localhost:26900
- Grafana: http://localhost:26910 (admin/sigmalang)
- Redis: localhost:26500

### Option 2: Local Installation (Development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run API server
sigmalang-server

# In another terminal, run CLI
sigmalang encode "Hello, World!"
```

---

## Testing Locally

### API Testing

```bash
# Health check
curl http://localhost:26080/health

# Encode text
curl -X POST http://localhost:26080/api/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, World!"}'

# View API docs
open http://localhost:26080/docs
```

### CLI Testing

```bash
# Encode text
sigmalang encode "The quick brown fox jumps over the lazy dog"

# Encode from file
sigmalang encode -i input.txt -o output.bin

# Decode
sigmalang decode -i output.bin -o decoded.txt

# Search entities
sigmalang entities "Apple Inc is located in Cupertino, California"
```

### Running Full Test Suite

```bash
# Run all tests (1,656 tests)
pytest tests/ -v

# Run specific test module
pytest tests/test_optimizations.py -v

# Run with coverage
pytest tests/ --cov=sigmalang --cov-report=html

# Run performance benchmarks
pytest tests/ -v -k benchmark
```

---

## Configuration

### Environment Variables

**API Server:**
```bash
SIGMALANG_API_HOST=0.0.0.0          # Bind address
SIGMALANG_API_PORT=8000             # API port
SIGMALANG_API_WORKERS=4             # Worker processes
SIGMALANG_DEBUG=false               # Debug mode
```

**Cache & Storage:**
```bash
SIGMALANG_CACHE_BACKEND=redis       # redis or memory
SIGMALANG_REDIS_URL=redis://localhost:6379/0
SIGMALANG_CACHE_ENABLED=true
```

**Monitoring:**
```bash
SIGMALANG_METRICS_ENABLED=true      # Prometheus metrics
SIGMALANG_METRICS_PORT=9090
SIGMALANG_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
```

### Configuration File

Create `.env` file in project root:
```ini
SIGMALANG_API_PORT=8000
SIGMALANG_DEBUG=false
SIGMALANG_CACHE_BACKEND=redis
SIGMALANG_LOG_LEVEL=INFO
```

---

## Development Workflows

### Add a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Write tests
# tests/test_my_feature.py

# 3. Run tests (should fail)
pytest tests/test_my_feature.py

# 4. Implement feature
# sigmalang/core/my_feature.py

# 5. Run tests (should pass)
pytest tests/test_my_feature.py -v

# 6. Run full suite to ensure no regressions
pytest tests/ -x

# 7. Commit and push
git add .
git commit -m "feat: Add my feature"
git push origin feature/my-feature
```

### Performance Testing

```bash
# Run benchmark tests
pytest tests/ -v -k benchmark

# Profile memory usage
python -m cProfile -s cumulative -o profile.stats tests/test_memory_profiling.py

# View profiling results
python -m pstats profile.stats
```

### Integration Testing

```bash
# Run integration tests
pytest tests/integration/ -v

# Test full pipeline
python examples/full_pipeline.py
```

---

## Troubleshooting

### API won't start
```bash
# Check logs
docker compose logs sigmalang

# Check port availability
lsof -i :26080  # macOS/Linux
netstat -ano | findstr :26080  # Windows

# Rebuild image
docker compose build --no-cache sigmalang
```

### Redis connection errors
```bash
# Verify Redis is running
docker compose ps redis

# Check Redis health
docker compose exec redis redis-cli ping

# Restart Redis
docker compose restart redis
```

### Tests fail with timeout
```bash
# Increase timeout (default is 300s)
pytest tests/ --timeout=600

# Run specific tests only
pytest tests/test_optimizations.py -v
```

### Memory issues
```bash
# Reduce Docker memory limit if needed
# Edit docker-compose.yml and adjust:
# resources.limits.memory: 1G

# Or run tests with reduced batch size
pytest tests/ -v --tb=short
```

---

## Production Readiness Checklist

- [x] All 1,656 tests passing
- [x] CI/CD pipeline configured and hardened
- [x] Docker image optimized (<500MB)
- [x] Health checks configured
- [x] Metrics and monitoring set up
- [x] Rate limiting enabled
- [x] Graceful shutdown implemented
- [ ] Security audit completed
- [ ] Load testing completed
- [ ] Documentation site built

---

## Next Steps for Release

1. **Security Hardening** - Scan for credentials, fix OWASP issues
2. **Load Testing** - Run Locust tests to validate performance
3. **Documentation** - Build MkDocs site for users
4. **Publication** - Push to PyPI and Docker Hub
5. **Helm Chart** - Create Kubernetes deployment files

---

## Support & Debugging

### Enable Debug Logging
```bash
export SIGMALANG_LOG_LEVEL=DEBUG
sigmalang-server
```

### View Metrics
- Prometheus: http://localhost:26900
- Grafana: http://localhost:26910

### Check Health Status
```bash
curl http://localhost:26080/health/detailed
```

### Performance Metrics
```bash
curl http://localhost:26080/metrics
```

---

**Last Updated:** February 19, 2026
**Project Status:** 97-98% Complete, 1,656/1656 Tests Passing ✅
