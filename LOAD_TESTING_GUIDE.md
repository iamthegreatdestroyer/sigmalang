# Load Testing Guide - ΣLANG

## Overview

This guide covers load testing ΣLANG to verify performance and stability under realistic production conditions.

## Prerequisites

```bash
# Install Locust
pip install locust

# Ensure ΣLANG is running
docker compose up -d

# Verify API is accessible
curl http://localhost:26080/health
```

## Quick Start

### Basic Load Test (10 users, 1 user/sec spawn rate)

```bash
# Terminal 1: Start API server
docker compose up -d

# Terminal 2: Run load test with UI
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 10 \
  --spawn-rate 1 \
  --run-time 5m

# Then open http://localhost:8089 in browser
```

### CLI Mode (Headless)

```bash
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 50 \
  --spawn-rate 5 \
  --run-time 10m \
  --headless
```

## Test Scenarios

### 1. Baseline Performance (Development Verification)

**Configuration:**
```bash
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 5 \
  --spawn-rate 1 \
  --run-time 3m \
  --headless
```

**Expected Results:**
- Success rate: >99%
- Avg encoding: <20ms
- P95 encoding: <50ms
- Error count: 0-1

### 2. Normal Load (Typical Production)

**Configuration:**
```bash
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 50 \
  --spawn-rate 5 \
  --run-time 10m \
  --headless
```

**Expected Results:**
- Success rate: >99%
- Avg encoding: <30ms
- P95 encoding: <100ms
- Throughput: >1,000 req/min

### 3. Peak Load (High Concurrent Users)

**Configuration:**
```bash
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 200 \
  --spawn-rate 10 \
  --run-time 15m \
  --headless
```

**Expected Results:**
- Success rate: >95%
- Avg encoding: 50-100ms
- P99 encoding: <500ms
- System maintains stability

### 4. Spike Test (Sudden Traffic Surge)

**Configuration:**
```bash
# Ramp up quickly to peak
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 100 \
  --spawn-rate 50 \
  --run-time 5m \
  --headless
```

**Measures:**
- How quickly system adapts
- Queue handling
- Error recovery

### 5. Endurance Test (Long-running)

**Configuration:**
```bash
locust -f load_test.py \
  --host=http://localhost:26080 \
  --users 30 \
  --spawn-rate 3 \
  --run-time 1h \
  --headless
```

**Measures:**
- Memory leaks
- Connection pool issues
- Cache performance degradation
- Long-term stability

## Understanding Results

### Key Metrics

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|-----------|------|
| Success Rate | >99.9% | >99% | >95% | <95% |
| P95 Latency | <50ms | <100ms | <500ms | >500ms |
| P99 Latency | <100ms | <200ms | <1000ms | >1000ms |
| Error Rate | 0% | <0.1% | <1% | >1% |

### Response Distribution

Look for:
- **Consistent response times** - Good sign
- **Bimodal distribution** - May indicate cache hits vs misses
- **Long tail** - Some requests slow (investigate)
- **Exponential growth** - System struggling under load

### Error Types

- **Connection refused** - Server crashed or overloaded
- **Timeout** - Server too slow responding
- **500 errors** - Application errors
- **429 errors** - Rate limiting active

## Interpreting the Report

### Success Indicators
```
✅ Success Rate: 99.5%
✅ Encoding Performance: Avg 15ms, P95 42ms
✅ No errors in 500+ requests
✅ Memory stable throughout test
```

### Warning Signs
```
⚠️  Errors increasing over time (potential leak)
⚠️  Response times growing (cache eviction?)
⚠️  Success rate dropping (resource exhaustion)
⚠️  Error spike at specific concurrency level
```

### Failure Indicators
```
❌ Success rate <95%
❌ Many connection refused errors
❌ Memory keeps growing (leak)
❌ Cannot reach target user count
```

## Custom Test Configuration

Edit `load_test.py` to customize:

### Change request distribution
```python
@task(50)  # 50% of requests
def encode_text(self):
    ...

@task(25)  # 25% of requests
def extract_entities(self):
    ...
```

### Add new test scenarios
```python
@task(10)
def my_custom_test(self):
    with self.client.post(
        "/api/endpoint",
        json={"param": "value"},
        catch_response=True
    ) as response:
        if response.status_code == 200:
            response.success()
        else:
            response.failure(f"Error {response.status_code}")
```

### Modify test data
```python
SAMPLE_TEXTS = [
    "Your custom test data here",
    "More test examples",
]
```

## Performance Baselines

### Development Machine (Local Testing)

```
CPU: 4 cores, 8GB RAM
Configuration: Medium optimization

Baseline Results:
- Encoding: 10-20ms (single request)
- Batch encoding: 30-50ms (5 items)
- Entity extraction: 15-25ms
- Analogy solving: 20-30ms
- Throughput: 1,000+ req/min
- Success rate: 100% (no load)
```

### Production Server (Expected)

```
CPU: 2 cores, 2GB RAM (docker-compose)
Configuration: Medium optimization

Expected Results:
- Encoding: 5-15ms (with caching)
- Batch encoding: 20-40ms (5 items)
- Entity extraction: 10-20ms
- Analogy solving: 15-25ms
- Throughput: 500-1,000 req/min under load
- Success rate: >99% at 50 concurrent users
```

## Troubleshooting

### All requests fail immediately

```bash
# Check API is running
curl http://localhost:26080/health

# Check Docker compose
docker compose ps

# View API logs
docker compose logs sigmalang
```

### Response times increasing over time

Indicates potential memory leak or resource exhaustion:

```bash
# Monitor memory during test
watch -n 1 'docker stats --no-stream'

# Check for leaks
docker compose logs sigmalang | grep -i "error\|memory"
```

### Intermittent failures

Check for rate limiting or timeout issues:

```bash
# Increase request timeout (edit load_test.py)
connection_timeout = 10  # seconds

# Check rate limit settings
curl http://localhost:26080/health/detailed | jq '.rate_limit'
```

### Cannot reach target concurrency

System may be bottlenecked:

```bash
# Reduce spawn rate
--spawn-rate 2  # Slower ramp-up

# Check available resources
docker stats

# Increase resource limits in docker-compose.yml
resources:
  limits:
    cpus: '4'
    memory: 4G
```

## Continuous Load Testing

### Scheduled Tests

```bash
# Run daily baseline test at 2 AM
0 2 * * * locust -f load_test.py --host=http://localhost:26080 --users 50 --spawn-rate 5 --run-time 30m --headless --csv=results_$(date +\%Y\%m\%d).csv
```

### CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/load-test.yml
name: Load Testing

on:
  - push
  - schedule:
      - cron: '0 2 * * *'  # Daily 2 AM

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start services
        run: docker compose up -d
      - name: Wait for service
        run: sleep 10
      - name: Run load test
        run: |
          pip install locust
          locust -f load_test.py \
            --host=http://localhost:26080 \
            --users 50 \
            --spawn-rate 5 \
            --run-time 10m \
            --headless \
            --csv=results.csv
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: load-test-results
          path: results.csv
```

## Reporting Results

### Sample Report

```
ΣLANG Load Test Report
Date: 2026-02-19
Duration: 10 minutes
Target: http://localhost:26080

Configuration:
- Users: 50 concurrent
- Spawn rate: 5 users/sec
- Request distribution: 40% encode, 20% entities, 15% analogy, 15% batch, 10% health

Results:
- Total requests: 5,234
- Successful: 5,210 (99.54%)
- Failed: 24 (0.46%)

Performance:
- Encoding: Avg 18ms, P95 52ms, P99 125ms
- Entity extraction: Avg 22ms, P95 68ms
- Analogy solving: Avg 25ms, P95 75ms
- Batch encoding: Avg 35ms, P95 95ms

Resource Usage:
- CPU: 0.8-1.2 cores
- Memory: 800MB-950MB
- Network: 2.4 Mbps avg

Recommendation: ✅ APPROVED FOR PRODUCTION
- All metrics within acceptable ranges
- No errors under sustained load
- Resource usage reasonable
- Response times consistent
```

## Next Steps

1. **Baseline Test** - Run baseline to understand normal performance
2. **Peak Load Test** - Verify system handles 2-3x normal load
3. **Stress Test** - Find breaking point
4. **Soak Test** - Run long test to find resource leaks
5. **Document Results** - Create baseline document
6. **Automate Testing** - Add to CI/CD pipeline
7. **Monitor Production** - Compare production metrics to baseline

## Resources

- [Locust Documentation](https://docs.locust.io/)
- [Performance Testing Best Practices](https://en.wikipedia.org/wiki/Software_performance_testing)
- [Load Testing Tools Comparison](https://www.sitepoint.com/open-source-load-testing-tools-jmeter-locust-gatling/)

---

**Last Updated:** February 19, 2026
**Project:** ΣLANG Load Testing
**Status:** Production ready
