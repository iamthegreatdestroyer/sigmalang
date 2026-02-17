# Load Testing Guide

## Prerequisites

```bash
pip install locust
```

## Running Load Tests

### Option 1: Headless Mode (Automated)

```bash
# Start the API server (if available)
python -m uvicorn sigmalang.core.api_server:app --host 0.0.0.0 --port 8000 &

# Run 5-minute load test with 100 concurrent users
locust -f scripts/locustfile.py --headless -u 100 -r 10 --run-time 5m --host http://localhost:8000
```

### Option 2: Web UI Mode (Interactive)

```bash
# Start Locust web interface
locust -f scripts/locustfile.py --host http://localhost:8000

# Open browser to http://localhost:8089
# Configure users and spawn rate in the web UI
```

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| P95 Latency | < 100ms | 95th percentile response time |
| Error Rate | < 0.1% | Failed requests / total requests |
| Throughput | > 200 req/s | Requests per second |
| Memory | No leaks | Stable over 5-minute run |

## Load Distribution

The load test simulates realistic user behavior:

- 50% encode operations
- 20% search operations
- 15% analogy operations
- 10% entity extraction
- 5% health checks

## Notes

**Important:** The current API server implementation is a service layer without HTTP endpoints.
To run real load tests, you need to:

1. Implement FastAPI wrapper around SigmalangAPI services, OR
2. Use the service layer tests instead of HTTP load tests

For now, this locustfile serves as a template for when HTTP endpoints are implemented.

## Alternative: Service-Level Load Testing

Since HTTP endpoints aren't implemented yet, you can test service performance directly:

```python
from sigmalang.core.api_server import create_api
from sigmalang.core.api_models import EncodeRequest
import time

api = create_api()
api.initialize()

# Measure encode performance
start = time.time()
for i in range(1000):
    request = EncodeRequest(text=f"Test text {i}")
    response = api.encoder.encode(request)
duration = time.time() - start

print(f"Throughput: {1000/duration:.2f} req/s")
print(f"Avg latency: {duration/1000*1000:.2f}ms")
```
