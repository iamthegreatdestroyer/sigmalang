# Logging & Troubleshooting

## Logging Configuration

### Log Levels

| Level | Verbosity | Use Case |
|-------|-----------|----------|
| DEBUG | Very high | Development and troubleshooting |
| INFO | Normal | Production standard |
| WARNING | Minimal | Warnings only |
| ERROR | Errors only | Critical issues |

### Set Log Level

```bash
# Environment variable
export SIGMALANG_LOG_LEVEL=DEBUG

# Command line
sigmalang-server --log-level DEBUG

# Docker
docker run -e SIGMALANG_LOG_LEVEL=DEBUG sigmalang/sigmalang

# Kubernetes
kubectl set env deployment/sigmalang SIGMALANG_LOG_LEVEL=DEBUG -n sigmalang
```

## Log Locations

### Docker Compose

```bash
docker compose logs -f sigmalang
```

### Kubernetes

```bash
kubectl logs -f deployment/sigmalang -n sigmalang
```

### Local Installation

```bash
# stdout (default)
sigmalang-server

# File output
export SIGMALANG_LOG_FILE=/var/log/sigmalang/app.log
sigmalang-server
```

## Log Format

```
[TIMESTAMP] LEVEL MODULE: MESSAGE

Example:
[2026-02-19 12:00:00.123] INFO sigmalang.api.encode: Processing request
[2026-02-19 12:00:00.125] DEBUG sigmalang.core.parser: Tree has 5 nodes
[2026-02-19 12:00:00.126] INFO sigmalang.api.encode: Response 200 OK
```

## Structured Logging

Enable JSON logging for easier parsing:

```bash
export SIGMALANG_LOG_FORMAT=json
```

Output:
```json
{
  "timestamp": "2026-02-19T12:00:00.123Z",
  "level": "INFO",
  "module": "sigmalang.api.encode",
  "message": "Processing request",
  "request_id": "uuid-here",
  "duration_ms": 2.5
}
```

## Common Issues and Solutions

### API Won't Start

**Error:**
```
[ERROR] Failed to bind to port 8000
```

**Solution:**
```bash
# Check if port is in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
export SIGMALANG_API_PORT=8001
sigmalang-server
```

### Redis Connection Error

**Error:**
```
[ERROR] Cannot connect to Redis at localhost:6379
```

**Solution:**
```bash
# Check Redis is running
docker compose ps redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Restart Redis
docker compose restart redis

# Check connection string
export SIGMALANG_REDIS_URL=redis://localhost:6379/0
```

### Out of Memory

**Error:**
```
[ERROR] Memory allocation failed
```

**Solution:**
```bash
# Check current memory usage
free -h  # Linux
vm_stat  # macOS

# Increase allocated memory
docker update --memory 4G container_name

# Reduce batch size
sigmalang encode -i large.txt --batch-size 100

# Enable streaming mode
export SIGMALANG_STREAMING_MODE=true
```

### Timeout on Large Files

**Error:**
```
[ERROR] Request timeout after 300s
```

**Solution:**
```bash
# Increase timeout
export SIGMALANG_REQUEST_TIMEOUT=600

# Process in batches
sigmalang encode -i huge_file.txt --optimization low --batch-size 50

# Use streaming encoding
python stream_encoder.py < huge_file.txt
```

### Cache Inconsistency

**Error:**
```
[WARNING] Cache hit returned stale data
```

**Solution:**
```bash
# Clear cache
docker compose exec redis redis-cli FLUSHALL

# Disable caching temporarily
export SIGMALANG_CACHE_ENABLED=false

# Investigate cache key
redis-cli --scan --pattern "sigmalang:*"
```

### Performance Degradation

**Symptom:** Requests getting slower over time

**Solution:**
```bash
# Check memory usage
curl http://localhost:26080/health/detailed | jq '.memory'

# Check cache hit rate
curl http://localhost:26080/metrics | grep cache_hit_ratio

# Restart service
docker compose restart sigmalang

# Review slow queries
export SIGMALANG_LOG_LEVEL=DEBUG
# Then check logs for slow operations
```

### High CPU Usage

**Symptom:** CPU usage stuck at 100%

**Solutions:**
```bash
# Identify hot code
python -m cProfile -s cumulative sigmalang-server

# Reduce optimization level
export SIGMALANG_OPTIMIZATION=low

# Reduce number of workers
export SIGMALANG_API_WORKERS=2

# Check for infinite loops in logs
docker compose logs sigmalang | grep -i loop
```

## Debug Mode

### Enable Debug Logging

```bash
export SIGMALANG_DEBUG=true
export SIGMALANG_LOG_LEVEL=DEBUG
sigmalang-server
```

### Debug CLI Commands

```bash
# Verbose output
sigmalang encode "text" --verbose

# Show timing
sigmalang encode "text" --benchmark

# Show parsed tree
sigmalang encode "text" --show-tree
```

### Interactive Debugging

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser

# Enable debugging
import logging
logging.basicConfig(level=logging.DEBUG)

parser = SemanticParser()
tree = parser.parse("test text")
# Now see debug output for parsing steps
```

## Log Analysis

### Find Errors

```bash
# Docker
docker compose logs sigmalang | grep ERROR

# File
grep ERROR /var/log/sigmalang/app.log

# Real-time
tail -f /var/log/sigmalang/app.log | grep ERROR
```

### Find Slow Operations

```bash
# Operations taking >5ms
grep -E "duration.*[5-9]ms|duration.*[0-9]{2,}ms" /var/log/sigmalang/app.log

# Summary
grep "duration_ms" /var/log/sigmalang/app.log | \
  awk '{print $NF}' | sort -n | tail -10
```

### Count Operations

```bash
# Count by operation type
grep "^\\[INFO\\]" /var/log/sigmalang/app.log | \
  awk '{print $4}' | sort | uniq -c | sort -rn

# Requests per minute
grep "^\\[INFO\\].*request" /var/log/sigmalang/app.log | \
  awk '{print $1}' | uniq -c
```

## Centralized Logging

### ELK Stack Setup

```yaml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    environment:
      - discovery.type=single-node

  kibana:
    image: docker.elastic.co/kibana/kibana:7.14.0
    ports:
      - "5601:5601"

  filebeat:
    image: docker.elastic.co/beats/filebeat:7.14.0
    volumes:
      - /var/log/sigmalang:/var/log/sigmalang
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml
```

### Loki Stack (Kubernetes)

```bash
helm repo add loki https://grafana.github.io/loki/charts
helm install loki loki/loki-stack \
  -n monitoring \
  --create-namespace
```

## Best Practices

1. **Use appropriate log level** for environment
2. **Include request IDs** for tracing
3. **Log at decision points** (not every line)
4. **Use structured logging** for easy parsing
5. **Rotate logs** to prevent disk space issues
6. **Monitor log files** for unusual patterns
7. **Archive old logs** for compliance

## Log Rotation

### Linux/macOS

Create `/etc/logrotate.d/sigmalang`:

```
/var/log/sigmalang/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 sigmalang sigmalang
    sharedscripts
    postrotate
        systemctl reload sigmalang > /dev/null 2>&1 || true
    endscript
}
```

## Next Steps

- Configure [Monitoring](monitoring.md)
- Optimize [Performance](performance.md)
- Review [Operations](../operations/monitoring.md)
