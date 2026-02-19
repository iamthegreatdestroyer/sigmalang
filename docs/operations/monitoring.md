# Monitoring & Observability

## Overview

ΣLANG includes built-in monitoring via Prometheus metrics and Grafana dashboards for real-time visibility into system performance.

## Prometheus Metrics

### Enabled by Default

```bash
# Verify metrics endpoint
curl http://localhost:26080/metrics
```

### Key Metrics

#### Encoding Performance

```
sigmalang_encode_duration_ms{quantile="0.95"} 2.5
sigmalang_encode_duration_ms{quantile="0.99"} 5.0
sigmalang_compression_ratio 15.0
sigmalang_encode_throughput_bytes_per_sec 1000000
```

#### Decoding Performance

```
sigmalang_decode_duration_ms{quantile="0.95"} 1.5
sigmalang_decode_throughput_bytes_per_sec 2000000
```

#### Request Metrics

```
sigmalang_request_count_total{method="POST",endpoint="/api/encode"} 10000
sigmalang_request_errors_total{method="POST",endpoint="/api/encode"} 5
sigmalang_request_duration_ms{quantile="0.95"} 3.0
```

#### Resource Metrics

```
sigmalang_memory_usage_bytes 524288000
sigmalang_buffer_pool_utilization 0.85
sigmalang_cache_hit_ratio 0.92
```

### Custom Queries

#### Average Compression Ratio

```promql
rate(sigmalang_compression_ratio[5m])
```

#### Error Rate

```promql
rate(sigmalang_request_errors_total[5m]) / rate(sigmalang_request_count_total[5m])
```

#### P95 Latency

```promql
histogram_quantile(0.95, sigmalang_encode_duration_ms_bucket)
```

#### Throughput

```promql
rate(sigmalang_bytes_encoded_total[1m])
```

## Grafana Dashboards

### Access Grafana

```bash
# URL
http://localhost:26910

# Credentials
Username: admin
Password: sigmalang
```

### Pre-built Dashboards

1. **System Overview**: CPU, memory, network
2. **Application Performance**: Request rates, latencies, errors
3. **Compression Metrics**: Ratio, throughput, file sizes
4. **Cache Performance**: Hit rates, misses, evictions
5. **Resource Usage**: Memory pools, buffer allocation

### Create Custom Dashboard

1. Click "+" → "Create Dashboard"
2. Add panel with PromQL query
3. Select visualization type
4. Save dashboard

### Example Dashboard JSON

```json
{
  "dashboard": {
    "title": "ΣLANG Performance",
    "panels": [
      {
        "title": "Compression Ratio",
        "targets": [
          {
            "expr": "sigmalang_compression_ratio"
          }
        ]
      },
      {
        "title": "Encoding Latency P95",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sigmalang_encode_duration_ms_bucket)"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Configure Alerts

Create `prometheus-rules.yaml`:

```yaml
groups:
  - name: sigmalang
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(sigmalang_request_errors_total[5m]) /
          rate(sigmalang_request_count_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: High error rate

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, sigmalang_encode_duration_ms_bucket) > 10
        for: 5m
        annotations:
          summary: High encoding latency

      # Low cache hit rate
      - alert: LowCacheHitRate
        expr: sigmalang_cache_hit_ratio < 0.7
        for: 10m
        annotations:
          summary: Low cache hit rate

      # High memory usage
      - alert: HighMemoryUsage
        expr: sigmalang_memory_usage_bytes / sigmalang_memory_limit_bytes > 0.9
        for: 5m
        annotations:
          summary: Memory usage above 90%
```

### Add to Prometheus

```bash
# Update prometheus.yml
global:
  scrape_interval: 15s

rule_files:
  - prometheus-rules.yaml

scrape_configs:
  - job_name: sigmalang
    static_configs:
      - targets: ['localhost:8000']
```

## Health Checks

### Liveness Probe

```bash
curl http://localhost:26080/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-19T12:00:00Z"
}
```

### Readiness Probe

```bash
curl http://localhost:26080/health/ready
```

Indicates if service can accept traffic.

### Detailed Status

```bash
curl http://localhost:26080/health/detailed
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "redis": "connected",
    "database": "connected",
    "memory": "normal"
  },
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

## Logging

### Log Levels

```bash
export SIGMALANG_LOG_LEVEL=DEBUG    # Verbose
export SIGMALANG_LOG_LEVEL=INFO     # Default
export SIGMALANG_LOG_LEVEL=WARNING  # Warnings only
export SIGMALANG_LOG_LEVEL=ERROR    # Errors only
```

### View Logs

```bash
# Docker
docker compose logs -f sigmalang

# Kubernetes
kubectl logs -f deployment/sigmalang -n sigmalang

# File
tail -f /var/log/sigmalang/app.log
```

### Log Format

```
[2026-02-19 12:00:00.123] INFO sigmalang.api.encode: Encoding request
[2026-02-19 12:00:00.125] DEBUG sigmalang.core.parser: Parsed 44 bytes
[2026-02-19 12:00:00.126] INFO sigmalang.api.encode: Compression 2.9x
```

## Performance Analysis

### Slow Query Detection

```promql
# Queries taking >5ms
histogram_quantile(0.95,
  rate(sigmalang_encode_duration_ms_bucket[5m])
) > 5
```

### Throughput Analysis

```promql
# Bytes encoded per second
rate(sigmalang_bytes_encoded_total[1m])
```

### Compression Efficiency

```promql
# Average compression ratio
avg(sigmalang_compression_ratio) by (optimization_level)
```

## Profiling

### Enable Profiling

```bash
export SIGMALANG_PROFILING=true
export SIGMALANG_PROFILE_DIR=/tmp/profiles
```

### View Profiles

```python
import pstats
stats = pstats.Stats('/tmp/profiles/encode.prof')
stats.sort_stats('cumulative').print_stats(20)
```

## Integration Examples

### Python Client Monitoring

```python
from prometheus_client import Counter, Histogram
import time

encode_time = Histogram(
    'encode_duration_ms',
    'Encoding duration'
)

@encode_time.time()
def encode_and_monitor(text):
    return encoder.encode(text)
```

### Send Custom Metrics

```python
from prometheus_client import start_http_server, Gauge

compression_ratio = Gauge(
    'custom_compression_ratio',
    'Custom compression ratio'
)

encoded = encoder.encode(text)
compression_ratio.set(len(text) / len(encoded))
```

## Best Practices

1. **Monitor in production**: Always enable Prometheus metrics
2. **Set appropriate alerts**: Alert on meaningful thresholds
3. **Review regularly**: Check dashboards daily for anomalies
4. **Retain metrics**: Keep at least 15 days of data
5. **Document dashboards**: Add descriptions to dashboard panels

## Troubleshooting

### Metrics not appearing

```bash
# Check metrics endpoint
curl http://localhost:26080/metrics

# Enable metrics
export SIGMALANG_METRICS_ENABLED=true

# Verify Prometheus config
curl http://localhost:26900/api/v1/targets
```

### High cardinality metrics

```promql
# Check cardinality
count(count(sigmalang_request_count_total) by (__name__))
```

Reduce by removing high-cardinality labels.

## Next Steps

- Configure [Logging](logging.md)
- Optimize [Performance](performance.md)
- Deploy with [Docker](../deployment/docker.md)
