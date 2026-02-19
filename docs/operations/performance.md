# Performance Optimization

## Benchmarking

### Built-in Benchmarks

```bash
# Run encoding benchmark
sigmalang encode "text" --benchmark

# Output
# Encoding time: 1.2ms
# Throughput: 36.6 MB/s
# Compression: 4.2x
```

### Measure Compression

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.parser import SemanticParser
import time

parser = SemanticParser()
encoder = SigmaEncoder()

text = "A" * 1000000  # 1MB

start = time.time()
tree = parser.parse(text)
encoded = encoder.encode(tree)
elapsed = time.time() - start

print(f"Throughput: {len(text) / elapsed / 1e6:.1f} MB/s")
print(f"Compression: {len(text) / len(encoded):.1f}x")
```

## Optimization Levels

### Low (Fast)

```bash
# CLI
sigmalang encode -i input.txt --optimization low

# Python
config = SigmaConfig(optimization_level="low")
encoder = SigmaEncoder(config=config)
```

**Characteristics:**
- Throughput: ~10 MB/s
- Compression: 5-8x
- Use for: Streaming, real-time, logging

### Medium (Balanced, Default)

```bash
sigmalang encode -i input.txt --optimization medium
```

**Characteristics:**
- Throughput: ~5 MB/s
- Compression: 10-20x
- Use for: General purpose

### High (Maximum)

```bash
sigmalang encode -i input.txt --optimization high
```

**Characteristics:**
- Throughput: ~1 MB/s
- Compression: 20-50x
- Use for: Offline, batch processing

## Caching Strategy

### Enable Caching

```bash
export SIGMALANG_CACHE_ENABLED=true
export SIGMALANG_CACHE_TTL=3600  # 1 hour
```

### Check Cache Performance

```bash
curl http://localhost:26080/metrics | grep cache_hit_ratio
```

### Tune Cache Size

```bash
# Increase cache size
export SIGMALANG_CACHE_MAX_SIZE=1000000

# Adjust TTL based on usage pattern
export SIGMALANG_CACHE_TTL=7200  # 2 hours for static content
```

## Batch Processing

### Batch Encoding

```python
from sigmalang.core.processor import BatchProcessor

texts = [f"Document {i}" for i in range(1000)]

processor = BatchProcessor(batch_size=100)
results = processor.encode_batch(texts)

# Faster than individual encoding
```

### Optimal Batch Size

```python
# Test different batch sizes
for batch_size in [10, 50, 100, 500, 1000]:
    processor = BatchProcessor(batch_size=batch_size)
    start = time.time()
    results = processor.encode_batch(texts)
    elapsed = time.time() - start
    print(f"Batch {batch_size}: {elapsed:.2f}s")
```

## Streaming Processing

### Stream Large Files

```python
from sigmalang.core.processor import StreamingProcessor

processor = StreamingProcessor(chunk_size=10000)

with open("large_file.txt") as f:
    for encoded_chunk in processor.encode_stream(f):
        send_to_network(encoded_chunk)
```

### Memory-Efficient Decoding

```python
processor = StreamingProcessor()

with open("encoded.bin", "rb") as f:
    for decoded_chunk in processor.decode_stream(f):
        process_chunk(decoded_chunk)
```

## Resource Management

### Memory Usage

```bash
# Monitor memory
watch -n 1 'docker stats sigmalang'

# Set memory limit
docker update --memory 2G sigmalang_sigmalang_1
```

### CPU Optimization

```bash
# Set CPU limit
docker update --cpus 2 sigmalang_sigmalang_1

# Set worker count to match CPUs
export SIGMALANG_API_WORKERS=4  # for 4-core machine
```

### Buffer Pool Tuning

```bash
# Increase pool size for better reuse
export SIGMALANG_BUFFER_POOL_SIZE=500
export SIGMALANG_BUFFER_SIZE=8192
```

## Network Optimization

### Compression for Transmission

```python
import gzip

encoded = encoder.encode(tree)
compressed = gzip.compress(encoded)

# Send over network
requests.post(url, data=compressed,
              headers={'Content-Encoding': 'gzip'})
```

### Batch API Requests

```bash
# Better: Single request with multiple items
curl -X POST http://localhost:26080/api/encode/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["text1", "text2", ...]}'

# Avoid: Multiple requests
for text in texts:
    curl -X POST http://localhost:26080/api/encode \
      -d "{\"text\": \"$text\"}"
```

## Database Query Optimization

### Use Indexes

```python
# Create index for fast lookup
index = SemanticIndex()
index.add(doc_id, encoded_text)
index.create_index()

# Query indexed data
results = index.search("query")
```

### Query Patterns

```python
# Efficient: Single query
results = index.search("query", limit=10)

# Avoid: Multiple queries in loop
for term in terms:
    result = index.search(term)
```

## Profiling

### CPU Profiling

```bash
python -m cProfile -s cumulative -o profile.stats sigmalang-server
python -m pstats profile.stats
```

### Memory Profiling

```bash
pip install memory-profiler

python -m memory_profiler script.py
```

### Flame Graphs

```bash
pip install py-spy
py-spy record -o profile.svg -- sigmalang-server
```

## Benchmarking Suite

### Run Full Benchmarks

```bash
pytest tests/ -k benchmark -v
```

### Benchmark Results

```
test_encode_small_text ... 1.2ms ✓
test_encode_medium_text ... 5.3ms ✓
test_encode_large_text ... 45.2ms ✓
test_decode_small ... 0.8ms ✓
test_decode_large ... 32.1ms ✓
```

## Performance Monitoring

### Real-time Monitoring

```bash
# Watch metrics
watch -n 1 'curl -s http://localhost:26080/metrics | grep sigmalang_'
```

### Identify Bottlenecks

```promql
# Slowest endpoints
topk(5, rate(sigmalang_request_duration_ms_bucket[5m]))

# Highest error rate
topk(5, rate(sigmalang_request_errors_total[5m]))

# Lowest cache hit rate
bottomk(5, sigmalang_cache_hit_ratio)
```

## Performance Targets

### SLOs (Service Level Objectives)

| Metric | Target |
|--------|--------|
| P95 Latency | <10ms |
| P99 Latency | <50ms |
| Error Rate | <0.1% |
| Cache Hit Ratio | >90% |
| Throughput | >1MB/s |

### Track Performance

```python
# Log metrics periodically
import time
from prometheus_client import Gauge

performance = Gauge('encoding_performance', 'Performance metrics')

start = time.time()
encoded = encoder.encode(tree)
elapsed = time.time() - start

performance.labels(metric='latency').set(elapsed * 1000)
performance.labels(metric='throughput').set(len(tree.text) / elapsed)
```

## Troubleshooting Performance

### Slow Encoding

```bash
# Check optimization level
env | grep SIGMALANG_OPTIMIZATION

# Try lower level
export SIGMALANG_OPTIMIZATION=low

# Check for GC pressure
docker stats sigmalang | grep MEM%
```

### High Memory Usage

```bash
# Reduce buffer pool size
export SIGMALANG_BUFFER_POOL_SIZE=50

# Reduce batch size
batch_size = 10  # was 100

# Check for memory leaks
python -m memory_profiler script.py
```

### Cache Misses

```bash
# Increase cache size
export SIGMALANG_CACHE_MAX_SIZE=1000000

# Increase TTL
export SIGMALANG_CACHE_TTL=7200

# Monitor hit rate
curl http://localhost:26080/metrics | grep cache_hit_ratio
```

## Best Practices

1. **Choose right optimization level** for your use case
2. **Use batch processing** for multiple items
3. **Enable caching** for repeated operations
4. **Monitor metrics** continuously
5. **Profile regularly** to find bottlenecks
6. **Test with realistic data** sizes
7. **Set appropriate SLOs** for your service

## Next Steps

- Configure [Monitoring](monitoring.md)
- Set up [Logging](logging.md)
- Deploy with [Docker](../deployment/docker.md)
