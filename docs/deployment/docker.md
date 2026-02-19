# Docker Deployment

## Quick Start

```bash
# Clone repository
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang

# Start services
docker compose up -d

# Verify
docker compose ps

# View logs
docker compose logs -f sigmalang
```

## Available Services

| Service | Port | Purpose |
|---------|------|---------|
| ΣLANG API | 26080 | Main API server |
| Redis | 26500 | Cache backend |
| Prometheus | 26900 | Metrics collection |
| Grafana | 26910 | Monitoring dashboard |

## Accessing Services

### API Server

```bash
# Health check
curl http://localhost:26080/health

# API Documentation
open http://localhost:26080/docs

# Make request
curl -X POST http://localhost:26080/api/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, World!"}'
```

### Prometheus Metrics

```bash
open http://localhost:26900
```

Useful queries:
```
sigmalang_encode_duration_ms
sigmalang_compression_ratio
sigmalang_request_count_total
```

### Grafana Dashboards

```bash
# Access Grafana
open http://localhost:26910

# Login
Username: admin
Password: sigmalang
```

## Docker Compose Configuration

### File Structure

```yaml
version: '3.8'

services:
  sigmalang:
    build: .
    ports:
      - "26080:8000"
    environment:
      SIGMALANG_CACHE_BACKEND: redis
      SIGMALANG_REDIS_URL: redis://redis:6379/0
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "26500:6379"

  prometheus:
    image: prom/prometheus
    ports:
      - "26900:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "26910:3000"
```

## Environment Variables

### API Configuration

```bash
SIGMALANG_API_HOST=0.0.0.0          # Bind address
SIGMALANG_API_PORT=8000              # Listen port
SIGMALANG_API_WORKERS=4              # Worker processes
SIGMALANG_DEBUG=false                # Debug mode
```

### Caching

```bash
SIGMALANG_CACHE_BACKEND=redis        # redis or memory
SIGMALANG_REDIS_URL=redis://localhost:6379/0
SIGMALANG_CACHE_ENABLED=true
SIGMALANG_CACHE_TTL=3600             # Cache TTL in seconds
```

### Monitoring

```bash
SIGMALANG_METRICS_ENABLED=true
SIGMALANG_METRICS_PORT=9090
SIGMALANG_LOG_LEVEL=INFO
```

## Building Custom Images

### Build without cache

```bash
docker compose build --no-cache sigmalang
```

### Build specific service

```bash
docker compose build sigmalang
```

## Volume Management

### Persistent Storage

```bash
# Create volume
docker volume create sigmalang-data

# Mount in compose
volumes:
  - sigmalang-data:/data
```

### View volumes

```bash
docker volume ls
docker volume inspect sigmalang-data
```

## Health Checks

### Check service health

```bash
# HTTP health check
curl http://localhost:26080/health

# Detailed status
curl http://localhost:26080/health/detailed

# Docker health check
docker compose ps sigmalang
```

## Logs and Debugging

### View logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs sigmalang

# Follow logs
docker compose logs -f sigmalang

# Last 100 lines
docker compose logs --tail 100 sigmalang
```

### Enable debug logging

```bash
docker compose exec sigmalang \
  /bin/sh -c "export SIGMALANG_LOG_LEVEL=DEBUG && sigmalang-server"
```

## Performance Tuning

### Resource Limits

```yaml
services:
  sigmalang:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Optimize workers

```bash
export SIGMALANG_API_WORKERS=8  # Match CPU cores
docker compose up -d
```

## Scaling

### Horizontal scaling with load balancer

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "26080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - sigmalang_1
      - sigmalang_2

  sigmalang_1:
    build: .
    environment:
      SIGMALANG_API_PORT: 8001

  sigmalang_2:
    build: .
    environment:
      SIGMALANG_API_PORT: 8002
```

## Networking

### Connect to services from host

```bash
# Direct connection
curl http://localhost:26080/health

# Via docker network
docker compose exec sigmalang curl http://redis:6379
```

### Custom network

```yaml
networks:
  backend:
    driver: bridge

services:
  sigmalang:
    networks:
      - backend

  redis:
    networks:
      - backend
```

## Backup and Restore

### Backup data

```bash
docker compose exec redis \
  redis-cli BGSAVE

docker cp sigmalang_redis_1:/data/dump.rdb ./backup/
```

### Restore data

```bash
docker cp ./backup/dump.rdb sigmalang_redis_1:/data/
docker compose restart redis
```

## Troubleshooting

### Port already in use

```bash
# Find process using port
lsof -i :26080

# Change port in docker-compose.yml
ports:
  - "26081:8000"  # Changed from 26080
```

### Redis connection errors

```bash
# Check Redis
docker compose exec redis redis-cli ping

# Restart Redis
docker compose restart redis

# Check logs
docker compose logs redis
```

### API won't start

```bash
# Check logs
docker compose logs sigmalang

# Rebuild image
docker compose build --no-cache sigmalang

# Restart
docker compose restart sigmalang
```

### Out of memory

```bash
# Increase memory limit
docker update --memory 4G sigmalang_sigmalang_1

# Or update docker-compose.yml
# Restart services
docker compose down
docker compose up -d
```

## Production Deployment

### Security hardening

```yaml
services:
  sigmalang:
    # Don't use root
    user: sigmalang:sigmalang

    # Read-only filesystem
    read_only: true

    # Drop capabilities
    cap_drop:
      - ALL

    # Security options
    security_opt:
      - no-new-privileges:true
```

### Monitoring setup

```bash
# Enable Prometheus metrics
export SIGMALANG_METRICS_ENABLED=true

# Configure alerting
docker compose -f docker-compose.yml -f monitoring.yml up -d
```

## Next Steps

- Explore [Kubernetes Deployment](kubernetes.md)
- See [Local Setup Guide](../../LOCAL_SETUP_GUIDE.md)
- Read [Operations](../operations/monitoring.md)
