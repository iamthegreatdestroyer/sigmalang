# ΣLANG Deployment Guide

Complete guide for deploying ΣLANG in development, Docker, and Kubernetes environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Provider Guides](#cloud-provider-guides)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.9+
- pip or poetry
- Redis (optional, for caching)

### Installation

```bash
# Clone repository
git clone https://github.com/iamthegreatdestroyer/sigmalang.git
cd sigmalang

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Running the Server

```bash
# Development mode with hot reload
sigmalang serve --reload --debug

# Production mode
sigmalang serve --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# .env
SIGMALANG_HOST=0.0.0.0
SIGMALANG_PORT=8000
SIGMALANG_DEBUG=true
SIGMALANG_LOG_LEVEL=DEBUG
SIGMALANG_CACHE_ENABLED=false
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=sigmalang --cov-report=html

# Specific test file
pytest tests/test_api_server.py -v
```

---

## Docker Deployment

### Quick Start

```bash
# Build image
docker build -t sigmalang:latest .

# Run container
docker run -d \
  --name sigmalang-api \
  -p 8000:8000 \
  -e SIGMALANG_DEBUG=false \
  sigmalang:latest
```

### Using Docker Compose

#### Production Stack

Start the full production stack (API + Redis + Prometheus + Grafana):

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f sigmalang

# Check status
docker compose ps

# Stop all services
docker compose down
```

**Services included:**
| Service | Port | Description |
|---------|------|-------------|
| sigmalang | 8000 | API Server |
| redis | 6379 | Cache |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards (admin/admin) |

#### Development Stack

Start development environment with hot reload:

```bash
# Start dev environment
docker compose -f docker-compose.dev.yml up

# Rebuild after changes
docker compose -f docker-compose.dev.yml up --build
```

**Features:**

- Hot reload on code changes
- Debug logging enabled
- Source code mounted as volume
- No rate limiting

### Building Custom Images

```bash
# Development image
docker build --target development -t sigmalang:dev .

# CLI-only image
docker build --target cli -t sigmalang:cli .

# Production with custom tag
docker build -t ghcr.io/youruser/sigmalang:v1.0.0 .
```

### Image Optimization

The multi-stage Dockerfile produces optimized images:

| Target      | Size   | Use Case       |
| ----------- | ------ | -------------- |
| production  | ~450MB | API server     |
| development | ~600MB | Dev with tools |
| cli         | ~400MB | CLI tools only |

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3 (optional)

### Quick Deploy

```bash
# Apply all manifests
kubectl apply -k k8s/

# Check deployment status
kubectl get all -n sigmalang

# View logs
kubectl logs -n sigmalang -l app.kubernetes.io/name=sigmalang -f
```

### Manifest Overview

```
k8s/
├── namespace.yaml       # sigmalang namespace
├── configmap.yaml       # Non-sensitive configuration
├── secret.yaml          # Sensitive configuration (API keys, etc.)
├── deployment.yaml      # API deployment (3 replicas)
├── service.yaml         # ClusterIP service
├── ingress.yaml         # NGINX ingress with TLS
├── hpa.yaml             # Horizontal Pod Autoscaler
├── redis.yaml           # Redis StatefulSet
├── networkpolicy.yaml   # Network isolation
└── kustomization.yaml   # Kustomize configuration
```

### Configuration

#### Update ConfigMap

```bash
kubectl edit configmap sigmalang-config -n sigmalang
```

Key settings:

```yaml
data:
  SIGMALANG_HOST: "0.0.0.0"
  SIGMALANG_PORT: "8000"
  SIGMALANG_WORKERS: "4"
  SIGMALANG_LOG_LEVEL: "INFO"
  SIGMALANG_RATE_LIMIT_ENABLED: "true"
```

#### Update Secrets

```bash
# Create from literal
kubectl create secret generic sigmalang-secret \
  --from-literal=SIGMALANG_API_KEY=your-key-here \
  --from-literal=SIGMALANG_JWT_SECRET=your-jwt-secret \
  -n sigmalang

# Or update existing
kubectl edit secret sigmalang-secret -n sigmalang
```

### Scaling

#### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment sigmalang-api --replicas=5 -n sigmalang
```

#### Auto-scaling (HPA)

The HPA is configured to:

- Min replicas: 2
- Max replicas: 10
- Scale up at 70% CPU utilization
- Scale up at 80% memory utilization

```bash
# Check HPA status
kubectl get hpa -n sigmalang

# View HPA details
kubectl describe hpa sigmalang-api-hpa -n sigmalang
```

### Ingress

#### Configure Domain

Edit `k8s/ingress.yaml`:

```yaml
spec:
  rules:
    - host: api.yourdomain.com # Change this
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: sigmalang-api
                port:
                  number: 80
```

#### TLS/HTTPS

The ingress is configured to use cert-manager for automatic TLS certificates:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
EOF
```

### Kustomize Overlays

Create environment-specific overlays:

```bash
# Create production overlay
mkdir -p k8s/overlays/production

cat > k8s/overlays/production/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../

namespace: sigmalang-prod

patches:
  - patch: |-
      - op: replace
        path: /spec/replicas
        value: 5
    target:
      kind: Deployment
      name: sigmalang-api

images:
  - name: ghcr.io/iamthegreatdestroyer/sigmalang
    newTag: v1.0.0
EOF

# Apply production overlay
kubectl apply -k k8s/overlays/production
```

---

## Cloud Provider Guides

### AWS (EKS)

```bash
# Create EKS cluster
eksctl create cluster --name sigmalang --region us-east-1 --nodes 3

# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=sigmalang

# Deploy ΣLANG
kubectl apply -k k8s/

# Use ALB Ingress (uncomment in ingress.yaml)
# kubernetes.io/ingress.class: alb
# alb.ingress.kubernetes.io/scheme: internet-facing
```

### GCP (GKE)

```bash
# Create GKE cluster
gcloud container clusters create sigmalang \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials sigmalang --zone us-central1-a

# Deploy ΣLANG
kubectl apply -k k8s/
```

### Azure (AKS)

```bash
# Create resource group
az group create --name sigmalang-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group sigmalang-rg \
  --name sigmalang-aks \
  --node-count 3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group sigmalang-rg --name sigmalang-aks

# Deploy ΣLANG
kubectl apply -k k8s/
```

---

## Configuration

### Environment Variables Reference

| Variable                       | Default                  | Description                             |
| ------------------------------ | ------------------------ | --------------------------------------- |
| `SIGMALANG_HOST`               | 0.0.0.0                  | Server bind address                     |
| `SIGMALANG_PORT`               | 8000                     | Server port                             |
| `SIGMALANG_WORKERS`            | 4                        | Number of worker processes              |
| `SIGMALANG_DEBUG`              | false                    | Enable debug mode                       |
| `SIGMALANG_LOG_LEVEL`          | INFO                     | Log level (DEBUG, INFO, WARNING, ERROR) |
| `SIGMALANG_LOG_FORMAT`         | json                     | Log format (json, console)              |
| `SIGMALANG_CACHE_ENABLED`      | true                     | Enable Redis caching                    |
| `SIGMALANG_REDIS_URL`          | redis://localhost:6379/0 | Redis connection URL                    |
| `SIGMALANG_RATE_LIMIT_ENABLED` | true                     | Enable rate limiting                    |
| `SIGMALANG_RATE_LIMIT_RPM`     | 60                       | Requests per minute                     |
| `SIGMALANG_METRICS_ENABLED`    | true                     | Enable Prometheus metrics               |
| `SIGMALANG_CORS_ORIGINS`       | \*                       | Allowed CORS origins                    |

---

## Monitoring

### Prometheus Metrics

Available at `/metrics`:

| Metric                               | Type      | Description         |
| ------------------------------------ | --------- | ------------------- |
| `sigmalang_requests_total`           | Counter   | Total API requests  |
| `sigmalang_request_duration_seconds` | Histogram | Request latency     |
| `sigmalang_encode_total`             | Counter   | Encoding operations |
| `sigmalang_encode_duration_seconds`  | Histogram | Encoding latency    |
| `sigmalang_analogy_total`            | Counter   | Analogy operations  |
| `sigmalang_search_total`             | Counter   | Search operations   |
| `sigmalang_errors_total`             | Counter   | Error count by type |

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

Pre-configured dashboards:

- **ΣLANG Overview**: Request rate, latency, error rate
- **API Performance**: Per-endpoint metrics
- **System Resources**: CPU, memory, connections

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready

# Detailed health
curl http://localhost:8000/health/detailed
```

---

## Troubleshooting

### Common Issues

#### Container fails to start

```bash
# Check logs
docker logs sigmalang-api

# Or in Kubernetes
kubectl logs -n sigmalang deployment/sigmalang-api
```

#### High memory usage

```bash
# Reduce workers
SIGMALANG_WORKERS=2

# Enable memory limits in Kubernetes
resources:
  limits:
    memory: "1Gi"
```

#### Redis connection issues

```bash
# Test Redis connectivity
redis-cli -h redis ping

# Check Redis URL format
SIGMALANG_REDIS_URL=redis://redis:6379/0
```

#### SSL/TLS certificate issues

```bash
# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Check certificate status
kubectl describe certificate sigmalang-tls -n sigmalang
```

### Debug Mode

Enable detailed logging:

```bash
# Docker
docker run -e SIGMALANG_DEBUG=true -e SIGMALANG_LOG_LEVEL=DEBUG sigmalang:latest

# Kubernetes
kubectl set env deployment/sigmalang-api SIGMALANG_DEBUG=true SIGMALANG_LOG_LEVEL=DEBUG -n sigmalang
```

### Performance Tuning

```yaml
# Increase resources for high load
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"

# Increase replicas
replicas: 5

# Tune HPA thresholds
targetCPUUtilizationPercentage: 60
```

---

## Makefile Commands

```bash
# Development
make install-dev     # Install dev dependencies
make test            # Run tests
make lint            # Run linters
make format          # Format code

# Docker
make docker-build    # Build Docker image
make docker-run      # Run container
make docker-push     # Push to registry

# Docker Compose
make compose-up      # Start production stack
make compose-dev     # Start dev environment
make compose-down    # Stop all services

# Kubernetes
make k8s-deploy      # Deploy to Kubernetes
make k8s-delete      # Delete from Kubernetes
make k8s-logs        # View logs
```

---

## Next Steps

- [API Reference](../api/README.md) - Complete REST API documentation
- [SDK Guide](../sdk/README.md) - Python SDK usage
- [Examples](../../examples/) - Runnable code examples
