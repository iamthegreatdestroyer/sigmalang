# ΣLANG Helm Chart

Production-ready Helm chart for deploying ΣLANG on Kubernetes.

## Prerequisites

- Kubernetes 1.20+
- Helm 3.0+
- kubectl configured

## Installation

### Add Helm Repository

```bash
helm repo add sigmalang https://charts.sigmalang.io
helm repo update
```

### Install Chart

```bash
# Basic installation
helm install sigmalang sigmalang/sigmalang \
  --namespace sigmalang \
  --create-namespace

# With custom values
helm install sigmalang sigmalang/sigmalang \
  -f values.yaml \
  --namespace sigmalang
```

### Production Installation

```bash
# Create namespace
kubectl create namespace sigmalang

# Create secrets
kubectl create secret generic sigmalang-secrets \
  --from-literal=jwt-secret=$(openssl rand -hex 32) \
  --from-literal=redis-url=redis://redis:6379/0 \
  -n sigmalang

# Install with ingress and autoscaling
helm install sigmalang sigmalang/sigmalang \
  --set api.ingress.enabled=true \
  --set api.ingress.hosts[0].host=api.sigmalang.io \
  --set api.autoscaling.enabled=true \
  --namespace sigmalang
```

## Configuration

### Key Values

| Key | Description | Default |
|-----|-------------|---------|
| `api.replicaCount` | Number of API replicas | 3 |
| `api.image.tag` | Docker image tag | latest |
| `api.resources.requests.cpu` | CPU request | 500m |
| `api.resources.limits.cpu` | CPU limit | 2000m |
| `redis.enabled` | Deploy Redis | true |
| `prometheus.enabled` | Deploy Prometheus | true |
| `grafana.enabled` | Deploy Grafana | true |

### Customize Values

```bash
# Create custom values file
cat > custom-values.yaml <<EOF
api:
  replicaCount: 5
  resources:
    limits:
      cpu: "4"
      memory: "4Gi"
  ingress:
    enabled: true
    hosts:
      - host: sigmalang.example.com
        paths:
          - path: /
            pathType: Prefix
redis:
  enabled: true
  master:
    persistence:
      size: 50Gi
prometheus:
  enabled: true
  retention: 30d
EOF

# Install with custom values
helm install sigmalang sigmalang/sigmalang \
  -f custom-values.yaml \
  --namespace sigmalang
```

## Usage

### Check Deployment Status

```bash
# Get release status
helm status sigmalang -n sigmalang

# Watch deployment
kubectl rollout status deployment/sigmalang -n sigmalang

# Get pods
kubectl get pods -n sigmalang
```

### Access Services

```bash
# Port forward to API
kubectl port-forward svc/sigmalang 8000:8000 -n sigmalang

# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n sigmalang

# Get API endpoint
kubectl get ingress -n sigmalang
```

### Upgrade Release

```bash
helm upgrade sigmalang sigmalang/sigmalang \
  -f values.yaml \
  --namespace sigmalang
```

### Rollback Release

```bash
# List releases
helm history sigmalang -n sigmalang

# Rollback to previous
helm rollback sigmalang -n sigmalang
```

### Uninstall Release

```bash
helm uninstall sigmalang -n sigmalang
```

## Advanced Configuration

### External Redis

```yaml
redis:
  enabled: false
api:
  env:
    SIGMALANG_REDIS_URL: "redis://external-redis:6379/0"
```

### Custom SSL Certificate

```yaml
api:
  ingress:
    tls:
      - hosts:
          - sigmalang.example.com
        secretName: sigmalang-cert
```

### Node Affinity

```yaml
api:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: node-type
                operator: In
                values:
                  - compute
```

### Resource Quotas

```bash
# Create namespace with quotas
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: sigmalang-quota
  namespace: sigmalang
spec:
  hard:
    requests.cpu: "10"
    requests.memory: "20Gi"
    limits.cpu: "20"
    limits.memory: "40Gi"
EOF
```

## Monitoring

### Prometheus Queries

```promql
# Request rate
rate(sigmalang_request_count_total[5m])

# Error rate
rate(sigmalang_request_errors_total[5m])

# P95 latency
histogram_quantile(0.95, sigmalang_request_duration_ms_bucket)
```

### Grafana Dashboards

Pre-configured dashboards for:
- System Overview
- Application Performance
- Compression Metrics
- Cache Performance
- Resource Usage

## Troubleshooting

### Pods not starting

```bash
kubectl describe pod -n sigmalang <pod-name>
kubectl logs -n sigmalang <pod-name>
```

### Ingress not working

```bash
kubectl get ingress -n sigmalang
kubectl describe ingress -n sigmalang sigmalang
```

### Resource issues

```bash
kubectl top nodes
kubectl top pods -n sigmalang
```

## Updates

### Chart Version Bump

```bash
# Update chart version
helm lint .
helm package .
helm repo index .

# Publish to repository
# aws s3 cp sigmalang-x.x.x.tgz s3://sigmalang-charts/
```

## Support

- GitHub: https://github.com/iamthegreatdestroyer/sigmalang
- Issues: https://github.com/iamthegreatdestroyer/sigmalang/issues
- Documentation: https://sigmalang.io

---

**Chart Version**: 1.0.0
**App Version**: 1.0.0
**Last Updated**: February 19, 2026
