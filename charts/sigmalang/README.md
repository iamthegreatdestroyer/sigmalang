# ΣLANG Helm Chart

Official Helm chart for deploying ΣLANG on Kubernetes.

## Prerequisites

- Kubernetes 1.20+
- Helm 3.8+
- PersistentVolume provisioner support (if using Redis persistence)

## Installing the Chart

### Add Bitnami repository (for Redis dependency)

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### Install from source

```bash
cd charts
helm install sigmalang ./sigmalang
```

### Install with custom values

```bash
helm install sigmalang ./sigmalang -f custom-values.yaml
```

### Install in specific namespace

```bash
helm install sigmalang ./sigmalang --namespace sigmalang --create-namespace
```

## Uninstalling the Chart

```bash
helm uninstall sigmalang
```

## Configuration

The following table lists the configurable parameters and their default values.

### Global parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Global Docker image registry | `""` |
| `global.imagePullSecrets` | Global Docker registry secret names | `[]` |

### Image parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.registry` | ΣLANG image registry | `docker.io` |
| `image.repository` | ΣLANG image repository | `sigmalang/sigmalang` |
| `image.tag` | ΣLANG image tag | `1.0.0` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |

### Deployment parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `3` |
| `revisionHistoryLimit` | Number of old ReplicaSets to retain | `5` |

### Service parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | HTTP API service port | `8000` |
| `service.metricsPort` | Prometheus metrics port | `9091` |

### Ingress parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hostname` | Default hostname | `sigmalang.example.com` |
| `ingress.tls` | Enable TLS | `true` |
| `ingress.tlsSecret` | TLS secret name | `sigmalang-tls` |

### Resource limits

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.limits.cpu` | CPU limit | `2000m` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `1Gi` |

### Autoscaling parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Minimum replicas | `3` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU % | `70` |
| `autoscaling.targetMemoryUtilizationPercentage` | Target Memory % | `80` |

### Configuration parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `config.logLevel` | Log level | `INFO` |
| `config.logFormat` | Log format | `json` |
| `config.workers` | Number of workers | `4` |
| `config.compressionWorkers` | Compression workers | `8` |
| `config.metricsEnabled` | Enable metrics | `true` |
| `config.cacheEnabled` | Enable caching | `true` |
| `config.cacheTTL` | Cache TTL (seconds) | `3600` |

### Redis parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `redis.enabled` | Deploy Redis subchart | `true` |
| `redis.architecture` | Redis architecture | `standalone` |
| `redis.auth.enabled` | Enable authentication | `true` |
| `redis.master.persistence.enabled` | Enable persistence | `true` |
| `redis.master.persistence.size` | PVC size | `8Gi` |

## Examples

### Minimal installation

```bash
helm install sigmalang ./sigmalang \
  --set replicaCount=1 \
  --set autoscaling.enabled=false \
  --set redis.enabled=false
```

### Production installation

```bash
helm install sigmalang ./sigmalang \
  --set replicaCount=5 \
  --set autoscaling.maxReplicas=20 \
  --set resources.limits.cpu=4000m \
  --set resources.limits.memory=8Gi \
  --set ingress.hostname=api.sigmalang.io \
  --set redis.master.persistence.size=50Gi
```

### With external Redis

```bash
helm install sigmalang ./sigmalang \
  --set redis.enabled=false \
  --set externalRedis.host=redis.example.com \
  --set externalRedis.port=6379
```

## Upgrading

### Upgrade to new version

```bash
helm upgrade sigmalang ./sigmalang --set image.tag=1.1.0
```

### Upgrade with new values

```bash
helm upgrade sigmalang ./sigmalang -f updated-values.yaml
```

## Monitoring

### Prometheus ServiceMonitor

Enable Prometheus Operator ServiceMonitor:

```yaml
monitoring:
  serviceMonitor:
    enabled: true
    interval: 30s
```

### Grafana Dashboards

Import the Grafana dashboard from `grafana/dashboards/sigmalang-overview.json`

## Troubleshooting

### Check pod status

```bash
kubectl get pods -l app.kubernetes.io/name=sigmalang
```

### View logs

```bash
kubectl logs -l app.kubernetes.io/name=sigmalang -f
```

### Describe pod

```bash
kubectl describe pod <pod-name>
```

### Check configuration

```bash
kubectl get configmap <release-name>-sigmalang-config -o yaml
```

## License

MIT
