# Helm Chart Deployment

## Prerequisites

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version
```

## Quick Start

```bash
# Add repository
helm repo add sigmalang https://charts.sigmalang.io
helm repo update

# Search chart
helm search repo sigmalang

# Install
helm install sigmalang sigmalang/sigmalang \
  --namespace sigmalang \
  --create-namespace

# Verify
helm list -n sigmalang
kubectl get all -n sigmalang
```

## Chart Values

### Basic Configuration

```yaml
replicaCount: 3

image:
  repository: sigmalang/sigmalang
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  hosts:
    - host: api.sigmalang.io
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi
```

### Install with Custom Values

```bash
helm install sigmalang sigmalang/sigmalang \
  -f values.yaml \
  -n sigmalang \
  --create-namespace
```

## Configuration

### Persistence

```yaml
persistence:
  enabled: true
  storageClass: "standard"
  size: 10Gi
```

### Redis

```yaml
redis:
  enabled: true
  auth:
    enabled: true
    password: "your-secure-password"
```

### Monitoring

```yaml
prometheus:
  enabled: true
  port: 9090

grafana:
  enabled: true
  adminPassword: "admin"
```

## Commands

### Install

```bash
helm install sigmalang sigmalang/sigmalang -n sigmalang --create-namespace
```

### Upgrade

```bash
helm upgrade sigmalang sigmalang/sigmalang -n sigmalang
```

### Rollback

```bash
helm rollback sigmalang -n sigmalang
```

### Uninstall

```bash
helm uninstall sigmalang -n sigmalang
```

### Get Values

```bash
helm get values sigmalang -n sigmalang
```

### Get Manifest

```bash
helm get manifest sigmalang -n sigmalang
```

## Customization

### Custom Values File

Create `custom-values.yaml`:

```yaml
replicaCount: 5

image:
  tag: v1.0.0

ingress:
  enabled: true
  hosts:
    - host: sigmalang.example.com
      paths:
        - path: /

resources:
  limits:
    cpu: 2000m
    memory: 4Gi

persistence:
  enabled: true
  size: 50Gi
```

Install with custom values:

```bash
helm install sigmalang sigmalang/sigmalang \
  -f custom-values.yaml \
  -n sigmalang \
  --create-namespace
```

## Chart Structure

```
sigmalang/
├── Chart.yaml              # Chart metadata
├── values.yaml              # Default values
├── templates/
│   ├── deployment.yaml      # Deployment definition
│   ├── service.yaml         # Service definition
│   ├── ingress.yaml         # Ingress configuration
│   ├── statefulset.yaml     # Redis StatefulSet
│   └── configmap.yaml       # Configuration
└── README.md
```

## Development

### Create Custom Chart

```bash
helm create sigmalang-custom
cd sigmalang-custom

# Edit templates/ files
helm lint .
helm template sigmalang .
helm package .
```

### Test Chart

```bash
# Lint
helm lint sigmalang/

# Dry-run
helm install sigmalang sigmalang/ --dry-run --debug

# Install
helm install sigmalang sigmalang/

# Test
helm test sigmalang -n sigmalang
```

## Examples

### Production Deployment

```bash
# Production values
cat > prod-values.yaml <<EOF
replicaCount: 5

image:
  tag: v1.0.0
  pullPolicy: Always

service:
  type: LoadBalancer
  port: 443

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.example.com
      paths:
        - path: /

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

persistence:
  enabled: true
  storageClass: fast-ssd
  size: 100Gi

prometheus:
  enabled: true
  retention: 15d

grafana:
  enabled: true
  persistence:
    enabled: true
    size: 10Gi
EOF

# Install
helm install sigmalang sigmalang/sigmalang \
  -f prod-values.yaml \
  -n sigmalang \
  --create-namespace
```

### Development Deployment

```bash
cat > dev-values.yaml <<EOF
replicaCount: 1

image:
  tag: latest
  pullPolicy: Always

service:
  type: NodePort

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

persistence:
  enabled: false

prometheus:
  enabled: false

grafana:
  enabled: false
EOF

helm install sigmalang sigmalang/sigmalang \
  -f dev-values.yaml \
  -n sigmalang-dev \
  --create-namespace
```

## Troubleshooting

### Check Chart Syntax

```bash
helm lint sigmalang/
```

### Debug Installation

```bash
helm install sigmalang sigmalang/sigmalang \
  --dry-run \
  --debug \
  -n sigmalang
```

### Check Released Resources

```bash
helm get manifest sigmalang -n sigmalang
kubectl get all -n sigmalang
```

### View Release History

```bash
helm history sigmalang -n sigmalang
```

### Rollback on Failure

```bash
# Check history
helm history sigmalang -n sigmalang

# Rollback to previous
helm rollback sigmalang -n sigmalang

# Rollback to specific revision
helm rollback sigmalang 2 -n sigmalang
```

## Next Steps

- Read [Kubernetes Deployment](kubernetes.md)
- Configure [Monitoring](../operations/monitoring.md)
- Explore [Docker Deployment](docker.md)
