# Kubernetes Deployment

## Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm (optional, but recommended)
- Docker registry access

## Quick Start with Helm

```bash
# Add Helm repository
helm repo add sigmalang https://charts.sigmalang.io
helm repo update

# Install chart
helm install sigmalang sigmalang/sigmalang \
  --namespace sigmalang \
  --create-namespace

# Check deployment
kubectl get pods -n sigmalang
kubectl get services -n sigmalang
```

## Manual Deployment

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sigmalang
```

Apply:
```bash
kubectl apply -f namespace.yaml
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sigmalang-config
  namespace: sigmalang
data:
  SIGMALANG_CACHE_BACKEND: "redis"
  SIGMALANG_LOG_LEVEL: "INFO"
  SIGMALANG_METRICS_ENABLED: "true"
```

### Secret

```bash
kubectl create secret generic sigmalang-secrets \
  --from-literal=redis-password=your-password \
  -n sigmalang
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sigmalang
  namespace: sigmalang
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sigmalang
  template:
    metadata:
      labels:
        app: sigmalang
    spec:
      containers:
      - name: sigmalang
        image: sigmalang/sigmalang:latest
        ports:
        - containerPort: 8000
        env:
        - name: SIGMALANG_CACHE_BACKEND
          valueFrom:
            configMapKeyRef:
              name: sigmalang-config
              key: SIGMALANG_CACHE_BACKEND
        - name: SIGMALANG_REDIS_URL
          value: redis://redis:6379/0
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

Apply:
```bash
kubectl apply -f deployment.yaml
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: sigmalang
  namespace: sigmalang
spec:
  selector:
    app: sigmalang
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Redis StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: sigmalang
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## Ingress Configuration

### NGINX Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sigmalang
  namespace: sigmalang
spec:
  ingressClassName: nginx
  rules:
  - host: api.sigmalang.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sigmalang
            port:
              number: 80
  tls:
  - hosts:
    - api.sigmalang.io
    secretName: sigmalang-tls
```

## Monitoring Setup

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sigmalang
  namespace: sigmalang
spec:
  selector:
    matchLabels:
      app: sigmalang
  endpoints:
  - port: metrics
    interval: 30s
```

### PrometheusRule

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: sigmalang
  namespace: sigmalang
spec:
  groups:
  - name: sigmalang
    rules:
    - alert: HighErrorRate
      expr: rate(sigmalang_errors_total[5m]) > 0.05
      for: 5m
```

## Scaling

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sigmalang-hpa
  namespace: sigmalang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sigmalang
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Commands

### Deploy

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f redis.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Check Status

```bash
# Pods
kubectl get pods -n sigmalang

# Services
kubectl get svc -n sigmalang

# Deployment
kubectl get deployment -n sigmalang

# Events
kubectl get events -n sigmalang
```

### View Logs

```bash
# Pod logs
kubectl logs -n sigmalang deployment/sigmalang

# Stream logs
kubectl logs -n sigmalang deployment/sigmalang -f

# Previous logs (if crashed)
kubectl logs -n sigmalang deployment/sigmalang --previous
```

### Port Forwarding

```bash
# Forward local port to service
kubectl port-forward -n sigmalang service/sigmalang 8000:80

# Access at http://localhost:8000
```

### Execute Commands

```bash
# Execute command in pod
kubectl exec -n sigmalang deployment/sigmalang -- sigmalang --version

# Interactive shell
kubectl exec -n sigmalang deployment/sigmalang -it -- /bin/sh
```

### Update Deployment

```bash
# Update image
kubectl set image -n sigmalang deployment/sigmalang \
  sigmalang=sigmalang/sigmalang:v1.0.1 \
  --record

# View rollout history
kubectl rollout history -n sigmalang deployment/sigmalang

# Rollback to previous version
kubectl rollout undo -n sigmalang deployment/sigmalang
```

## Troubleshooting

### Pod won't start

```bash
# Check pod status
kubectl describe pod -n sigmalang <pod-name>

# View logs
kubectl logs -n sigmalang <pod-name>

# Previous logs
kubectl logs -n sigmalang <pod-name> --previous
```

### Service not accessible

```bash
# Check service
kubectl describe service -n sigmalang sigmalang

# Check endpoints
kubectl get endpoints -n sigmalang sigmalang

# Test from pod
kubectl run -n sigmalang test-pod --image=curlimages/curl -it -- \
  curl http://sigmalang:80/health
```

### High memory usage

```bash
# Check resource usage
kubectl top pods -n sigmalang

# Increase limits
kubectl set resources deployment -n sigmalang sigmalang \
  --limits=memory=3Gi --requests=memory=2Gi
```

## Next Steps

- Deploy with [Helm Chart](helm.md)
- Configure [Monitoring](../operations/monitoring.md)
- Set up [Logging](../operations/logging.md)
