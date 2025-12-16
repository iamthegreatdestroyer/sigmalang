# Phase 14: Ryot LLM Production Deployment Guide

**Status:** ✅ COMPLETE  
**Generated:** December 16, 2025  
**Validation:** 13/13 Checks PASS

---

## Overview

Phase 14 implements production-ready Kubernetes manifests for Ryot LLM deployment in the shared `neurectomy` namespace. The deployment includes GPU support, auto-scaling, security policies, and high availability features.

---

## Files Created

### 1. `ryot-llm-deployment.yaml` (10.2 KB)

**Contains 8 Kubernetes resources:**

1. **Namespace** - `neurectomy` (shared unified namespace)
2. **ServiceAccount** - `ryot-sa` (minimal permissions)
3. **PersistentVolumeClaim** - `ryot-models-pvc` (500Gi EFS storage)
4. **Deployment** - `ryot-llm` (3 replicas, GPU-enabled)
5. **Service** - `ryot-service` (ClusterIP, ports 8000, 9090)
6. **HorizontalPodAutoscaler** - `ryot-llm-hpa` (1-5 replicas)
7. **PodDisruptionBudget** - `ryot-llm-pdb` (min 2 available)
8. **NetworkPolicy** - `ryot-llm-network-policy` (ingress/egress rules)

### 2. `ryot-llm-secrets.yaml` (1.3 KB)

**Contains 1 resource:**

- **Secret** - `ryot-llm-secrets` (API keys and credentials)
  - `model-api-key`
  - `huggingface-token`
  - `openai-api-key` (optional)

### 3. `ryot-llm-rbac.yaml` (1.3 KB)

**Contains 2 resources:**

1. **Role** - `ryot-llm-role` (minimal RBAC permissions)

   - ConfigMap: get, list, watch
   - Secrets: get, list
   - Pods: list, watch

2. **RoleBinding** - `ryot-llm-rolebinding` (binds SA to Role)

---

## Deployment Specifications

### Infrastructure

| Component            | Value                      | Notes                    |
| -------------------- | -------------------------- | ------------------------ |
| **Namespace**        | neurectomy                 | Shared unified namespace |
| **Deployment Name**  | ryot-llm                   | Production service       |
| **Service Name**     | ryot-service               | ClusterIP service        |
| **Image**            | neurectomy/ryot-llm:latest | Always pull latest       |
| **Initial Replicas** | 3                          | GPU-intensive workload   |

### Computing Resources

| Resource   | Request            | Limit              | Notes            |
| ---------- | ------------------ | ------------------ | ---------------- |
| **CPU**    | 4 cores            | 8 cores            | High performance |
| **Memory** | 8Gi                | 16Gi               | LLM model memory |
| **GPU**    | 1 × nvidia.com/gpu | 1 × nvidia.com/gpu | Per pod          |

### Networking

| Port     | Name    | Purpose            |
| -------- | ------- | ------------------ |
| **8000** | http    | Inference endpoint |
| **9090** | metrics | Prometheus scrape  |

### Storage

| Name                | Type     | Size  | Filesystem               |
| ------------------- | -------- | ----- | ------------------------ |
| **ryot-models-pvc** | PVC      | 500Gi | EFS (ReadWriteMany)      |
| **cache**           | emptyDir | 2Gi   | In-memory with spillover |

### Security

| Feature                  | Setting     | Purpose                  |
| ------------------------ | ----------- | ------------------------ |
| **User**                 | 1000:1000   | Non-root container       |
| **RootFilesystem**       | Read-only   | Immutable root           |
| **Privilege Escalation** | Disabled    | No privilege escalation  |
| **Capabilities**         | Dropped ALL | Minimal capabilities     |
| **ServiceAccount Token** | Not mounted | Explicit secret mounting |

### High Availability

| Feature                 | Config                       | Purpose                 |
| ----------------------- | ---------------------------- | ----------------------- |
| **Pod Anti-affinity**   | Preferred, weight 100        | Spread across nodes     |
| **Topology Spread**     | Zone distribution            | Geographic distribution |
| **PodDisruptionBudget** | Min 2 available              | Survive disruptions     |
| **RollingUpdate**       | maxSurge=1, maxUnavailable=0 | Zero-downtime updates   |
| **Revision History**    | Keep 10                      | Rollback capability     |
| **Termination Grace**   | 60 seconds                   | Graceful shutdown       |

### Auto-Scaling

| Metric            | Value       | Policy                   |
| ----------------- | ----------- | ------------------------ |
| **Min Replicas**  | 1           | Minimum pods             |
| **Max Replicas**  | 5           | GPU-limited maximum      |
| **CPU Target**    | 70%         | Scale at 70% utilization |
| **Memory Target** | 80%         | Scale at 80% utilization |
| **Scale Up**      | Immediate   | Aggressive scaling       |
| **Scale Down**    | 300s window | Conservative scaling     |

### Health Checks

| Probe         | Path    | Initial | Period | Timeout | Threshold              |
| ------------- | ------- | ------- | ------ | ------- | ---------------------- |
| **Liveness**  | /health | 60s     | 30s    | 5s      | 3 failures             |
| **Readiness** | /ready  | 30s     | 10s    | 5s      | 3 failures             |
| **Startup**   | /health | 0s      | 5s     | 5s      | 30 attempts (150s max) |

### Network Policy

**Ingress:**

- From `neurectomy` namespace: ports 8000, 9090
- From `monitoring` namespace: port 9090 (Prometheus)

**Egress:**

- DNS: port 53 (UDP/TCP)
- Internal services: ports 8000-8002, 8080
- External APIs: ports 80, 443

---

## Pre-Deployment Checklist

- [ ] Cluster has at least 3 GPU nodes available
- [ ] `neurectomy` namespace exists or will be created
- [ ] EFS storage with `efs-sc` storage class configured
- [ ] `neurectomy-config` ConfigMap exists
- [ ] Container image `neurectomy/ryot-llm:latest` is available
- [ ] Replace placeholder secrets in `ryot-llm-secrets.yaml`
- [ ] Update `kustomization.yaml` to include new manifests

---

## Deployment Steps

### Step 1: Update Secrets

Edit `ryot-llm-secrets.yaml` and replace placeholder credentials:

```bash
# Generate base64 encoded values
echo -n "your-actual-api-key" | base64
# Use output for each secret field
```

### Step 2: Update Kustomization

Add to `infrastructure/kubernetes/kustomization.yaml`:

```yaml
resources:
  # ... existing resources ...
  - deployments/ryot-llm-deployment.yaml
  - deployments/ryot-llm-secrets.yaml
  - deployments/ryot-llm-rbac.yaml
```

### Step 3: Validate Manifests

```bash
# Check YAML syntax
kubectl apply -k infrastructure/kubernetes/ --dry-run=client -o yaml

# Or use Python validator
python infrastructure/kubernetes/deployments/validate-ryot-llm.py
```

### Step 4: Deploy

```bash
# Deploy to cluster
kubectl apply -k infrastructure/kubernetes/

# Or individual files
kubectl apply -f infrastructure/kubernetes/deployments/ryot-llm-deployment.yaml
kubectl apply -f infrastructure/kubernetes/deployments/ryot-llm-rbac.yaml
kubectl apply -f infrastructure/kubernetes/deployments/ryot-llm-secrets.yaml
```

### Step 5: Verify Deployment

```bash
# Watch pod startup
kubectl get pods -n neurectomy -l app=ryot-llm -w

# Check deployment status
kubectl describe deployment ryot-llm -n neurectomy

# View container logs
kubectl logs -n neurectomy -l app=ryot-llm -f

# Check service endpoints
kubectl get endpoints ryot-service -n neurectomy

# Verify HPA status
kubectl get hpa -n neurectomy ryot-llm-hpa
```

---

## Operational Commands

### Monitoring

```bash
# Real-time pod monitoring
kubectl get pods -n neurectomy -l app=ryot-llm -w

# Resource usage
kubectl top pods -n neurectomy -l app=ryot-llm

# HPA metrics
kubectl get hpa -n neurectomy -w

# Event monitoring
kubectl get events -n neurectomy --sort-by=.metadata.creationTimestamp
```

### Debugging

```bash
# Describe pod
kubectl describe pod -n neurectomy -l app=ryot-llm

# Check logs
kubectl logs -n neurectomy -l app=ryot-llm --tail=100

# Port forward for testing
kubectl port-forward -n neurectomy svc/ryot-service 8000:8000

# Execute shell in pod
kubectl exec -it -n neurectomy deployment/ryot-llm -- /bin/sh
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment ryot-llm -n neurectomy --replicas=5

# Check HPA status
kubectl describe hpa ryot-llm-hpa -n neurectomy

# Adjust HPA limits
kubectl patch hpa ryot-llm-hpa -n neurectomy -p '{"spec":{"maxReplicas":6}}'
```

### Updates

```bash
# Update image
kubectl set image deployment/ryot-llm \
  ryot-llm=neurectomy/ryot-llm:v1.1.0 \
  -n neurectomy

# Check rollout status
kubectl rollout status deployment/ryot-llm -n neurectomy

# Rollback if needed
kubectl rollout undo deployment/ryot-llm -n neurectomy

# View rollout history
kubectl rollout history deployment/ryot-llm -n neurectomy
```

---

## Validation Results

```
✓ YAML Syntax:              PASS
✓ Namespace Configuration:  PASS
✓ Deployment Spec:          PASS
✓ Service Configuration:    PASS
✓ GPU Support:              PASS
✓ Health Probes:            PASS
✓ Security Context:         PASS
✓ HPA Configuration:        PASS
✓ Pod Anti-affinity:        PASS
✓ NetworkPolicy:            PASS
✓ PodDisruptionBudget:      PASS
✓ ServiceAccount & RBAC:    PASS
✓ Storage Configuration:    PASS

Total: 13/13 Checks PASSED
```

---

## Next Steps

1. **Configure Secrets**

   - Replace placeholder credentials with actual values
   - Consider using sealed-secrets or external-secrets operator

2. **Update Kustomization**

   - Add Ryot LLM manifests to kustomization.yaml
   - Include all three YAML files

3. **Deploy**

   - Run `kubectl apply -k infrastructure/kubernetes/`
   - Monitor initial pod startup

4. **Verify**

   - Check pod status and logs
   - Verify service endpoints
   - Test inference endpoint (port 8000)

5. **Monitor**
   - Set up Prometheus scraping (metrics on port 9090)
   - Configure Grafana dashboards
   - Set up alerts for pod failures

---

## Key Differences from ΣLANG

| Feature              | Ryot LLM | ΣLANG |
| -------------------- | -------- | ----- |
| **GPU Support**      | ✅ Yes   | ❌ No |
| **Initial Replicas** | 3        | 5     |
| **HPA Range**        | 1-5      | 5-20  |
| **HTTP Port**        | 8000     | 8001  |
| **Metrics Port**     | 9090     | 9091  |
| **Memory Request**   | 8Gi      | 4Gi   |
| **Storage Type**     | EFS      | EBS   |

---

## Support & References

- **Kubernetes Docs:** https://kubernetes.io/docs/
- **GPU Support:** https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- **HPA:** https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
- **NetworkPolicy:** https://kubernetes.io/docs/concepts/services-networking/network-policies/

---

**Phase 14 Deployment: PRODUCTION-READY ✅**

All manifests have been validated and are ready for deployment to the cluster.
