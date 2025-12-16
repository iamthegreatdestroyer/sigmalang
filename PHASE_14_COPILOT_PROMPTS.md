# ============================================================================

# PHASE 14 COPILOT PROMPTS - Clear Instructions for Each Project

# ============================================================================

# Copy and paste the appropriate prompt into Copilot for each project

# These prompts incorporate lessons learned from successful ΣLANG implementation

# ============================================================================

---

## PROMPT 1: RYOT LLM PROJECT

## Phase 14: Production Deployment - Inference Service

```
You are implementing Phase 14 Production Deployment for the Ryot LLM project.

CRITICAL SPECIFICATIONS (from Phase 14 directive):
- Namespace: neurectomy (NOT ryot-specific - shared infrastructure)
- Port HTTP: 8000 (inference endpoint)
- Port Metrics: 9090 (Prometheus metrics)
- Replicas: 3 (GPU-intensive workload)
- HPA Range: 1-5 replicas (limited by GPU availability)
- CPU Request/Limit: 4 / 8
- Memory Request/Limit: 8Gi / 16Gi
- GPU: nvidia.com/gpu: "1"
- Container User: non-root (1000)
- Image: neurectomy/ryot-llm:latest
- Service Name: ryot-llm (NOT ryot-service)
- ConfigMap: neurectomy-config (SHARED - already exists)

YOUR TASK:
Create FOUR files in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:

1. FILE: ryot-llm-deployment.yaml
   INCLUDE:
   - Deployment (name: ryot-llm, 3 replicas, GPU node affinity/tolerations)
   - Service (name: ryot-service, ports 8000 and 9090)
   - PersistentVolumeClaim (name: ryot-models-pvc, 500Gi, efs-sc)
   - HPA (1-5 replicas, CPU 70%, Memory 80%)
   - ServiceAccount (name: ryot-sa)
   - PodDisruptionBudget (minAvailable: 2)
   - NetworkPolicy (allow ingress/egress from neurectomy namespace)

   REQUIRED CONFIGURATION:
   - envFrom: configMapRef neurectomy-config (shared)
   - securityContext: runAsUser 1000, runAsNonRoot true
   - livenessProbe: /health on port 8000
   - readinessProbe: /ready on port 8000
   - startupProbe: /health with 30 retries
   - Node selector: nvidia.com/gpu: "true", node-type: gpu
   - Tolerations: nvidia.com/gpu=true:NoSchedule
   - Volume mounts: /models (PVC), /cache (emptyDir)
   - Init container to wait for ConfigMap

2. FILE: ryot-llm-secrets.yaml
   INCLUDE:
   - Secret for model API keys, credentials
   - HPA metrics authorization (if needed)

3. FILE: ryot-llm-rbac.yaml
   INCLUDE:
   - Role with permissions to read ConfigMaps, Secrets
   - RoleBinding to ServiceAccount

4. FILE: kustomization.yaml (at infrastructure/kubernetes/)
   UPDATE existing kustomization.yaml to include:
   - resources: deployments/ryot-llm-deployment.yaml

VALIDATION:
After creating, verify with:
  python infrastructure/scripts/verify_sigmalang.py --local-only
  kubectl apply -k infrastructure/kubernetes/ --dry-run=client
```

---

## PROMPT 2: ΣVAULT PROJECT

## Phase 14: Production Deployment - Storage Service

```
You are implementing Phase 14 Production Deployment for the ΣVAULT project.

CRITICAL SPECIFICATIONS (from Phase 14 directive):
- Namespace: neurectomy (NOT sigmavault-specific - shared infrastructure)
- Port HTTP: 8002 (storage API endpoint)
- Port Metrics: 9092 (Prometheus metrics)
- Replicas: 3 (StatefulSet, NOT Deployment)
- HPA Range: 1-5 replicas
- CPU Request/Limit: 2 / 4
- Memory Request/Limit: 4Gi / 8Gi
- Storage: 1Ti per pod (StatefulSet volumeClaimTemplates)
- Storage Class: gp3 (EBS)
- Container User: non-root (1000)
- Image: neurectomy/sigmavault:latest
- Service Name: sigmavault (headless ClusterIP: None)
- ConfigMap: neurectomy-config (SHARED - already exists)
- Special: privileged: true, cap: SYS_ADMIN (for FUSE filesystem)

YOUR TASK:
Create FIVE files in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:

1. FILE: sigmavault-statefulset.yaml
   INCLUDE:
   - StatefulSet (name: sigmavault, 3 replicas, serviceName: sigmavault)
   - Headless Service (clusterIP: None, ports 8002 and 9092)
   - volumeClaimTemplates (1Ti gp3 storage, RWO access mode)
   - HPA (1-5 replicas, CPU 70%, Memory 80%)
   - ServiceAccount (name: sigmavault-sa)
   - PodDisruptionBudget (minAvailable: 2)
   - NetworkPolicy (allow ingress/egress from neurectomy namespace)

   REQUIRED CONFIGURATION:
   - envFrom: configMapRef neurectomy-config (shared)
   - env: POD_NAME, POD_NAMESPACE, REPLICATION_FACTOR=3
   - securityContext:
     * runAsUser 1000, runAsNonRoot true
     * privileged: true
     * capabilities: add SYS_ADMIN
   - volumeMounts: /data (StatefulSet PVC), /secrets (Secret), /mnt/sigmavault (FUSE)
   - livenessProbe: /health on port 8002
   - readinessProbe: /ready on port 8002
   - startupProbe: /health with 30 retries
   - Init container to wait for ConfigMap and secrets
   - Volumes: secrets (Secret volume), FUSE mount point
   - Pod anti-affinity (prefer different nodes)
   - Storage backend: distributed (env var)

2. FILE: sigmavault-secrets.yaml
   INCLUDE:
   - Secret (name: sigmavault-secrets) with:
     * encryption.key (base64 encoded)
     * replication-peers (cluster topology)

3. FILE: sigmavault-rbac.yaml
   INCLUDE:
   - Role with permissions:
     * read ConfigMaps, Secrets
     * list/watch StatefulSet
   - RoleBinding to sigmavault-sa

4. FILE: sigmavault-storage.yaml
   INCLUDE:
   - StorageClass (gp3 if not exists)
   - Optional: PersistentVolume for shared snapshots

5. FILE: kustomization.yaml (at infrastructure/kubernetes/)
   UPDATE existing kustomization.yaml to include:
   - resources: deployments/sigmavault-statefulset.yaml

VALIDATION:
After creating, verify with:
  kubectl apply -k infrastructure/kubernetes/ --dry-run=client
  kubectl get statefulset -n neurectomy sigmavault
  kubectl get pvc -n neurectomy
```

---

## PROMPT 3: NEURECTOMY PROJECT

## Phase 14: Production Deployment - API Gateway & Core

```
You are implementing Phase 14 Production Deployment for the Neurectomy project.

CRITICAL SPECIFICATIONS (from Phase 14 directive):
- Namespace: neurectomy
- Port HTTP: 8080 (API endpoint - primary)
- Port Metrics: 9093 (Prometheus metrics)
- Replicas: 5 (API gateway workload)
- HPA Range: 5-50 replicas (can scale significantly)
- CPU Request/Limit: 1 / 2
- Memory Request/Limit: 2Gi / 4Gi
- Service Type: LoadBalancer (external traffic)
- Container User: non-root (1000)
- Image: neurectomy/api:latest
- Service Name: neurectomy-api
- ConfigMap: neurectomy-config (SHARED - already exists)
- Secrets: neurectomy-secrets (db-password, redis-password, jwt-secret)
- Dependencies: Wait for postgres, redis, ryot-llm, sigmalang, sigmavault

YOUR TASK:
Create FIVE files in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:

1. FILE: neurectomy-api-deployment.yaml
   INCLUDE:
   - Deployment (name: neurectomy-api, 5 replicas)
   - Service (type: LoadBalancer, ports 80→8080 and 9093)
   - HPA (5-50 replicas, CPU 70%, Memory 80%)
   - ServiceAccount (name: neurectomy-api-sa)
   - PodDisruptionBudget (minAvailable: 3)
   - NetworkPolicy (allow ingress from ingress-nginx, egress to all services)

   REQUIRED CONFIGURATION:
   - envFrom:
     * configMapRef: neurectomy-config (shared endpoints)
     * secretRef: neurectomy-secrets (sensitive data)
   - env: DATABASE_USER, REDIS_PASSWORD (from secrets), POD_NAME, etc.
   - securityContext: runAsUser 1000, runAsNonRoot true
   - livenessProbe: /health on port 8080 (30s interval)
   - readinessProbe: /ready on port 8080 (5s interval)
   - startupProbe: /health with 30 retries
   - Init container to wait for:
     * postgres (DATABASE_HOST:5432)
     * redis (REDIS_HOST:6379)
   - Volumes: config (ConfigMap), tmp (emptyDir for logs/uploads)
   - Pod anti-affinity (prefer different nodes/zones)
   - RollingUpdate strategy: maxUnavailable=1, maxSurge=2
   - Service affinity: None (load balance across all pods)
   - Termination grace period: 60s

2. FILE: neurectomy-secrets.yaml
   INCLUDE:
   - Secret (name: neurectomy-secrets) with:
     * db-password
     * redis-password
     * jwt-secret
     * api-key (optional)
   NOTE: In production, use external secret management (Vault, AWS Secrets Manager)

3. FILE: neurectomy-rbac.yaml
   INCLUDE:
   - ClusterRole with permissions:
     * read ConfigMaps, Secrets (all namespaces if multi-tenant)
     * list/watch Pods (monitoring)
   - RoleBinding to neurectomy-api-sa

4. FILE: neurectomy-ingress.yaml
   INCLUDE:
   - Ingress (name: neurectomy-ingress)
     * Hosts: api.neurectomy.ai, app.neurectomy.ai
     * Backend: neurectomy-api service port 80
     * TLS: letsencrypt-prod issuer
     * Annotations: rate-limit, ssl-redirect, proxy-timeouts

5. FILE: kustomization.yaml (at infrastructure/kubernetes/)
   UPDATE existing kustomization.yaml to include:
   - resources: deployments/neurectomy-api-deployment.yaml

VALIDATION:
After creating, verify with:
  kubectl apply -k infrastructure/kubernetes/ --dry-run=client
  kubectl get deployment -n neurectomy neurectomy-api
  kubectl get svc -n neurectomy neurectomy-api
  kubectl get ingress -n neurectomy
```

---

## KEY LESSONS FROM ΣLANG IMPLEMENTATION

✅ DO:

1. Use SHARED infrastructure (neurectomy-config ConfigMap)
2. Use neurectomy namespace for ALL services
3. Include livenessProbe + readinessProbe + startupProbe (3-layer health)
4. Use init containers to wait for dependencies (ConfigMap, secrets, other services)
5. Include anti-affinity rules for high availability
6. Include NetworkPolicy for security isolation
7. Include PodDisruptionBudget for reliability
8. Include RBACs with minimal required permissions
9. Non-root containers (runAsUser: 1000)
10. Reference Phase 14 ConfigMap EXACTLY: neurectomy-config

❌ DON'T:

1. Create project-specific namespaces (use neurectomy)
2. Duplicate ConfigMaps - reference shared neurectomy-config
3. Skip init containers - they ensure dependency ordering
4. Forget resource requests/limits
5. Use service port != phase 14 spec (Ryot:8000, ΣLANG:8001, ΣVAULT:8002, Neurectomy:8080)
6. Skip metrics ports (9090, 9091, 9092, 9093 for each service)
7. Forget service account creation
8. Skip network policies
9. Forget termination grace periods
10. Use root container user

---

## FILE PATH REFERENCE (for all projects)

```
c:\Users\sgbil\sigmalang\infrastructure\kubernetes\
├── namespace.yaml                          (already created)
├── configmaps/
│   └── app-config.yaml                    (already created - SHARED)
├── deployments/
│   ├── sigmalang-deployment.yaml           (ΣLANG - DONE)
│   ├── ryot-llm-deployment.yaml            (RYOT - TODO)
│   ├── sigmavault-statefulset.yaml         (ΣVAULT - TODO)
│   └── neurectomy-api-deployment.yaml      (NEURECTOMY - TODO)
├── kustomization.yaml                     (update to include all)
└── (optional) ingress.yaml                 (for Neurectomy routing)
```

## TESTING AFTER EACH PROJECT

```bash
# Verify files locally
python infrastructure/scripts/verify_sigmalang.py --local-only

# Dry run Kubernetes
kubectl apply -k infrastructure/kubernetes/ --dry-run=client -o yaml | head -100

# After cluster access
kubectl apply -k infrastructure/kubernetes/
kubectl get all -n neurectomy
kubectl describe deployment/hpa/statefulset -n neurectomy [name]
```

---

**READY**: Copy the appropriate prompt (1, 2, or 3) into Copilot for each project.
This approach ensures consistency, avoids namespace confusion, and enforces Phase 14 specifications.
