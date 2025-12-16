# ============================================================================

# PHASE 14 QUICK-PASTE PROMPTS FOR COPILOT

# Copy each section exactly as-is to Copilot for each project

# ============================================================================

## FOR RYOT LLM PROJECT

## Paste this exactly into Copilot:

---

**PHASE 14: Ryot LLM Production Deployment**

You are implementing Kubernetes manifests for Ryot LLM Phase 14 Production Deployment in the Neurectomy unified namespace.

**EXACT SPECIFICATIONS:**

- Namespace: `neurectomy` (shared - do NOT create project-specific namespace)
- Deployment name: `ryot-llm`
- Service name: `ryot-service` (NOT ryot-llm)
- HTTP Port: 8000, Metrics Port: 9090
- Replicas: 3, HPA: 1-5 (GPU limited)
- Resources: 4 CPU / 8 GB (request/limit)
- GPU: nvidia.com/gpu: "1" per pod
- Image: `neurectomy/ryot-llm:latest`
- Container user: non-root (1000)
- ConfigMap: Use `neurectomy-config` (shared - already exists)

**Create these files in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:**

1. **ryot-llm-deployment.yaml**:

   - Deployment (3 replicas, GPU node selector + tolerations)
   - Service (ClusterIP, ports 8000, 9090)
   - PVC (500Gi, efs-sc)
   - HPA (1-5 replicas, CPU 70%, Memory 80%)
   - ServiceAccount + PodDisruptionBudget + NetworkPolicy

2. **ryot-llm-secrets.yaml**: Secret with API credentials

3. **ryot-llm-rbac.yaml**: Role + RoleBinding for ServiceAccount

MUST INCLUDE:

- Init container waiting for ConfigMap availability
- envFrom configMapRef: neurectomy-config
- livenessProbe + readinessProbe + startupProbe (/health endpoint)
- securityContext: runAsNonRoot: true, runAsUser: 1000
- Volume mounts: /models (PVC), /cache (emptyDir)
- Pod anti-affinity for spreading across nodes

REFERENCE: Use Phase 14 ΣLANG implementation as template structure

---

## FOR ΣVAULT PROJECT

## Paste this exactly into Copilot:

---

**PHASE 14: ΣVAULT Production Deployment**

You are implementing Kubernetes manifests for ΣVAULT Phase 14 Production Deployment in the Neurectomy unified namespace.

**EXACT SPECIFICATIONS:**

- Namespace: `neurectomy` (shared - do NOT create project-specific namespace)
- StatefulSet name: `sigmavault` (NOT Deployment)
- Service name: `sigmavault` (headless, clusterIP: None)
- HTTP Port: 8002, Metrics Port: 9092
- Replicas: 3, HPA: 1-5
- Resources: 2 CPU / 4 GB (request/limit)
- Storage: 1Ti per pod (StatefulSet volumeClaimTemplates, gp3)
- Image: `neurectomy/sigmavault:latest`
- Container user: non-root (1000)
- ConfigMap: Use `neurectomy-config` (shared - already exists)
- Special: privileged: true, capabilities: SYS_ADMIN (FUSE filesystem)

**Create these files in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:**

1. **sigmavault-statefulset.yaml**:

   - StatefulSet (3 replicas, headless service)
   - Headless Service (clusterIP: None, ports 8002, 9092)
   - volumeClaimTemplates (1Ti RWO gp3)
   - HPA (1-5 replicas, CPU 70%, Memory 80%)
   - ServiceAccount + PodDisruptionBudget + NetworkPolicy

2. **sigmavault-secrets.yaml**: Secret with encryption.key, replication-peers

3. **sigmavault-rbac.yaml**: Role + RoleBinding for ServiceAccount

4. **sigmavault-storage.yaml**: StorageClass (gp3)

MUST INCLUDE:

- Init container waiting for ConfigMap + Secrets
- envFrom configMapRef: neurectomy-config
- env: POD_NAME, REPLICATION_FACTOR: "3"
- livenessProbe + readinessProbe + startupProbe (/health)
- securityContext: runAsNonRoot: true, runAsUser: 1000, privileged: true, SYS_ADMIN capability
- Volume mounts: /data (StatefulSet PVC), /secrets, /mnt/sigmavault
- Pod anti-affinity across nodes + zones
- Termination grace period: 60s

REFERENCE: Use Phase 14 ΣLANG implementation as template structure

---

## FOR NEURECTOMY PROJECT

## Paste this exactly into Copilot:

---

**PHASE 14: Neurectomy API Gateway Production Deployment**

You are implementing Kubernetes manifests for Neurectomy Phase 14 Production Deployment in the Neurectomy unified namespace (API Gateway / Core Service).

**EXACT SPECIFICATIONS:**

- Namespace: `neurectomy` (core namespace for all ecosystem services)
- Deployment name: `neurectomy-api`
- Service name: `neurectomy-api` (type: LoadBalancer)
- HTTP Port: 8080 (external), Metrics Port: 9093
- Replicas: 5, HPA: 5-50
- Resources: 1 CPU / 2 GB (request/limit)
- Image: `neurectomy/api:latest`
- Container user: non-root (1000)
- ConfigMap: Use `neurectomy-config` (shared - already exists)
- Secrets: `neurectomy-secrets` (db-password, redis-password, jwt-secret)

**Create these files in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:**

1. **neurectomy-api-deployment.yaml**:

   - Deployment (5 replicas, RollingUpdate)
   - LoadBalancer Service (port 80→8080, 9093)
   - HPA (5-50 replicas, CPU 70%, Memory 80%)
   - ServiceAccount + PodDisruptionBudget + NetworkPolicy

2. **neurectomy-secrets.yaml**: Secret with db-password, redis-password, jwt-secret

3. **neurectomy-rbac.yaml**: ClusterRole + RoleBinding for ServiceAccount

4. **neurectomy-ingress.yaml**: Ingress with hosts api.neurectomy.ai, app.neurectomy.ai

MUST INCLUDE:

- Init container waiting for: postgres (5432), redis (6379)
- envFrom configMapRef: neurectomy-config
- envFrom secretRef: neurectomy-secrets
- env: DATABASE_USER, POD_NAME, NODE_NAME
- livenessProbe + readinessProbe + startupProbe (/health, /ready)
- securityContext: runAsNonRoot: true, runAsUser: 1000
- Volume mounts: /tmp (emptyDir), config volume
- Pod anti-affinity across nodes + zones
- RollingUpdate: maxUnavailable: 1, maxSurge: 2
- Termination grace period: 60s
- NetworkPolicy: allow ingress from ingress-nginx, egress to all neurectomy services

REFERENCE: Use Phase 14 ΣLANG implementation as template structure

---

## SHARED INFRASTRUCTURE (already created)

Do NOT recreate these - they are shared across all 4 services:

- **namespace.yaml**: neurectomy namespace with ResourceQuota + LimitRange
- **configmaps/app-config.yaml**: neurectomy-config (used by ALL services via envFrom)
- **kustomization.yaml**: Orchestrates all manifests

---

## AFTER EACH PROJECT, RUN:

```bash
# Test locally
python infrastructure/scripts/verify_sigmalang.py --local-only

# Dry-run deployment
kubectl apply -k infrastructure/kubernetes/ --dry-run=client -o yaml | head -50

# After cluster setup
kubectl apply -k infrastructure/kubernetes/
kubectl get all -n neurectomy
```

---

## KEY RULE FOR ALL 4 PROJECTS

**Use neurectomy-config ConfigMap - it's already created and shared.**
Do not create project-specific ConfigMaps. All services read from `neurectomy-config` via:

```yaml
envFrom:
  - configMapRef:
      name: neurectomy-config
```

This ensures service discovery, endpoints, and configuration consistency across the entire unified architecture.
