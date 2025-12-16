# ============================================================================
# PHASE 14 PRODUCTION DEPLOYMENT - RYOT LLM PROJECT
# ============================================================================
# This is a standalone prompt file for the Ryot LLM project
# Copy this entire prompt directly into Copilot when working on Ryot LLM
# DO NOT reference other files - everything you need is here
# ============================================================================

**PHASE 14: Ryot LLM Production Deployment**

You are implementing Kubernetes manifests for Ryot LLM Phase 14 Production Deployment in the Neurectomy unified namespace.

**EXACT SPECIFICATIONS FOR RYOT LLM:**
- Namespace: `neurectomy` (shared - do NOT create project-specific namespace)
- Deployment name: `ryot-llm`
- Service name: `ryot-service` (NOT ryot-llm)
- HTTP Port: 8000 (inference endpoint)
- Metrics Port: 9090 (Prometheus scrape target)
- Initial Replicas: 3 (GPU-intensive workload)
- HPA Range: 1-5 replicas (GPU limited - can't exceed available GPUs)
- CPU Request: 4, CPU Limit: 8
- Memory Request: 8Gi, Memory Limit: 16Gi
- GPU: nvidia.com/gpu: "1" per pod (requires 1 GPU per replica)
- Image: `neurectomy/ryot-llm:latest`
- Container user: non-root (1000:1000)
- ConfigMap: Use `neurectomy-config` (shared - already exists)
- ServiceAccount: `ryot-sa`
- PVC Name: `ryot-models-pvc`

**CREATE THESE 3 FILES in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:**

---

### FILE 1: ryot-llm-deployment.yaml

Include ALL of the following in one YAML file:

1. **Deployment** (name: ryot-llm):
   - spec.replicas: 3
   - GPU node selector: `nvidia.com/gpu: "true"`, `node-type: gpu`
   - GPU tolerations: `nvidia.com/gpu=true:NoSchedule`
   - Pod anti-affinity (preferred, weight 100) to spread across nodes
   - Topology spread constraint for zone distribution
   - RollingUpdate strategy: maxSurge=1, maxUnavailable=0
   - Revision history limit: 10
   - Service account: ryot-sa
   - Security context: runAsNonRoot=true, runAsUser=1000, runAsGroup=1000

2. **Container spec**:
   - name: ryot-llm
   - image: neurectomy/ryot-llm:latest
   - imagePullPolicy: Always
   - Ports: 8000 (http), 9090 (metrics)
   - Resource requests: cpu: "4", memory: "8Gi"
   - Resource limits: cpu: "8", memory: "16Gi"
   - envFrom configMapRef: neurectomy-config (SHARED - already exists)
   - Volume mounts: /models (PVC), /cache (emptyDir)
   - securityContext: allowPrivilegeEscalation=false, readOnlyRootFilesystem=true

3. **Health probes**:
   - livenessProbe: httpGet /health on port 8000, initialDelay=60s, period=30s
   - readinessProbe: httpGet /ready on port 8000, initialDelay=30s, period=10s
   - startupProbe: httpGet /health on port 8000, failureThreshold=30 (allows up to 155s startup)

4. **Init container**:
   - Wait for ConfigMap at /config/COMPRESSION_WORKERS
   - Ensures dependencies ready before main container starts

5. **Volumes**:
   - model-storage: persistentVolumeClaim (ryot-models-pvc)
   - cache: emptyDir (local node cache)

6. **Service** (name: ryot-service):
   - type: ClusterIP
   - selector: app=ryot-llm
   - ports: 8000→8000 (http), 9090→9090 (metrics)
   - No session affinity

7. **PersistentVolumeClaim** (name: ryot-models-pvc):
   - accessModes: ReadWriteMany
   - storageClassName: efs-sc
   - storage: 500Gi

8. **HorizontalPodAutoscaler** (name: ryot-llm-hpa):
   - scaleTargetRef: Deployment ryot-llm
   - minReplicas: 1
   - maxReplicas: 5
   - Metrics: CPU 70%, Memory 80%
   - scaleDown: stabilizationWindow=300s, conservative
   - scaleUp: stabilizationWindow=0 (immediate), aggressive

9. **ServiceAccount** (name: ryot-sa):
   - automountServiceAccountToken: false

10. **PodDisruptionBudget** (name: ryot-llm-pdb):
    - minAvailable: 2 (keep 2 pods running during disruptions)

11. **NetworkPolicy** (name: ryot-llm-network-policy):
    - Ingress: from neurectomy namespace, from monitoring namespace (Prometheus scrape)
    - Egress: DNS (53 UDP/TCP), communication to other neurectomy services (8000-8002, 8080)

---

### FILE 2: ryot-llm-secrets.yaml

Single Secret resource:
- name: ryot-llm-secrets
- namespace: neurectomy
- type: Opaque
- Data to include:
  * model-api-key: (base64 encoded)
  * huggingface-token: (base64 encoded)
  * openai-api-key: (base64 encoded - optional)

---

### FILE 3: ryot-llm-rbac.yaml

Include BOTH:

1. **Role** (name: ryot-llm-role):
   - namespace: neurectomy
   - Rules:
     * apiGroups: [""] - resources: ["configmaps"] - verbs: ["get", "list", "watch"]
     * apiGroups: [""] - resources: ["secrets"] - verbs: ["get", "list"]
     * apiGroups: [""] - resources: ["pods"] - verbs: ["list", "watch"]

2. **RoleBinding** (name: ryot-llm-rolebinding):
   - namespace: neurectomy
   - roleRef: Role ryot-llm-role
   - subjects: ServiceAccount ryot-sa (in neurectomy namespace)

---

**CRITICAL MUST-HAVES:**

✅ Init container waits for ConfigMap availability
✅ envFrom configMapRef: neurectomy-config (use SHARED config)
✅ GPU node selector + tolerations
✅ 3-layer health probes (liveness + readiness + startup)
✅ Non-root container (1000:1000)
✅ Pod anti-affinity for high availability
✅ HPA with 1-5 replicas range
✅ NetworkPolicy for security isolation
✅ PodDisruptionBudget for reliability
✅ Service account with minimal RBAC
✅ Termination grace period: 60s (for graceful shutdown)

**VALIDATION AFTER CREATING:**

1. Check local files:
   ```bash
   ls -la infrastructure/kubernetes/deployments/ryot-llm*
   ```

2. Check YAML syntax:
   ```bash
   kubectl apply -k infrastructure/kubernetes/ --dry-run=client -o yaml | head -100
   ```

3. When cluster ready:
   ```bash
   kubectl apply -k infrastructure/kubernetes/
   kubectl get pods -n neurectomy -l app=ryot-llm
   kubectl describe deployment ryot-llm -n neurectomy
   kubectl logs -n neurectomy -l app=ryot-llm -f
   ```

**KEY REFERENCE:**

This deployment follows the same patterns as the Phase 14 ΣLANG implementation (already in infrastructure/kubernetes/deployments/sigmalang-deployment.yaml). Use that as a template for structure and best practices.

The main differences for Ryot LLM:
- GPU support (node selector, tolerations, nvidia.com/gpu resource)
- 3 replicas (GPU limited, vs ΣLANG's 5)
- HPA: 1-5 range (vs ΣLANG's 5-20)
- Port 8000 (vs ΣLANG's 8001)
- Metrics port 9090 (vs ΣLANG's 9091)
- Larger memory (8Gi request, vs ΣLANG's 4Gi)

**INFRASTRUCTURE ALREADY EXISTS** - do NOT recreate:
- Namespace: neurectomy (created)
- ConfigMap: neurectomy-config (created, shared by all 4 services)
- kustomization.yaml (will be updated to include ryot-llm-deployment.yaml)

---

**COPY THIS ENTIRE PROMPT AND PASTE INTO COPILOT TO GENERATE ALL THREE FILES**
