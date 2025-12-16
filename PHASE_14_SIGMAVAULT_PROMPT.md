# ============================================================================
# PHASE 14 PRODUCTION DEPLOYMENT - ΣVAULT PROJECT
# ============================================================================
# This is a standalone prompt file for the ΣVAULT project
# Copy this entire prompt directly into Copilot when working on ΣVAULT
# DO NOT reference other files - everything you need is here
# ============================================================================

**PHASE 14: ΣVAULT Production Deployment**

You are implementing Kubernetes manifests for ΣVAULT Phase 14 Production Deployment in the Neurectomy unified namespace.

**EXACT SPECIFICATIONS FOR ΣVAULT:**
- Namespace: `neurectomy` (shared - do NOT create project-specific namespace)
- StatefulSet name: `sigmavault` (use StatefulSet, NOT Deployment)
- Service name: `sigmavault` (headless service with clusterIP: None)
- HTTP Port: 8002 (storage API endpoint)
- Metrics Port: 9092 (Prometheus scrape target)
- Initial Replicas: 3 (StatefulSet for stateful distributed storage)
- HPA Range: 1-5 replicas
- CPU Request: 2, CPU Limit: 4
- Memory Request: 4Gi, Memory Limit: 8Gi
- Storage: 1Ti per pod (StatefulSet volumeClaimTemplates, gp3)
- Storage Class: gp3 (EBS GP3 volumes for each pod)
- Image: `neurectomy/sigmavault:latest`
- Container user: non-root (1000:1000)
- Special: privileged=true, SYS_ADMIN capability (for FUSE filesystem mounting)
- ConfigMap: Use `neurectomy-config` (shared - already exists)
- ServiceAccount: `sigmavault-sa`

**CREATE THESE 4 FILES in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:**

---

### FILE 1: sigmavault-statefulset.yaml

Include ALL of the following in one YAML file:

1. **StatefulSet** (name: sigmavault):
   - spec.replicas: 3
   - serviceName: sigmavault (headless service for DNS)
   - Pod anti-affinity (preferred, weight 100) to spread across nodes/zones
   - Topology spread constraint for zone distribution
   - Service account: sigmavault-sa
   - Termination grace period: 60s
   - DNS policy: ClusterFirst (default)

2. **Init container** (name: wait-for-config):
   - image: busybox:1.36
   - Wait for ConfigMap volume mount
   - Ensure /config/COMPRESSION_WORKERS exists before proceeding
   - Security context: readOnlyRootFilesystem=true, allowPrivilegeEscalation=false

3. **Container spec** (name: sigmavault):
   - image: neurectomy/sigmavault:latest
   - imagePullPolicy: Always
   - Ports: 8002 (http), 9092 (metrics)
   - Resource requests: cpu: "2", memory: "4Gi", ephemeral-storage: "1Gi"
   - Resource limits: cpu: "4", memory: "8Gi", ephemeral-storage: "5Gi"
   - envFrom configMapRef: neurectomy-config (SHARED - already exists)
   - envFrom secretRef: sigmavault-secrets (will be created)
   - env variables:
     * POD_NAME: from metadata.name
     * POD_NAMESPACE: from metadata.namespace
     * POD_IP: from status.podIP
     * REPLICATION_FACTOR: "3"
   - Volume mounts: /data (StatefulSet PVC), /secrets (Secret), /mnt/sigmavault
   - securityContext:
     * allowPrivilegeEscalation: false
     * runAsNonRoot: true
     * runAsUser: 1000
     * runAsGroup: 1000
     * privileged: true
     * capabilities: add [SYS_ADMIN]

4. **Health probes**:
   - livenessProbe: httpGet /health on port 8002, initialDelay=30s, period=30s
   - readinessProbe: httpGet /ready on port 8002, initialDelay=15s, period=10s
   - startupProbe: httpGet /health on port 8002, failureThreshold=30 (allows 155s startup)

5. **Volumes**:
   - config-volume: configMap (neurectomy-config)
   - secrets: secret (sigmavault-secrets)
   - tmp-volume: emptyDir (sizeLimit: 1Gi)
   - cache-volume: emptyDir (sizeLimit: 2Gi)

6. **Headless Service** (name: sigmavault):
   - type: ClusterIP
   - clusterIP: None (headless - required for StatefulSet)
   - selector: app=sigmavault
   - ports: 8002→8002 (http), 9092→9092 (metrics)
   - sessionAffinity: None

7. **volumeClaimTemplates**:
   - metadata.name: data
   - accessModes: ReadWriteOnce
   - storageClassName: gp3
   - storage: 1Ti
   - This creates a separate 1Ti PVC for each StatefulSet pod

8. **HorizontalPodAutoscaler** (name: sigmavault-hpa):
   - scaleTargetRef: StatefulSet sigmavault
   - minReplicas: 1
   - maxReplicas: 5
   - Metrics: CPU 70%, Memory 80%
   - scaleDown: stabilizationWindow=300s, conservative
   - scaleUp: stabilizationWindow=60s, moderate pace

9. **ServiceAccount** (name: sigmavault-sa):
   - automountServiceAccountToken: false

10. **PodDisruptionBudget** (name: sigmavault-pdb):
    - minAvailable: 2 (keep 2 pods running during disruptions)

11. **NetworkPolicy** (name: sigmavault-network-policy):
    - Ingress: from neurectomy namespace pods, from monitoring namespace (Prometheus)
    - Egress: DNS (53 UDP/TCP), communication to neurectomy services (8000-8003)

---

### FILE 2: sigmavault-secrets.yaml

Single Secret resource:
- name: sigmavault-secrets
- namespace: neurectomy
- type: Opaque
- Data to include:
  * encryption.key: (32-byte base64 encoded AES-256 key)
  * replication-peers: (base64 encoded comma-separated pod list: sigmavault-0,sigmavault-1,sigmavault-2)
  * master-key-passphrase: (base64 encoded passphrase for encryption key)

---

### FILE 3: sigmavault-rbac.yaml

Include BOTH:

1. **Role** (name: sigmavault-role):
   - namespace: neurectomy
   - Rules:
     * apiGroups: [""] - resources: ["configmaps"] - verbs: ["get", "list", "watch"]
     * apiGroups: [""] - resources: ["secrets"] - verbs: ["get", "list", "watch"]
     * apiGroups: [""] - resources: ["pods"] - verbs: ["list", "watch", "get"]
     * apiGroups: [""] - resources: ["statefulsets"] - verbs: ["list", "watch"]

2. **RoleBinding** (name: sigmavault-rolebinding):
   - namespace: neurectomy
   - roleRef: Role sigmavault-role
   - subjects: ServiceAccount sigmavault-sa (in neurectomy namespace)

---

### FILE 4: sigmavault-storage.yaml

Include BOTH:

1. **StorageClass** (name: gp3):
   - provisioner: ebs.csi.aws.com
   - parameters:
     * type: gp3
     * iops: "3000"
     * throughput: "125"
   - allowVolumeExpansion: true

2. **ResourceQuota** (optional, name: sigmavault-quota):
   - hard:
     * persistentvolumeclaims: "10"
     * requests.storage: "50Ti"

---

**CRITICAL MUST-HAVES:**

✅ StatefulSet (NOT Deployment) for stateful distributed storage
✅ Headless Service (clusterIP: None) for pod DNS discovery
✅ volumeClaimTemplates (1Ti per pod, gp3, RWO)
✅ Init container waits for ConfigMap + Secrets
✅ envFrom configMapRef: neurectomy-config (use SHARED config)
✅ env: POD_NAME, REPLICATION_FACTOR: "3"
✅ privileged: true, SYS_ADMIN capability (for FUSE)
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
   ls -la infrastructure/kubernetes/deployments/sigmavault*
   ```

2. Check YAML syntax:
   ```bash
   kubectl apply -k infrastructure/kubernetes/ --dry-run=client -o yaml | head -100
   ```

3. When cluster ready:
   ```bash
   kubectl apply -k infrastructure/kubernetes/
   kubectl get statefulset -n neurectomy sigmavault
   kubectl get pvc -n neurectomy
   kubectl describe statefulset sigmavault -n neurectomy
   kubectl logs -n neurectomy sigmavault-0 -f
   ```

**KEY REFERENCE:**

This deployment follows the same patterns as the Phase 14 ΣLANG implementation (already in infrastructure/kubernetes/deployments/sigmalang-deployment.yaml). Use that as a template for structure and best practices.

The main differences for ΣVAULT:
- StatefulSet (instead of Deployment) for stateful workload
- Headless service (clusterIP: None) for pod discovery
- volumeClaimTemplates (1Ti per pod, instead of single shared PVC)
- GPU not required (compute workload)
- Port 8002 (vs ΣLANG's 8001)
- Metrics port 9092 (vs ΣLANG's 9091)
- privileged mode + SYS_ADMIN for FUSE filesystem mounting

**INFRASTRUCTURE ALREADY EXISTS** - do NOT recreate:
- Namespace: neurectomy (created)
- ConfigMap: neurectomy-config (created, shared by all 4 services)
- kustomization.yaml (will be updated to include sigmavault-statefulset.yaml)

---

**COPY THIS ENTIRE PROMPT AND PASTE INTO COPILOT TO GENERATE ALL FOUR FILES**
