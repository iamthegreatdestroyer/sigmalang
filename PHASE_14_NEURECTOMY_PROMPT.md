# ============================================================================

# PHASE 14 PRODUCTION DEPLOYMENT - NEURECTOMY PROJECT

# ============================================================================

# This is a standalone prompt file for the Neurectomy project

# Copy this entire prompt directly into Copilot when working on Neurectomy

# DO NOT reference other files - everything you need is here

# ============================================================================

**PHASE 14: Neurectomy API Gateway Production Deployment**

You are implementing Kubernetes manifests for Neurectomy Phase 14 Production Deployment in the Neurectomy unified namespace. This is the core API gateway and orchestration service.

**EXACT SPECIFICATIONS FOR NEURECTOMY:**

- Namespace: `neurectomy` (core namespace for entire ecosystem)
- Deployment name: `neurectomy-api`
- Service name: `neurectomy-api` (type: LoadBalancer - external traffic)
- HTTP Port: 8080 (internal), exposed as 80 via LoadBalancer
- Metrics Port: 9093 (Prometheus scrape target)
- Initial Replicas: 5 (high availability API tier)
- HPA Range: 5-50 replicas (can scale significantly for load)
- CPU Request: 1, CPU Limit: 2
- Memory Request: 2Gi, Memory Limit: 4Gi
- Image: `neurectomy/api:latest`
- Container user: non-root (1000:1000)
- ConfigMap: Use `neurectomy-config` (shared - already exists)
- Secrets: Use `neurectomy-secrets` (will be created)
- ServiceAccount: `neurectomy-api-sa`
- Dependencies: PostgreSQL (5432), Redis (6379), Ryot LLM (8000), ΣLANG (8001), ΣVAULT (8002)

**CREATE THESE 4 FILES in c:\Users\sgbil\sigmalang\infrastructure\kubernetes\deployments\:**

---

### FILE 1: neurectomy-api-deployment.yaml

Include ALL of the following in one YAML file:

1. **Deployment** (name: neurectomy-api):

   - spec.replicas: 5
   - Pod anti-affinity (preferred, weight 100) to spread across nodes
   - Topology spread constraint for zone distribution
   - RollingUpdate strategy: maxSurge=2, maxUnavailable=1 (aggressive for API tier)
   - Revision history limit: 10
   - Service account: neurectomy-api-sa
   - Security context: runAsNonRoot=true, runAsUser=1000, runAsGroup=1000, seccompProfile.type=RuntimeDefault

2. **Init container** (name: wait-for-dependencies):

   - image: busybox:1.36
   - Wait for PostgreSQL (DATABASE_HOST:5432)
   - Wait for Redis (REDIS_HOST:6379)
   - Ensures database connectivity before API starts
   - Security context: readOnlyRootFilesystem=true

3. **Container spec** (name: neurectomy-api):

   - image: neurectomy/api:latest
   - imagePullPolicy: Always
   - Ports: 8080 (http), 9093 (metrics)
   - Resource requests: cpu: "1", memory: "2Gi", ephemeral-storage: "500Mi"
   - Resource limits: cpu: "2", memory: "4Gi", ephemeral-storage: "2Gi"
   - envFrom configMapRef: neurectomy-config (SHARED - already exists)
   - envFrom secretRef: neurectomy-secrets (will be created)
   - env variables:
     - DATABASE_USER: "neurectomy"
     - DATABASE_PASSWORD: from secret neurectomy-secrets[db-password]
     - REDIS_PASSWORD: from secret neurectomy-secrets[redis-password]
     - JWT_SECRET: from secret neurectomy-secrets[jwt-secret]
     - POD_NAME: from metadata.name
     - POD_NAMESPACE: from metadata.namespace
     - POD_IP: from status.podIP
     - NODE_NAME: from spec.nodeName
   - Volume mounts: /tmp (emptyDir), /config (ConfigMap)
   - securityContext: allowPrivilegeEscalation=false, readOnlyRootFilesystem=true, drop ALL capabilities

4. **Health probes**:

   - livenessProbe: httpGet /health on port 8080, initialDelay=30s, period=30s, timeout=10s
   - readinessProbe: httpGet /ready on port 8080, initialDelay=15s, period=10s, timeout=5s
   - startupProbe: httpGet /health on port 8080, failureThreshold=30 (allows 155s startup)

5. **Lifecycle hooks**:

   - preStop: 15-second grace period before SIGTERM
     ```
     exec: /bin/sh -c "echo 'Graceful shutdown initiated...' && sleep 15"
     ```

6. **Volumes**:

   - config-volume: configMap (neurectomy-config)
   - tmp-volume: emptyDir (sizeLimit: 1Gi)

7. **LoadBalancer Service** (name: neurectomy-api):

   - type: LoadBalancer (exposes external IP)
   - selector: app=neurectomy-api, component=gateway
   - ports:
     - port: 80 (external)
     - targetPort: 8080 (container)
     - protocol: TCP
     - name: http
   - ports:
     - port: 9093 (metrics)
     - targetPort: 9093
     - protocol: TCP
     - name: metrics
   - sessionAffinity: None (round-robin across pods)

8. **HorizontalPodAutoscaler** (name: neurectomy-api-hpa):

   - scaleTargetRef: Deployment neurectomy-api
   - minReplicas: 5
   - maxReplicas: 50
   - Metrics: CPU 70%, Memory 80%
   - scaleDown: stabilizationWindow=300s, conservative (5% per minute or 1 pod per minute)
   - scaleUp: stabilizationWindow=60s, moderate (50% per 30s or 4 pods per 30s)

9. **ServiceAccount** (name: neurectomy-api-sa):

   - automountServiceAccountToken: false

10. **PodDisruptionBudget** (name: neurectomy-api-pdb):

    - minAvailable: 3 (keep 3 pods running during disruptions)

11. **NetworkPolicy** (name: neurectomy-api-network-policy):
    - Ingress:
      - from ingress-nginx namespace on port 8080
      - from monitoring/observability namespace on port 9093 (Prometheus)
      - from neurectomy namespace on port 8080 (internal services)
    - Egress:
      - DNS resolution (port 53 UDP/TCP to any)
      - PostgreSQL (port 5432 to neurectomy namespace)
      - Redis (port 6379 to neurectomy namespace)
      - Ryot LLM (port 8000 to neurectomy namespace)
      - ΣLANG (port 8001 to neurectomy namespace)
      - ΣVAULT (port 8002 to neurectomy namespace)

---

### FILE 2: neurectomy-secrets.yaml

Single Secret resource:

- name: neurectomy-secrets
- namespace: neurectomy
- type: Opaque
- Data to include (all base64 encoded):
  - db-password: (PostgreSQL password for neurectomy user)
  - redis-password: (Redis AUTH password)
  - jwt-secret: (JWT signing secret, min 32 characters)
  - api-key: (API key for external integrations, optional)
  - encryption-key: (Master encryption key for sensitive data, optional)

NOTE: In production, use external secret management (HashiCorp Vault, AWS Secrets Manager, etc.) instead of inline secrets.

---

### FILE 3: neurectomy-rbac.yaml

Include BOTH:

1. **ClusterRole** (name: neurectomy-api-role):

   - Rules:
     - apiGroups: [""] - resources: ["configmaps"] - verbs: ["get", "list", "watch"] (all namespaces)
     - apiGroups: [""] - resources: ["secrets"] - verbs: ["get", "list"] (all namespaces)
     - apiGroups: [""] - resources: ["pods"] - verbs: ["list", "watch", "get"] (all namespaces)
     - apiGroups: ["apps"] - resources: ["deployments", "statefulsets"] - verbs: ["list", "watch"]

2. **RoleBinding** (name: neurectomy-api-rolebinding):
   - namespace: neurectomy
   - roleRef: ClusterRole neurectomy-api-role
   - subjects: ServiceAccount neurectomy-api-sa (in neurectomy namespace)

---

### FILE 4: neurectomy-ingress.yaml

Single Ingress resource:

- name: neurectomy-ingress
- namespace: neurectomy
- annotations:
  - kubernetes.io/ingress.class: nginx
  - cert-manager.io/cluster-issuer: letsencrypt-prod (or your issuer)
  - nginx.ingress.kubernetes.io/ssl-redirect: "true"
  - nginx.ingress.kubernetes.io/rate-limit: "1000"
  - nginx.ingress.kubernetes.io/proxy-body-size: "100m"
  - nginx.ingress.kubernetes.io/proxy-connect-timeout: "60"
  - nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
  - nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
- TLS:
  - hosts: [api.neurectomy.ai, app.neurectomy.ai]
  - secretName: neurectomy-tls
- Rules:
  - host: api.neurectomy.ai
    - path: / (pathType: Prefix)
    - backend: service neurectomy-api, port 80
  - host: app.neurectomy.ai
    - path: / (pathType: Prefix)
    - backend: service neurectomy-api, port 80

---

**CRITICAL MUST-HAVES:**

✅ Deployment (5 replicas, rolling update aggressive)
✅ LoadBalancer Service (type: LoadBalancer for external access)
✅ Init container waits for PostgreSQL (5432) and Redis (6379)
✅ envFrom configMapRef: neurectomy-config (use SHARED config)
✅ envFrom secretRef: neurectomy-secrets
✅ env: DATABASE_USER, DATABASE_PASSWORD, REDIS_PASSWORD, JWT_SECRET, POD_NAME, NODE_NAME
✅ 3-layer health probes (liveness + readiness + startup)
✅ Non-root container (1000:1000)
✅ preStop lifecycle hook for graceful shutdown
✅ Pod anti-affinity for high availability
✅ HPA with 5-50 replicas range (aggressive scaling)
✅ NetworkPolicy (ingress from ingress-nginx, egress to services)
✅ PodDisruptionBudget (minAvailable: 3)
✅ Service account with ClusterRole (minimal required permissions)
✅ Termination grace period: 60s (for graceful shutdown)
✅ Ingress with TLS termination

**VALIDATION AFTER CREATING:**

1. Check local files:

   ```bash
   ls -la infrastructure/kubernetes/deployments/neurectomy-api*
   ```

2. Check YAML syntax:

   ```bash
   kubectl apply -k infrastructure/kubernetes/ --dry-run=client -o yaml | head -150
   ```

3. When cluster ready:
   ```bash
   kubectl apply -k infrastructure/kubernetes/
   kubectl get deployment -n neurectomy neurectomy-api
   kubectl get svc -n neurectomy neurectomy-api
   kubectl get ingress -n neurectomy
   kubectl describe deployment neurectomy-api -n neurectomy
   kubectl logs -n neurectomy -l app=neurectomy-api -f
   ```

**KEY REFERENCE:**

This deployment follows the same patterns as the Phase 14 ΣLANG implementation (already in infrastructure/kubernetes/deployments/sigmalang-deployment.yaml). Use that as a template for structure and best practices.

The main differences for Neurectomy:

- LoadBalancer Service (instead of ClusterIP) for external traffic
- 5 replicas (vs ΣLANG's 5, same)
- HPA: 5-50 (vs ΣLANG's 5-20, much larger scaling)
- Port 8080 (vs ΣLANG's 8001)
- Metrics port 9093 (vs ΣLANG's 9091)
- Init container waits for PostgreSQL + Redis (not just ConfigMap)
- ClusterRole (vs Namespaced Role) for multi-namespace permissions
- Includes Ingress for domain routing
- Includes preStop lifecycle hook
- Smaller resources (1 CPU request vs ΣLANG's 2)

**INFRASTRUCTURE ALREADY EXISTS** - do NOT recreate:

- Namespace: neurectomy (created)
- ConfigMap: neurectomy-config (created, shared by all 4 services)
- kustomization.yaml (will be updated to include neurectomy-api-deployment.yaml)

---

**COPY THIS ENTIRE PROMPT AND PASTE INTO COPILOT TO GENERATE ALL FOUR FILES**
