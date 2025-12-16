#!/usr/bin/env python3
"""
Phase 14: Ryot LLM Kubernetes Deployment Validator

Validates the YAML manifests for Kubernetes deployment.
"""

import yaml
import sys
from pathlib import Path


def validate_yaml_file(file_path):
    """Validate YAML syntax and structure."""
    print(f"\n{'='*70}")
    print(f"Validating: {file_path.name}")
    print('='*70)
    
    try:
        with open(file_path, 'r') as f:
            documents = yaml.safe_load_all(f)
            doc_list = list(documents)
        
        print(f"✓ YAML syntax valid")
        print(f"✓ Documents found: {len(doc_list)}")
        
        for i, doc in enumerate(doc_list, 1):
            if doc and isinstance(doc, dict):
                kind = doc.get('kind', 'Unknown')
                name = doc.get('metadata', {}).get('name', 'unnamed')
                namespace = doc.get('metadata', {}).get('namespace', 'default')
                print(f"  [{i}] {kind:20} - {name:30} (ns: {namespace})")
        
        return True
        
    except yaml.YAMLError as e:
        print(f"✗ YAML syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_deployment_specs(deployment_path):
    """Check specific deployment specifications."""
    print(f"\n{'='*70}")
    print("Checking Deployment Specifications")
    print('='*70)
    
    try:
        with open(deployment_path, 'r') as f:
            documents = list(yaml.safe_load_all(f))
        
        checks = {
            "Namespace: neurectomy": False,
            "Deployment: ryot-llm": False,
            "Service: ryot-service": False,
            "Initial replicas: 3": False,
            "GPU node selector": False,
            "Health probes": False,
            "Non-root user (1000:1000)": False,
            "HPA: 1-5 replicas": False,
            "Pod anti-affinity": False,
            "NetworkPolicy": False,
            "PodDisruptionBudget": False,
            "ServiceAccount: ryot-sa": False,
            "PVC: ryot-models-pvc": False,
        }
        
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            
            kind = doc.get('kind')
            metadata = doc.get('metadata', {})
            name = metadata.get('name')
            namespace = metadata.get('namespace', 'default')
            
            # Check namespace
            if namespace == 'neurectomy':
                checks["Namespace: neurectomy"] = True
            
            # Check Deployment
            if kind == 'Deployment' and name == 'ryot-llm':
                checks["Deployment: ryot-llm"] = True
                spec = doc.get('spec', {})
                if spec.get('replicas') == 3:
                    checks["Initial replicas: 3"] = True
                
                # Check pod template spec
                template = spec.get('template', {})
                pod_spec = template.get('spec', {})
                
                # Check anti-affinity
                if pod_spec.get('affinity', {}).get('podAntiAffinity'):
                    checks["Pod anti-affinity"] = True
                
                # Check security context
                security_ctx = pod_spec.get('securityContext', {})
                if (security_ctx.get('runAsUser') == 1000 and 
                    security_ctx.get('runAsGroup') == 1000):
                    checks["Non-root user (1000:1000)"] = True
                
                # Check GPU selector
                affinity = pod_spec.get('affinity', {})
                node_aff = affinity.get('nodeAffinity', {})
                if node_aff:
                    checks["GPU node selector"] = True
                
                # Check health probes
                containers = pod_spec.get('containers', [])
                if containers and 'livenessProbe' in containers[0]:
                    checks["Health probes"] = True
                
                # Check service account
                if pod_spec.get('serviceAccountName') == 'ryot-sa':
                    checks["ServiceAccount: ryot-sa"] = True
            
            # Check Service
            if kind == 'Service' and name == 'ryot-service':
                checks["Service: ryot-service"] = True
            
            # Check HPA
            if kind == 'HorizontalPodAutoscaler':
                spec = doc.get('spec', {})
                if spec.get('minReplicas') == 1 and spec.get('maxReplicas') == 5:
                    checks["HPA: 1-5 replicas"] = True
            
            # Check PodDisruptionBudget
            if kind == 'PodDisruptionBudget':
                checks["PodDisruptionBudget"] = True
            
            # Check NetworkPolicy
            if kind == 'NetworkPolicy':
                checks["NetworkPolicy"] = True
            
            # Check PVC
            if kind == 'PersistentVolumeClaim' and name == 'ryot-models-pvc':
                checks["PVC: ryot-models-pvc"] = True
        
        # Display results
        passed = 0
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check}")
            if result:
                passed += 1
        
        print(f"\n{passed}/{len(checks)} checks passed")
        return passed == len(checks)
        
    except Exception as e:
        print(f"✗ Error checking specifications: {e}")
        return False


def main():
    """Main validation function."""
    deployment_dir = Path("infrastructure/kubernetes/deployments")
    
    if not deployment_dir.exists():
        print(f"Error: Directory not found: {deployment_dir}")
        return 1
    
    files = [
        deployment_dir / "ryot-llm-deployment.yaml",
        deployment_dir / "ryot-llm-secrets.yaml",
        deployment_dir / "ryot-llm-rbac.yaml",
    ]
    
    print("\n" + "="*70)
    print("PHASE 14: RYOT LLM KUBERNETES DEPLOYMENT VALIDATOR")
    print("="*70)
    
    results = {}
    for file_path in files:
        if file_path.exists():
            results[file_path.name] = validate_yaml_file(file_path)
        else:
            print(f"✗ File not found: {file_path}")
            results[file_path.name] = False
    
    # Check deployment specs
    deployment_valid = check_deployment_specs(deployment_dir / "ryot-llm-deployment.yaml")
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print('='*70)
    
    for file_name, valid in results.items():
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{status:10} | {file_name}")
    
    print(f"\nDeployment Specs: {'✓ PASS' if deployment_valid else '✗ FAIL'}")
    
    all_valid = all(results.values()) and deployment_valid
    
    if all_valid:
        print("\n✓ ALL VALIDATIONS PASSED")
        print("\nNext steps:")
        print("  1. Replace placeholder credentials in ryot-llm-secrets.yaml")
        print("  2. Update kustomization.yaml to include ryot-llm-deployment.yaml")
        print("  3. Deploy: kubectl apply -k infrastructure/kubernetes/")
        return 0
    else:
        print("\n✗ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
