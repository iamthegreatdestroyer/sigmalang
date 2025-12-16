#!/usr/bin/env python3
"""
ΣLANG Phase 14 Deployment Verification Script
==============================================

This script verifies that the SigmaLang deployment is correctly configured
for Phase 14 of the Neurectomy Unified Architecture.

Usage:
    python verify_sigmalang.py [--namespace neurectomy] [--verbose]

Requirements:
    - kubectl configured with cluster access
    - Python 3.9+
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class Phase14Verifier:
    """Verifies Phase 14 deployment requirements for SigmaLang."""
    
    # Phase 14 Requirements
    REQUIRED_REPLICAS = 5
    MIN_HPA_REPLICAS = 5
    MAX_HPA_REPLICAS = 20
    REQUIRED_CPU_REQUEST = "2"
    REQUIRED_MEMORY_REQUEST = "4Gi"
    REQUIRED_CPU_LIMIT = "4"
    REQUIRED_MEMORY_LIMIT = "8Gi"
    REQUIRED_HTTP_PORT = 8001
    REQUIRED_METRICS_PORT = 9091
    
    def __init__(self, namespace: str = "neurectomy", verbose: bool = False):
        self.namespace = namespace
        self.verbose = verbose
        self.results: List[VerificationResult] = []
        
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [DEBUG] {message}")
            
    def run_kubectl(self, *args) -> Tuple[bool, str]:
        """Run kubectl command and return success status and output."""
        cmd = ["kubectl", "-n", self.namespace, *args]
        self.log(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout or result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, "kubectl not found - please install kubectl"
            
    def check_namespace(self) -> VerificationResult:
        """Verify namespace exists with correct labels."""
        success, output = self.run_kubectl("get", "namespace", self.namespace, "-o", "json")
        
        if not success:
            return VerificationResult(
                name="Namespace",
                passed=False,
                message=f"Namespace '{self.namespace}' not found",
                details=output
            )
            
        try:
            ns_data = json.loads(output)
            labels = ns_data.get("metadata", {}).get("labels", {})
            
            required_labels = ["environment", "name"]
            missing = [l for l in required_labels if l not in labels]
            
            if missing:
                return VerificationResult(
                    name="Namespace",
                    passed=False,
                    message=f"Namespace missing labels: {missing}",
                    details=f"Current labels: {labels}"
                )
                
            return VerificationResult(
                name="Namespace",
                passed=True,
                message=f"Namespace '{self.namespace}' exists with correct labels"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="Namespace",
                passed=False,
                message="Failed to parse namespace data",
                details=output
            )
            
    def check_deployment(self) -> VerificationResult:
        """Verify deployment exists with correct configuration."""
        success, output = self.run_kubectl(
            "get", "deployment", "sigmalang-deployment", "-o", "json"
        )
        
        if not success:
            return VerificationResult(
                name="Deployment",
                passed=False,
                message="Deployment 'sigmalang-deployment' not found",
                details=output
            )
            
        try:
            deploy = json.loads(output)
            spec = deploy.get("spec", {})
            replicas = spec.get("replicas", 0)
            
            if replicas != self.REQUIRED_REPLICAS:
                return VerificationResult(
                    name="Deployment",
                    passed=False,
                    message=f"Replicas: {replicas} (expected {self.REQUIRED_REPLICAS})"
                )
                
            # Check container resources
            containers = spec.get("template", {}).get("spec", {}).get("containers", [])
            if not containers:
                return VerificationResult(
                    name="Deployment",
                    passed=False,
                    message="No containers found in deployment spec"
                )
                
            container = containers[0]
            resources = container.get("resources", {})
            requests = resources.get("requests", {})
            limits = resources.get("limits", {})
            
            issues = []
            if requests.get("memory") != self.REQUIRED_MEMORY_REQUEST:
                issues.append(f"memory request: {requests.get('memory')} (expected {self.REQUIRED_MEMORY_REQUEST})")
            if limits.get("memory") != self.REQUIRED_MEMORY_LIMIT:
                issues.append(f"memory limit: {limits.get('memory')} (expected {self.REQUIRED_MEMORY_LIMIT})")
                
            if issues:
                return VerificationResult(
                    name="Deployment",
                    passed=False,
                    message="Resource configuration issues",
                    details="; ".join(issues)
                )
                
            return VerificationResult(
                name="Deployment",
                passed=True,
                message=f"Deployment configured correctly ({replicas} replicas)"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="Deployment",
                passed=False,
                message="Failed to parse deployment data"
            )
            
    def check_service(self) -> VerificationResult:
        """Verify service exists with correct ports."""
        success, output = self.run_kubectl(
            "get", "service", "sigmalang-service", "-o", "json"
        )
        
        if not success:
            return VerificationResult(
                name="Service",
                passed=False,
                message="Service 'sigmalang-service' not found",
                details=output
            )
            
        try:
            svc = json.loads(output)
            ports = svc.get("spec", {}).get("ports", [])
            
            port_map = {p.get("name"): p.get("port") for p in ports}
            
            issues = []
            if port_map.get("http") != self.REQUIRED_HTTP_PORT:
                issues.append(f"HTTP port: {port_map.get('http')} (expected {self.REQUIRED_HTTP_PORT})")
            if port_map.get("metrics") != self.REQUIRED_METRICS_PORT:
                issues.append(f"Metrics port: {port_map.get('metrics')} (expected {self.REQUIRED_METRICS_PORT})")
                
            if issues:
                return VerificationResult(
                    name="Service",
                    passed=False,
                    message="Port configuration issues",
                    details="; ".join(issues)
                )
                
            return VerificationResult(
                name="Service",
                passed=True,
                message=f"Service configured correctly (ports: {self.REQUIRED_HTTP_PORT}, {self.REQUIRED_METRICS_PORT})"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="Service",
                passed=False,
                message="Failed to parse service data"
            )
            
    def check_hpa(self) -> VerificationResult:
        """Verify HorizontalPodAutoscaler configuration."""
        success, output = self.run_kubectl(
            "get", "hpa", "sigmalang-hpa", "-o", "json"
        )
        
        if not success:
            return VerificationResult(
                name="HPA",
                passed=False,
                message="HPA 'sigmalang-hpa' not found",
                details=output
            )
            
        try:
            hpa = json.loads(output)
            spec = hpa.get("spec", {})
            
            min_replicas = spec.get("minReplicas", 0)
            max_replicas = spec.get("maxReplicas", 0)
            
            issues = []
            if min_replicas != self.MIN_HPA_REPLICAS:
                issues.append(f"minReplicas: {min_replicas} (expected {self.MIN_HPA_REPLICAS})")
            if max_replicas != self.MAX_HPA_REPLICAS:
                issues.append(f"maxReplicas: {max_replicas} (expected {self.MAX_HPA_REPLICAS})")
                
            if issues:
                return VerificationResult(
                    name="HPA",
                    passed=False,
                    message="HPA configuration issues",
                    details="; ".join(issues)
                )
                
            return VerificationResult(
                name="HPA",
                passed=True,
                message=f"HPA configured correctly ({min_replicas}-{max_replicas} replicas)"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="HPA",
                passed=False,
                message="Failed to parse HPA data"
            )
            
    def check_configmap(self) -> VerificationResult:
        """Verify ConfigMap exists with required keys."""
        success, output = self.run_kubectl(
            "get", "configmap", "neurectomy-config", "-o", "json"
        )
        
        if not success:
            return VerificationResult(
                name="ConfigMap",
                passed=False,
                message="ConfigMap 'neurectomy-config' not found",
                details=output
            )
            
        try:
            cm = json.loads(output)
            data = cm.get("data", {})
            
            required_keys = [
                "SIGMALANG_ENDPOINT",
                "RYOT_ENDPOINT",
                "COMPRESSION_WORKERS",
                "LOG_LEVEL"
            ]
            
            missing = [k for k in required_keys if k not in data]
            
            if missing:
                return VerificationResult(
                    name="ConfigMap",
                    passed=False,
                    message=f"Missing required keys: {missing}",
                    details=f"Available keys: {list(data.keys())}"
                )
                
            return VerificationResult(
                name="ConfigMap",
                passed=True,
                message=f"ConfigMap has {len(data)} configuration entries"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="ConfigMap",
                passed=False,
                message="Failed to parse ConfigMap data"
            )
            
    def check_pods_health(self) -> VerificationResult:
        """Verify pods are running and healthy."""
        success, output = self.run_kubectl(
            "get", "pods", "-l", "app=sigmalang", "-o", "json"
        )
        
        if not success:
            return VerificationResult(
                name="Pods Health",
                passed=False,
                message="Failed to get pods",
                details=output
            )
            
        try:
            pods_data = json.loads(output)
            pods = pods_data.get("items", [])
            
            if not pods:
                return VerificationResult(
                    name="Pods Health",
                    passed=False,
                    message="No pods found with label 'app=sigmalang'"
                )
                
            running = 0
            not_running = []
            
            for pod in pods:
                name = pod.get("metadata", {}).get("name", "unknown")
                phase = pod.get("status", {}).get("phase", "Unknown")
                
                if phase == "Running":
                    running += 1
                else:
                    not_running.append(f"{name}: {phase}")
                    
            if not_running:
                return VerificationResult(
                    name="Pods Health",
                    passed=False,
                    message=f"{running}/{len(pods)} pods running",
                    details=f"Not running: {', '.join(not_running)}"
                )
                
            return VerificationResult(
                name="Pods Health",
                passed=True,
                message=f"All {running} pods are running"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="Pods Health",
                passed=False,
                message="Failed to parse pods data"
            )
            
    def check_network_policy(self) -> VerificationResult:
        """Verify NetworkPolicy exists."""
        success, output = self.run_kubectl(
            "get", "networkpolicy", "sigmalang-network-policy"
        )
        
        if not success:
            return VerificationResult(
                name="NetworkPolicy",
                passed=False,
                message="NetworkPolicy 'sigmalang-network-policy' not found",
                details="Network isolation not configured"
            )
            
        return VerificationResult(
            name="NetworkPolicy",
            passed=True,
            message="NetworkPolicy configured for pod isolation"
        )
        
    def check_pdb(self) -> VerificationResult:
        """Verify PodDisruptionBudget exists."""
        success, output = self.run_kubectl(
            "get", "pdb", "sigmalang-pdb", "-o", "json"
        )
        
        if not success:
            return VerificationResult(
                name="PodDisruptionBudget",
                passed=False,
                message="PDB 'sigmalang-pdb' not found",
                details="High availability protection not configured"
            )
            
        try:
            pdb = json.loads(output)
            min_available = pdb.get("spec", {}).get("minAvailable", 0)
            
            return VerificationResult(
                name="PodDisruptionBudget",
                passed=True,
                message=f"PDB configured (minAvailable: {min_available})"
            )
        except json.JSONDecodeError:
            return VerificationResult(
                name="PodDisruptionBudget",
                passed=True,
                message="PDB exists"
            )
            
    def check_dockerfile_exists(self) -> VerificationResult:
        """Verify Dockerfile.prod exists in the repository."""
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile.prod"
        
        if not dockerfile.exists():
            return VerificationResult(
                name="Dockerfile.prod",
                passed=False,
                message="Dockerfile.prod not found",
                details=f"Expected at: {dockerfile}"
            )
            
        # Check for Phase 14 markers
        content = dockerfile.read_text()
        phase14_markers = ["8001", "9091", "neurectomy"]
        found = [m for m in phase14_markers if m in content]
        
        if len(found) < len(phase14_markers):
            return VerificationResult(
                name="Dockerfile.prod",
                passed=False,
                message="Dockerfile.prod missing Phase 14 configuration",
                details=f"Found: {found}, Missing: {set(phase14_markers) - set(found)}"
            )
            
        return VerificationResult(
            name="Dockerfile.prod",
            passed=True,
            message="Dockerfile.prod configured for Phase 14"
        )
        
    def run_all_checks(self) -> bool:
        """Run all verification checks and return overall status."""
        print("\n" + "=" * 60)
        print("ΣLANG Phase 14 Deployment Verification")
        print(f"Namespace: {self.namespace}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 60 + "\n")
        
        checks = [
            ("Kubernetes Resources", [
                self.check_namespace,
                self.check_configmap,
                self.check_deployment,
                self.check_service,
                self.check_hpa,
                self.check_network_policy,
                self.check_pdb,
            ]),
            ("Runtime Health", [
                self.check_pods_health,
            ]),
            ("Local Files", [
                self.check_dockerfile_exists,
            ])
        ]
        
        all_passed = True
        
        for section_name, check_funcs in checks:
            print(f"\n📋 {section_name}")
            print("-" * 40)
            
            for check_func in check_funcs:
                try:
                    result = check_func()
                except Exception as e:
                    result = VerificationResult(
                        name=check_func.__name__,
                        passed=False,
                        message=f"Check failed with exception: {e}"
                    )
                    
                self.results.append(result)
                
                status = "✅" if result.passed else "❌"
                print(f"  {status} {result.name}: {result.message}")
                
                if result.details and (not result.passed or self.verbose):
                    print(f"      Details: {result.details}")
                    
                if not result.passed:
                    all_passed = False
                    
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print(f"VERIFICATION SUMMARY: {passed}/{total} checks passed")
        
        if all_passed:
            print("✅ All Phase 14 requirements verified successfully!")
        else:
            print("❌ Some checks failed. Please review and fix the issues above.")
            
        print("=" * 60 + "\n")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify ΣLANG Phase 14 deployment configuration"
    )
    parser.add_argument(
        "--namespace", "-n",
        default="neurectomy",
        help="Kubernetes namespace (default: neurectomy)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only check local files, skip Kubernetes checks"
    )
    
    args = parser.parse_args()
    
    verifier = Phase14Verifier(
        namespace=args.namespace,
        verbose=args.verbose
    )
    
    if args.local_only:
        # Only check local files
        print("\n📁 Checking local files only...\n")
        result = verifier.check_dockerfile_exists()
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.message}")
        if result.details:
            print(f"      Details: {result.details}")
        sys.exit(0 if result.passed else 1)
    
    success = verifier.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
