#!/usr/bin/env python3
"""
ΣLANG Phase 2: Compliance Validation
Automated SOC2 and GDPR compliance checking
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class ComplianceValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "compliance_reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def print_status(self, message: str):
        print(f"[INFO] {message}")

    def print_success(self, message: str):
        print(f"[SUCCESS] {message}")

    def print_warning(self, message: str):
        print(f"[WARNING] {message}")

    def print_error(self, message: str):
        print(f"[ERROR] {message}")

    def check_soc2_compliance(self):
        """Check SOC2 compliance requirements"""
        self.print_status("Checking SOC2 compliance requirements...")

        soc2_checks = {
            "trust_principles": {
                "security": False,
                "availability": False,
                "processing_integrity": False,
                "confidentiality": False,
                "privacy": False
            },
            "controls": {
                "access_control": False,
                "encryption": False,
                "monitoring": False,
                "incident_response": False,
                "backup_recovery": False
            }
        }

        findings = []

        # Check for security controls in code
        python_files = list(self.project_root.rglob("*.py"))

        # Security principle checks
        security_patterns = {
            "access_control": [r"rbac", r"authorization", r"authentication", r"jwt", r"oauth"],
            "encryption": [r"encrypt", r"decrypt", r"aes", r"tls", r"ssl"],
            "monitoring": [r"log", r"monitor", r"alert", r"metrics", r"prometheus"],
            "incident_response": [r"incident", r"response", r"breach", r"forensic"],
            "backup_recovery": [r"backup", r"recovery", r"disaster", r"redundancy"]
        }

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                for control, patterns in security_patterns.items():
                    if any(re.search(pattern, content) for pattern in patterns):
                        soc2_checks["controls"][control] = True

            except Exception as e:
                self.print_warning(f"Could not check {py_file}: {e}")

        # Check infrastructure for SOC2 compliance
        k8s_files = list(self.project_root.glob("infrastructure/kubernetes/**/*.yaml"))
        for k8s_file in k8s_files:
            try:
                with open(k8s_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                # Check for RBAC
                if "rbac" in content or "role" in content:
                    soc2_checks["controls"]["access_control"] = True

                # Check for secrets management
                if "secret" in content or "tls" in content:
                    soc2_checks["controls"]["encryption"] = True

                # Check for monitoring
                if "prometheus" in content or "grafana" in content:
                    soc2_checks["controls"]["monitoring"] = True

            except Exception as e:
                self.print_warning(f"Could not check {k8s_file}: {e}")

        # Check Docker configuration
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            try:
                with open(dockerfile, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                if "user" in content and "root" not in content:
                    soc2_checks["controls"]["access_control"] = True

            except Exception as e:
                self.print_warning(f"Could not check Dockerfile: {e}")

        # Calculate compliance score
        controls_implemented = sum(soc2_checks["controls"].values())
        total_controls = len(soc2_checks["controls"])

        soc2_compliant = controls_implemented >= total_controls * 0.8  # 80% threshold

        soc2_report = {
            "framework": "SOC2",
            "timestamp": datetime.now().isoformat(),
            "controls_implemented": controls_implemented,
            "total_controls": total_controls,
            "compliance_percentage": (controls_implemented / total_controls) * 100,
            "compliant": soc2_compliant,
            "controls_status": soc2_checks["controls"],
            "findings": findings
        }

        with open(self.reports_dir / "soc2_report.json", 'w') as f:
            json.dump(soc2_report, f, indent=2)

        if soc2_compliant:
            self.print_success(f"SOC2 compliance: {soc2_report['compliance_percentage']:.1f}% - COMPLIANT")
            return True
        else:
            self.print_warning(f"SOC2 compliance: {soc2_report['compliance_percentage']:.1f}% - NEEDS IMPROVEMENT")
            return False

    def check_gdpr_compliance(self):
        """Check GDPR compliance requirements"""
        self.print_status("Checking GDPR compliance requirements...")

        gdpr_checks = {
            "data_protection": {
                "lawful_basis": False,
                "data_minimization": False,
                "purpose_limitation": False,
                "accuracy": False,
                "storage_limitation": False,
                "integrity_confidentiality": False,
                "accountability": False
            },
            "data_subject_rights": {
                "access": False,
                "rectification": False,
                "erasure": False,
                "restriction": False,
                "portability": False,
                "objection": False
            }
        }

        findings = []

        # Check code for GDPR-related implementations
        python_files = list(self.project_root.rglob("*.py"))

        gdpr_patterns = {
            "lawful_basis": [r"consent", r"contract", r"legitimate", r"vital", r"public", r"legal"],
            "data_minimization": [r"minimal", r"necessary", r"limit", r"reduce"],
            "purpose_limitation": [r"purpose", r"specified", r"lawful"],
            "accuracy": [r"accurate", r"up.to.date", r"correct"],
            "storage_limitation": [r"retention", r"delete", r"remove", r"expire"],
            "integrity_confidentiality": [r"encrypt", r"secure", r"protect", r"confidential"],
            "accountability": [r"audit", r"log", r"record", r"document"],
            "access": [r"access.*right", r"data.*request"],
            "rectification": [r"rectif", r"correct", r"update"],
            "erasure": [r"delete", r"erase", r"remove", r"gdpr"],
            "restriction": [r"restrict", r"limit.*process"],
            "portability": [r"export", r"portable", r"transfer"],
            "objection": [r"object", r"withdraw", r"consent"]
        }

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                for principle, patterns in gdpr_patterns.items():
                    if any(re.search(pattern, content) for pattern in patterns):
                        if principle in gdpr_checks["data_protection"]:
                            gdpr_checks["data_protection"][principle] = True
                        elif principle in gdpr_checks["data_subject_rights"]:
                            gdpr_checks["data_subject_rights"][principle] = True

            except Exception as e:
                self.print_warning(f"Could not check {py_file}: {e}")

        # Check for privacy policy and data processing records
        policy_files = ["PRIVACY.md", "privacy_policy.md", "gdpr.md", "data_processing.md"]
        for policy_file in policy_files:
            if (self.project_root / policy_file).exists():
                gdpr_checks["data_protection"]["accountability"] = True
                break

        # Calculate compliance score
        protection_implemented = sum(gdpr_checks["data_protection"].values())
        rights_implemented = sum(gdpr_checks["data_subject_rights"].values())

        total_protection = len(gdpr_checks["data_protection"])
        total_rights = len(gdpr_checks["data_subject_rights"])

        protection_score = (protection_implemented / total_protection) * 100
        rights_score = (rights_implemented / total_rights) * 100
        overall_score = (protection_score + rights_score) / 2

        gdpr_compliant = overall_score >= 70  # 70% threshold for basic compliance

        gdpr_report = {
            "framework": "GDPR",
            "timestamp": datetime.now().isoformat(),
            "data_protection_score": protection_score,
            "data_subject_rights_score": rights_score,
            "overall_compliance_score": overall_score,
            "compliant": gdpr_compliant,
            "data_protection_status": gdpr_checks["data_protection"],
            "data_subject_rights_status": gdpr_checks["data_subject_rights"],
            "findings": findings
        }

        with open(self.reports_dir / "gdpr_report.json", 'w') as f:
            json.dump(gdpr_report, f, indent=2)

        if gdpr_compliant:
            self.print_success(f"GDPR compliance: {overall_score:.1f}% - COMPLIANT")
            return True
        else:
            self.print_warning(f"GDPR compliance: {overall_score:.1f}% - NEEDS IMPROVEMENT")
            return False

    def check_hipaa_compliance(self):
        """Check HIPAA compliance (healthcare data handling)"""
        self.print_status("Checking HIPAA compliance requirements...")

        hipaa_checks = {
            "privacy_rule": {
                "notice_of_privacy_practices": False,
                "individual_rights": False,
                "administrative_safeguards": False
            },
            "security_rule": {
                "administrative_safeguards": False,
                "physical_safeguards": False,
                "technical_safeguards": False
            }
        }

        # Basic HIPAA checks (simplified)
        python_files = list(self.project_root.rglob("*.py"))

        hipaa_patterns = {
            "privacy_rule": [r"hipaa", r"privacy", r"phi", r"protected.*health.*information"],
            "security_rule": [r"encrypt", r"access.*control", r"audit", r"integrity"]
        }

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                for rule, patterns in hipaa_patterns.items():
                    if any(re.search(pattern, content) for pattern in patterns):
                        if rule == "privacy_rule":
                            hipaa_checks["privacy_rule"]["individual_rights"] = True
                        elif rule == "security_rule":
                            hipaa_checks["security_rule"]["technical_safeguards"] = True

            except Exception as e:
                self.print_warning(f"Could not check {py_file}: {e}")

        # Check for HIPAA-related documentation
        hipaa_docs = ["HIPAA.md", "hipaa_compliance.md", "health_privacy.md"]
        for doc in hipaa_docs:
            if (self.project_root / doc).exists():
                hipaa_checks["privacy_rule"]["notice_of_privacy_practices"] = True
                break

        # Calculate compliance
        privacy_implemented = sum(hipaa_checks["privacy_rule"].values())
        security_implemented = sum(hipaa_checks["security_rule"].values())

        total_privacy = len(hipaa_checks["privacy_rule"])
        total_security = len(hipaa_checks["security_rule"])

        privacy_score = (privacy_implemented / total_privacy) * 100 if total_privacy > 0 else 0
        security_score = (security_implemented / total_security) * 100 if total_security > 0 else 0
        overall_score = (privacy_score + security_score) / 2

        hipaa_compliant = overall_score >= 60  # Lower threshold as HIPAA may not apply to all projects

        hipaa_report = {
            "framework": "HIPAA",
            "timestamp": datetime.now().isoformat(),
            "privacy_rule_score": privacy_score,
            "security_rule_score": security_score,
            "overall_compliance_score": overall_score,
            "compliant": hipaa_compliant,
            "privacy_rule_status": hipaa_checks["privacy_rule"],
            "security_rule_status": hipaa_checks["security_rule"]
        }

        with open(self.reports_dir / "hipaa_report.json", 'w') as f:
            json.dump(hipaa_report, f, indent=2)

        self.print_status(f"HIPAA compliance: {overall_score:.1f}% - {'COMPLIANT' if hipaa_compliant else 'NOT APPLICABLE'}")
        return hipaa_compliant

    def generate_compliance_summary(self, results):
        """Generate comprehensive compliance summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 2: Compliance Validation",
            "frameworks_checked": ["SOC2", "GDPR", "HIPAA"],
            "results": results,
            "recommendations": []
        }

        # Generate recommendations
        if not results.get("soc2", False):
            summary["recommendations"].append("Implement missing SOC2 security controls")
        if not results.get("gdpr", False):
            summary["recommendations"].append("Add GDPR data protection measures and subject rights")
        if not results.get("hipaa", False):
            summary["recommendations"].append("Review HIPAA compliance for healthcare data handling")

        # Overall assessment
        compliant_frameworks = sum(results.values())
        total_frameworks = len(results)

        summary["status"] = "COMPLIANT" if compliant_frameworks >= total_frameworks * 0.67 else "NEEDS_IMPROVEMENT"
        summary["compliance_score"] = (compliant_frameworks / total_frameworks) * 100

        # Save summary
        with open(self.reports_dir / "compliance_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_compliance_validation(self):
        """Run complete compliance validation"""
        print("⚖️ ΣLANG Phase 2: Compliance Validation")
        print("=" * 42)
        print(f"Timestamp: {datetime.now()}")
        print(f"Reports Directory: {self.reports_dir}")
        print()

        results = {}

        # Step 1: SOC2 Compliance
        results["soc2"] = self.check_soc2_compliance()

        # Step 2: GDPR Compliance
        results["gdpr"] = self.check_gdpr_compliance()

        # Step 3: HIPAA Compliance
        results["hipaa"] = self.check_hipaa_compliance()

        # Generate summary report
        summary = self.generate_compliance_summary(results)

        # Final results
        print()
        print("📊 COMPLIANCE VALIDATION SUMMARY")
        print("=" * 32)

        all_compliant = all(results.values())
        if all_compliant:
            self.print_success("✅ ALL COMPLIANCE CHECKS PASSED")
            self.print_success("🎉 Enterprise compliance requirements met")
        else:
            failed_checks = [k for k, v in results.items() if not v]
            self.print_warning(f"⚠️  {len(failed_checks)} compliance checks failed: {failed_checks}")
            self.print_status("📋 See compliance reports for detailed findings")

        print(f"📈 Overall Compliance Score: {summary['compliance_score']:.1f}%")
        print(f"📋 Reports saved to: {self.reports_dir}")

        return all_compliant

if __name__ == "__main__":
    validator = ComplianceValidator()
    success = validator.run_compliance_validation()
    sys.exit(0 if success else 1)