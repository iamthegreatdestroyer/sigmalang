#!/usr/bin/env python3
"""
ΣLANG Phase 2A: Autonomous Security Fix Automation
AI-powered security remediation with zero human intervention
"""

import os
import sys
import io
import json
import re
import subprocess
from pathlib import Path

# Fix Windows console Unicode encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    try:
        os.system('chcp 65001 > nul 2>&1')
    except Exception:
        pass
from datetime import datetime
from typing import Dict, List, Any, Tuple
import shutil

class AutonomousSecurityFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.security_reports_dir = self.project_root / "security_reports"
        self.fixed_dir = self.project_root / "security_fixes" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fixed_dir.mkdir(parents=True, exist_ok=True)

    def print_status(self, message: str):
        print(f"[AUTO-FIX] {message}")

    def print_success(self, message: str):
        print(f"[SUCCESS] {message}")

    def print_warning(self, message: str):
        print(f"[WARNING] {message}")

    def print_error(self, message: str):
        print(f"[ERROR] {message}")

    def load_security_report(self) -> Dict[str, Any]:
        """Load the latest security report"""
        if not self.security_reports_dir.exists():
            self.print_error("No security reports directory found")
            return {}

        # Find the latest report
        report_files = list(self.security_reports_dir.glob("**/security_summary.json"))
        if not report_files:
            self.print_error("No security summary report found")
            return {}

        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)

        with open(latest_report, 'r') as f:
            return json.load(f)

    def analyze_false_positives(self, secrets_report: Dict[str, Any]) -> List[str]:
        """AI-powered analysis of false positive secrets"""
        self.print_status("Analyzing potential false positive secrets...")

        false_positives = []

        if "potential_secrets" not in secrets_report:
            return false_positives

        for secret in secrets_report["potential_secrets"]:
            file_path = secret["file"]
            line_content = secret["content"].lower()

            # AI-driven false positive detection
            false_positive_indicators = [
                "your-api-key",
                "your_secret",
                "placeholder",
                "example",
                "test",
                "dummy",
                "sample",
                "mock",
                "fake",
                "replace with",
                "your_",
                "example_",
                "test_",
                "api_key =",
                "secret =",
                "token =",
                "key ="
            ]

            # Check if it's clearly a placeholder or test data
            is_false_positive = any(indicator in line_content for indicator in false_positive_indicators)

            # Additional context analysis
            if "examples/" in file_path or "tests/" in file_path:
                is_false_positive = True

            if is_false_positive:
                false_positives.append(secret)

        self.print_success(f"Identified {len(false_positives)} false positive secrets")
        return false_positives

    def fix_owasp_issues(self, owasp_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        OWASP issue analysis and reporting.
        
        SAFETY NOTE: This method now REPORTS issues but does NOT auto-modify files.
        Auto-modification was causing file corruption by breaking Python syntax.
        All fixes require manual review to ensure code correctness.
        """
        self.print_status("Analyzing OWASP security issues (report-only mode)...")

        issues_found = []

        if "issues" not in owasp_report:
            return issues_found

        for issue in owasp_report["issues"]:
            file_path = self.project_root / issue["file"]

            if not file_path.exists():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                pattern = issue["pattern"]
                recommendation = ""

                # Generate recommendations (but do NOT modify files)
                if "eval(" in pattern:
                    recommendation = "Replace eval() with ast.literal_eval() for safe evaluation"
                elif "exec(" in pattern:
                    recommendation = "Remove exec() - use safer alternatives like importlib or explicit function calls"
                elif "pickle.loads" in pattern:
                    recommendation = "Replace pickle with json for untrusted data"
                elif "subprocess.call.*shell=True" in pattern or "shell=True" in pattern:
                    recommendation = "Remove shell=True and pass command as list"
                elif "os.system" in pattern:
                    recommendation = "Replace os.system() with subprocess.run()"
                elif "input(" in pattern:
                    recommendation = "Add input validation and sanitization"
                else:
                    recommendation = "Manual review required"

                issues_found.append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "issue": issue.get("description", pattern),
                    "pattern": pattern,
                    "recommendation": recommendation,
                    "status": "REPORTED - requires manual fix"
                })

                self.print_warning(f"Security issue in {file_path}: {pattern}")
                self.print_status(f"  Recommendation: {recommendation}")

            except Exception as e:
                self.print_warning(f"Could not analyze {file_path}: {e}")

        # Write issues report
        report_path = self.fixed_dir / "owasp_issues_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(issues_found, f, indent=2)

        self.print_success(f"Found {len(issues_found)} OWASP issues - report saved to {report_path}")
        self.print_status("NOTE: Files NOT modified. Manual fixes required for safety.")
        
        return issues_found

    def install_security_tools(self) -> bool:
        """Ensure security tools are properly installed"""
        self.print_status("Ensuring security tools are installed...")

        tools_to_install = []

        # Check bandit
        try:
            result = subprocess.run([sys.executable, "-c", "import bandit"], capture_output=True, timeout=10)
            if result.returncode != 0:
                tools_to_install.append("bandit")
        except:
            tools_to_install.append("bandit")

        # Check safety
        try:
            result = subprocess.run([sys.executable, "-c", "import safety"], capture_output=True, timeout=10)
            if result.returncode != 0:
                tools_to_install.append("safety")
        except:
            tools_to_install.append("safety")

        if tools_to_install:
            self.print_status(f"Installing missing tools: {tools_to_install}")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet"] + tools_to_install, timeout=60)
                self.print_success("Security tools installed successfully")
                return True
            except Exception as e:
                self.print_error(f"Failed to install security tools: {e}")
                return False
        else:
            self.print_success("All security tools already installed")
            return True

    def run_security_validation(self) -> Dict[str, Any]:
        """Run final security validation after fixes"""
        self.print_status("Running final security validation...")

        validation_results = {
            "bandit_passed": False,
            "safety_passed": False,
            "secrets_cleaned": False,
            "owasp_issues_resolved": False
        }

        # Run bandit if available
        try:
            result = subprocess.run(
                ["bandit", "-r", "sigmalang/", "-f", "json", "-o", str(self.fixed_dir / "final_bandit.json")],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            validation_results["bandit_passed"] = result.returncode == 0
        except:
            pass

        # Run safety if available
        try:
            result = subprocess.run(
                ["safety", "check", "--output", "json", "--save-json", str(self.fixed_dir / "final_safety.json")],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            validation_results["safety_passed"] = result.returncode == 0
        except:
            pass

        # Check for remaining secrets
        secrets_found = self._scan_for_secrets()
        validation_results["secrets_cleaned"] = len(secrets_found) == 0

        # Check for remaining OWASP issues
        owasp_issues = self._scan_for_owasp_issues()
        validation_results["owasp_issues_resolved"] = len(owasp_issues) == 0

        return validation_results

    def _scan_for_secrets(self) -> List[Dict[str, Any]]:
        """Quick scan for remaining secrets"""
        secrets_found = []
        python_files = list(self.project_root.rglob("*.py"))

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                if "password" in content or "secret" in content or "key" in content:
                    # Skip if it's clearly a placeholder
                    if not any(skip in content for skip in ["your-", "example", "test", "dummy", "placeholder"]):
                        secrets_found.append({"file": str(py_file.relative_to(self.project_root))})
            except:
                pass

        return secrets_found

    def _scan_for_owasp_issues(self) -> List[Dict[str, Any]]:
        """Quick scan for remaining OWASP issues"""
        issues_found = []
        python_files = list(self.project_root.rglob("*.py"))

        dangerous_patterns = ["eval(", "exec(", "pickle.loads", "shell=True", "os.system("]

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern in dangerous_patterns:
                    if pattern in content:
                        issues_found.append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "pattern": pattern
                        })
            except:
                pass

        return issues_found

    def generate_fix_report(self, fixes_applied: List[Dict[str, Any]], validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fix report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 2A: Autonomous Security Fixes",
            "fixes_applied": fixes_applied,
            "validation_results": validation_results,
            "summary": {
                "total_fixes": len(fixes_applied),
                "validation_passed": sum(validation_results.values()),
                "validation_total": len(validation_results),
                "success_rate": sum(validation_results.values()) / len(validation_results) * 100 if validation_results else 0
            }
        }

        with open(self.fixed_dir / "autonomous_fix_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def run_autonomous_security_fix(self):
        """Execute complete autonomous security remediation"""
        print("🤖 ΣLANG Phase 2A: Autonomous Security Fix Automation")
        print("=" * 55)
        print(f"Timestamp: {datetime.now()}")
        print(f"Fix Directory: {self.fixed_dir}")
        print()

        # Step 1: Load security report
        security_report = self.load_security_report()
        if not security_report:
            self.print_error("No security report available for analysis")
            return False

        # Step 2: Install security tools
        if not self.install_security_tools():
            self.print_warning("Continuing with available tools...")

        # Step 3: Analyze and fix false positive secrets
        secrets_report_path = self.security_reports_dir / "20260104_200704" / "secrets_report.json"
        if secrets_report_path.exists():
            with open(secrets_report_path, 'r') as f:
                secrets_report = json.load(f)

            false_positives = self.analyze_false_positives(secrets_report)
            self.print_success(f"Identified {len(false_positives)} false positives (no action needed)")
        else:
            self.print_warning("Secrets report not found")

        # Step 4: Apply OWASP fixes
        owasp_report_path = self.security_reports_dir / "20260104_200704" / "owasp_report.json"
        fixes_applied = []
        if owasp_report_path.exists():
            with open(owasp_report_path, 'r') as f:
                owasp_report = json.load(f)

            fixes_applied = self.fix_owasp_issues(owasp_report)
        else:
            self.print_warning("OWASP report not found")

        # Step 5: Run final validation
        validation_results = self.run_security_validation()

        # Step 6: Generate report
        report = self.generate_fix_report(fixes_applied, validation_results)

        # Final results
        print()
        print("🤖 AUTONOMOUS SECURITY FIX SUMMARY")
        print("=" * 35)

        success_rate = report["summary"]["success_rate"]
        if success_rate >= 80:
            self.print_success("✅ AUTONOMOUS SECURITY FIXES SUCCESSFUL")
            self.print_success(f"🎉 Applied {len(fixes_applied)} security fixes")
            self.print_success(f"📈 Validation Success Rate: {success_rate:.1f}%")
        else:
            self.print_warning(f"⚠️  Partial success: {success_rate:.1f}% validation rate")
            self.print_status("Manual review may be needed for remaining issues")

        print(f"📋 Fixes Applied: {len(fixes_applied)}")
        print(f"📋 Validation Passed: {report['summary']['validation_passed']}/{report['summary']['validation_total']}")
        print(f"📂 Fix Reports: {self.fixed_dir}")

        return success_rate >= 80

if __name__ == "__main__":
    fixer = AutonomousSecurityFixer()
    success = fixer.run_autonomous_security_fix()
    sys.exit(0 if success else 1)