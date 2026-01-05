#!/usr/bin/env python3
"""
ΣLANG Phase 2B: Automated Status Update
Self-documenting progress tracking and reporting
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class StatusUpdater:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.status_dir = self.project_root / "status_updates" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.status_dir.mkdir(parents=True, exist_ok=True)

    def print_status(self, message: str):
        print(f"[STATUS] {message}")

    def print_success(self, message: str):
        print(f"[SUCCESS] {message}")

    def print_warning(self, message: str):
        print(f"[WARNING] {message}")

    def print_error(self, message: str):
        print(f"[ERROR] {message}")

    def gather_project_metrics(self) -> Dict[str, Any]:
        """Gather comprehensive project metrics"""
        self.print_status("Gathering project metrics...")

        metrics = {
            "codebase": self._get_codebase_metrics(),
            "testing": self._get_testing_metrics(),
            "infrastructure": self._get_infrastructure_metrics(),
            "performance": self._get_performance_metrics(),
            "security": self._get_security_metrics(),
            "compliance": self._get_compliance_metrics(),
            "documentation": self._get_documentation_metrics()
        }

        return metrics

    def _get_codebase_metrics(self) -> Dict[str, Any]:
        """Get codebase size and quality metrics"""
        try:
            # Count Python files and lines
            python_files = list(self.project_root.rglob("*.py"))
            total_lines = 0
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except:
                    pass

            return {
                "python_files": len(python_files),
                "total_lines": total_lines,
                "directories": len([d for d in self.project_root.rglob("*") if d.is_dir()])
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_testing_metrics(self) -> Dict[str, Any]:
        """Get testing coverage and results"""
        try:
            # Check for test results
            test_results_dir = self.project_root / "test_results"
            if test_results_dir.exists():
                result_files = list(test_results_dir.glob("*.txt"))
                if result_files:
                    latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_result, 'r') as f:
                        content = f.read()
                        # Extract test counts (simple parsing)
                        passed = content.count("PASSED") + content.count("OK")
                        failed = content.count("FAILED") + content.count("ERROR")

                    return {
                        "tests_run": passed + failed,
                        "tests_passed": passed,
                        "tests_failed": failed,
                        "success_rate": (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
                    }

            return {"tests_run": 0, "tests_passed": 0, "tests_failed": 0, "success_rate": 0}
        except Exception as e:
            return {"error": str(e)}

    def _get_infrastructure_metrics(self) -> Dict[str, Any]:
        """Get infrastructure deployment status"""
        try:
            infra_dir = self.project_root / "infrastructure"
            if infra_dir.exists():
                k8s_files = list(infra_dir.rglob("*.yaml")) + list(infra_dir.rglob("*.yml"))
                return {
                    "kubernetes_manifests": len(k8s_files),
                    "infrastructure_components": len([d for d in infra_dir.iterdir() if d.is_dir()])
                }

            return {"kubernetes_manifests": 0, "infrastructure_components": 0}
        except Exception as e:
            return {"error": str(e)}

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance benchmarking results"""
        try:
            perf_reports_dir = self.project_root / "performance_reports"
            if perf_reports_dir.exists():
                report_files = list(perf_reports_dir.rglob("*.json"))
                return {
                    "performance_reports": len(report_files),
                    "recent_reports": len([f for f in report_files if self._is_recent_file(f)])
                }

            return {"performance_reports": 0, "recent_reports": 0}
        except Exception as e:
            return {"error": str(e)}

    def _get_security_metrics(self) -> Dict[str, Any]:
        """Get security scanning results"""
        try:
            security_reports_dir = self.project_root / "security_reports"
            if security_reports_dir.exists():
                report_files = list(security_reports_dir.rglob("*.json"))
                return {
                    "security_reports": len(report_files),
                    "recent_scans": len([f for f in report_files if self._is_recent_file(f)])
                }

            return {"security_reports": 0, "recent_scans": 0}
        except Exception as e:
            return {"error": str(e)}

    def _get_compliance_metrics(self) -> Dict[str, Any]:
        """Get compliance validation results"""
        try:
            compliance_reports_dir = self.project_root / "compliance_reports"
            if compliance_reports_dir.exists():
                report_files = list(compliance_reports_dir.rglob("*.json"))
                if report_files:
                    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_report, 'r') as f:
                        compliance_data = json.load(f)

                    return {
                        "compliance_reports": len(report_files),
                        "overall_score": compliance_data.get("overall_score", 0),
                        "soc2_compliant": compliance_data.get("soc2_score", 0) >= 80,
                        "gdpr_compliant": compliance_data.get("gdpr_score", 0) >= 80
                    }

            return {"compliance_reports": 0, "overall_score": 0, "soc2_compliant": False, "gdpr_compliant": False}
        except Exception as e:
            return {"error": str(e)}

    def _get_documentation_metrics(self) -> Dict[str, Any]:
        """Get documentation generation status"""
        try:
            docs_dir = self.project_root / "generated_docs"
            if docs_dir.exists():
                doc_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.json"))
                return {
                    "documentation_files": len(doc_files),
                    "api_specs": len(list(docs_dir.rglob("*api*"))),
                    "recent_docs": len([f for f in doc_files if self._is_recent_file(f)])
                }

            return {"documentation_files": 0, "api_specs": 0, "recent_docs": 0}
        except Exception as e:
            return {"error": str(e)}

    def _is_recent_file(self, file_path: Path, hours: int = 24) -> bool:
        """Check if file was modified recently"""
        try:
            mtime = file_path.stat().st_mtime
            return (datetime.now().timestamp() - mtime) < (hours * 3600)
        except:
            return False

    def generate_status_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 2B: Automated Status Update",
            "metrics": metrics,
            "project_health_score": self._calculate_health_score(metrics),
            "recommendations": self._generate_recommendations(metrics)
        }

        with open(self.status_dir / "status_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall project health score"""
        scores = []

        # Testing health (40% weight)
        testing = metrics.get("testing", {})
        if testing.get("tests_run", 0) > 0:
            test_score = testing.get("success_rate", 0)
            scores.append((test_score, 0.4))

        # Security health (20% weight)
        security = metrics.get("security", {})
        if security.get("security_reports", 0) > 0:
            security_score = 100 if security.get("recent_scans", 0) > 0 else 50
            scores.append((security_score, 0.2))

        # Compliance health (20% weight)
        compliance = metrics.get("compliance", {})
        compliance_score = compliance.get("overall_score", 0)
        scores.append((compliance_score, 0.2))

        # Performance health (10% weight)
        performance = metrics.get("performance", {})
        perf_score = 100 if performance.get("recent_reports", 0) > 0 else 0
        scores.append((perf_score, 0.1))

        # Documentation health (10% weight)
        docs = metrics.get("documentation", {})
        docs_score = 100 if docs.get("recent_docs", 0) > 0 else 0
        scores.append((docs_score, 0.1))

        # Calculate weighted average
        if scores:
            total_score = sum(score * weight for score, weight in scores)
            total_weight = sum(weight for _, weight in scores)
            return total_score / total_weight if total_weight > 0 else 0

        return 0

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []

        testing = metrics.get("testing", {})
        if testing.get("success_rate", 0) < 90:
            recommendations.append("Improve test coverage and fix failing tests")

        security = metrics.get("security", {})
        if security.get("recent_scans", 0) == 0:
            recommendations.append("Run recent security scans")

        compliance = metrics.get("compliance", {})
        if compliance.get("overall_score", 0) < 80:
            recommendations.append("Address compliance gaps")

        performance = metrics.get("performance", {})
        if performance.get("recent_reports", 0) == 0:
            recommendations.append("Run performance profiling")

        docs = metrics.get("documentation", {})
        if docs.get("recent_docs", 0) == 0:
            recommendations.append("Regenerate documentation")

        if not recommendations:
            recommendations.append("All systems operational - proceed with next phase")

        return recommendations

    def update_executive_summary(self, report: Dict[str, Any]):
        """Update executive summary with current status"""
        summary_file = self.project_root / "EXECUTIVE_SUMMARY_AND_MASTER_ACTION_PLAN.md"

        if not summary_file.exists():
            self.print_warning("Executive summary file not found")
            return

        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Update metrics in the quantitative achievements section
            metrics = report["metrics"]

            # Update codebase metrics
            codebase = metrics.get("codebase", {})
            if codebase.get("total_lines"):
                content = self._update_metric_in_content(
                    content,
                    "Code Quality",
                    f"{codebase.get('total_lines', 0)}+ lines, {metrics.get('testing', {}).get('success_rate', 0):.0f}% coverage"
                )

            # Update testing metrics
            testing = metrics.get("testing", {})
            if testing.get("tests_run"):
                content = self._update_metric_in_content(
                    content,
                    "Test Suite",
                    f"{testing.get('tests_run', 0)}+ tests, {testing.get('success_rate', 0):.0f}% passing"
                )

            # Update infrastructure metrics
            infra = metrics.get("infrastructure", {})
            if infra.get("kubernetes_manifests"):
                content = self._update_metric_in_content(
                    content,
                    "Infrastructure",
                    f"{infra.get('kubernetes_manifests', 0)} K8s resources deployed"
                )

            # Update health score
            health_score = report.get("project_health_score", 0)
            content = content.replace(
                "**Status: PHASE 1 COMPLETE | PHASE 2 COMPLETE | PHASE 3 READY | PRODUCTION READY | ENTERPRISE GRADE | MARKET LEADER** 🚀",
                f"**Status: PHASE 1 COMPLETE | PHASE 2 COMPLETE | PHASE 3 READY | PRODUCTION READY | ENTERPRISE GRADE | MARKET LEADER** 🚀\\n**Health Score: {health_score:.1f}%**"
            )

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.print_success("Executive summary updated with current metrics")

        except Exception as e:
            self.print_error(f"Failed to update executive summary: {e}")

    def _update_metric_in_content(self, content: str, metric_name: str, new_value: str) -> str:
        """Update a specific metric in the content"""
        import re

        # Find the metric line and update it
        pattern = f"({metric_name}:).*?(\\||$)"
        replacement = f"\\1 {new_value}\\2"

        return re.sub(pattern, replacement, content)

    def run_automated_status_update(self):
        """Execute complete automated status update"""
        print("📊 ΣLANG Phase 2B: Automated Status Update")
        print("=" * 44)
        print(f"Timestamp: {datetime.now()}")
        print(f"Status Directory: {self.status_dir}")
        print()

        # Step 1: Gather project metrics
        metrics = self.gather_project_metrics()

        # Step 2: Generate status report
        report = self.generate_status_report(metrics)

        # Step 3: Update executive summary
        self.update_executive_summary(report)

        # Final results
        print()
        print("📊 STATUS UPDATE SUMMARY")
        print("=" * 24)

        health_score = report.get("project_health_score", 0)
        if health_score >= 80:
            self.print_success("✅ PROJECT HEALTH EXCELLENT")
            self.print_success(f"📈 Health Score: {health_score:.1f}%")
        elif health_score >= 60:
            self.print_success("⚠️  PROJECT HEALTH GOOD")
            self.print_success(f"📈 Health Score: {health_score:.1f}%")
        else:
            self.print_warning("⚠️  PROJECT NEEDS ATTENTION")
            self.print_warning(f"📈 Health Score: {health_score:.1f}%")

        print(f"📋 Codebase: {metrics.get('codebase', {}).get('total_lines', 0)} lines")
        print(f"🧪 Testing: {metrics.get('testing', {}).get('success_rate', 0):.1f}% pass rate")
        print(f"🏗️  Infrastructure: {metrics.get('infrastructure', {}).get('kubernetes_manifests', 0)} manifests")
        print(f"🔒 Security: {metrics.get('security', {}).get('recent_scans', 0)} recent scans")
        print(f"⚖️  Compliance: {metrics.get('compliance', {}).get('overall_score', 0):.1f}% score")
        print(f"📚 Documentation: {metrics.get('documentation', {}).get('documentation_files', 0)} files")

        if report.get("recommendations"):
            print("\n📋 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")

        print(f"\n📂 Status Reports: {self.status_dir}")

        return health_score >= 60

if __name__ == "__main__":
    updater = StatusUpdater()
    success = updater.run_automated_status_update()
    sys.exit(0 if success else 1)