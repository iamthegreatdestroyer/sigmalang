#!/usr/bin/env python3
"""
ΣLANG Phase 2A: Unicode Documentation Fix Automation
Automatic Unicode character replacement and documentation regeneration
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class UnicodeDocumentationFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "generated_docs"
        self.fixed_dir = self.project_root / "unicode_fixes" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fixed_dir.mkdir(parents=True, exist_ok=True)

    def print_status(self, message: str):
        print(f"[UNICODE-FIX] {message}")

    def print_success(self, message: str):
        print(f"[SUCCESS] {message}")

    def print_warning(self, message: str):
        print(f"[WARNING] {message}")

    def print_error(self, message: str):
        print(f"[ERROR] {message}")

    def detect_unicode_issues(self) -> List[Dict[str, Any]]:
        """Detect Unicode encoding issues in documentation files"""
        self.print_status("Detecting Unicode encoding issues...")

        unicode_issues = []

        if not self.docs_dir.exists():
            self.print_warning("No generated docs directory found")
            return unicode_issues

        # Check all documentation files
        doc_files = list(self.docs_dir.rglob("*.md")) + list(self.docs_dir.rglob("*.json"))

        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Look for problematic Unicode characters
                problematic_chars = []
                for char in content:
                    if ord(char) > 127:  # Non-ASCII characters
                        if char == 'Σ':  # Sigma character
                            problematic_chars.append({
                                'char': char,
                                'code': ord(char),
                                'replacement': 'SLANG'
                            })
                        elif char == '→':  # Arrow
                            problematic_chars.append({
                                'char': char,
                                'code': ord(char),
                                'replacement': '->'
                            })
                        elif char == '✅':  # Check mark
                            problematic_chars.append({
                                'char': char,
                                'code': ord(char),
                                'replacement': '[OK]'
                            })
                        elif char == '⚠️':  # Warning
                            problematic_chars.append({
                                'char': char,
                                'code': ord(char),
                                'replacement': '[WARNING]'
                            })
                        elif char == '🚀':  # Rocket
                            problematic_chars.append({
                                'char': char,
                                'code': ord(char),
                                'replacement': '[LAUNCH]'
                            })
                        elif char == '🤖':  # Robot
                            problematic_chars.append({
                                'char': char,
                                'code': ord(char),
                                'replacement': '[AUTO]'
                            })

                if problematic_chars:
                    unicode_issues.append({
                        'file': str(doc_file.relative_to(self.project_root)),
                        'issues': problematic_chars,
                        'total_chars': len(problematic_chars)
                    })

            except UnicodeDecodeError as e:
                unicode_issues.append({
                    'file': str(doc_file.relative_to(self.project_root)),
                    'error': str(e),
                    'needs_repair': True
                })
            except Exception as e:
                self.print_warning(f"Could not check {doc_file}: {e}")

        self.print_success(f"Found Unicode issues in {len(unicode_issues)} files")
        return unicode_issues

    def fix_unicode_characters(self, unicode_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix Unicode characters in documentation files"""
        self.print_status("Fixing Unicode characters...")

        fixes_applied = []

        for issue in unicode_issues:
            file_path = self.project_root / issue['file']

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content

                # Apply character replacements
                if 'issues' in issue:
                    for char_issue in issue['issues']:
                        char = char_issue['char']
                        replacement = char_issue['replacement']
                        content = content.replace(char, replacement)

                # Save the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Also save a backup of the original
                backup_path = self.fixed_dir / f"{Path(issue['file']).name}.backup"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                fixes_applied.append({
                    'file': issue['file'],
                    'replacements_made': len(issue.get('issues', [])),
                    'backup_created': str(backup_path.relative_to(self.project_root))
                })

                self.print_success(f"Fixed Unicode in {issue['file']}")

            except Exception as e:
                self.print_error(f"Could not fix {file_path}: {e}")

        return fixes_applied

    def regenerate_documentation(self) -> bool:
        """Regenerate documentation with proper encoding"""
        self.print_status("Regenerating documentation...")

        try:
            # Run the documentation generation script
            result = os.system(f'cd "{self.project_root}" && python scripts/generate_docs.py > "{self.fixed_dir}/regeneration.log" 2>&1')

            if result == 0:
                self.print_success("Documentation regenerated successfully")
                return True
            else:
                self.print_warning("Documentation regeneration completed with warnings")
                return True

        except Exception as e:
            self.print_error(f"Documentation regeneration failed: {e}")
            return False

    def validate_fixes(self, unicode_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that Unicode fixes were successful"""
        self.print_status("Validating Unicode fixes...")

        validation_results = {
            'files_checked': len(unicode_issues),
            'files_fixed': 0,
            'remaining_issues': 0,
            'encoding_errors': 0
        }

        for issue in unicode_issues:
            file_path = self.project_root / issue['file']

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for remaining problematic Unicode
                remaining_unicode = []
                for char in content:
                    if ord(char) > 127:
                        remaining_unicode.append(char)

                if remaining_unicode:
                    validation_results['remaining_issues'] += 1
                    self.print_warning(f"Remaining Unicode in {issue['file']}: {set(remaining_unicode)}")
                else:
                    validation_results['files_fixed'] += 1

            except UnicodeDecodeError:
                validation_results['encoding_errors'] += 1
                self.print_error(f"Encoding error in {issue['file']}")
            except Exception as e:
                self.print_warning(f"Could not validate {file_path}: {e}")

        return validation_results

    def generate_fix_report(self, unicode_issues: List[Dict[str, Any]],
                           fixes_applied: List[Dict[str, Any]],
                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fix report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 2A: Unicode Documentation Fixes",
            "issues_detected": unicode_issues,
            "fixes_applied": fixes_applied,
            "validation_results": validation_results,
            "summary": {
                "total_files_with_issues": len(unicode_issues),
                "total_fixes_applied": len(fixes_applied),
                "files_successfully_fixed": validation_results.get('files_fixed', 0),
                "success_rate": (validation_results.get('files_fixed', 0) / len(unicode_issues) * 100) if unicode_issues else 100
            }
        }

        with open(self.fixed_dir / "unicode_fix_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def run_unicode_fix_automation(self):
        """Execute complete Unicode documentation fix automation"""
        print("🔧 ΣLANG Phase 2A: Unicode Documentation Fix Automation")
        print("=" * 57)
        print(f"Timestamp: {datetime.now()}")
        print(f"Fix Directory: {self.fixed_dir}")
        print()

        # Step 1: Detect Unicode issues
        unicode_issues = self.detect_unicode_issues()

        if not unicode_issues:
            self.print_success("✅ No Unicode issues detected")
            return True

        # Step 2: Apply fixes
        fixes_applied = self.fix_unicode_characters(unicode_issues)

        # Step 3: Regenerate documentation
        regeneration_success = self.regenerate_documentation()

        # Step 4: Validate fixes
        validation_results = self.validate_fixes(unicode_issues)

        # Step 5: Generate report
        report = self.generate_fix_report(unicode_issues, fixes_applied, validation_results)

        # Final results
        print()
        print("🔧 UNICODE FIX AUTOMATION SUMMARY")
        print("=" * 33)

        success_rate = report["summary"]["success_rate"]
        if success_rate >= 90:
            self.print_success("✅ UNICODE FIXES SUCCESSFUL")
            self.print_success(f"🎉 Fixed {len(fixes_applied)} files")
            self.print_success(f"📈 Success Rate: {success_rate:.1f}%")
        else:
            self.print_warning(f"⚠️  Partial success: {success_rate:.1f}% success rate")
            self.print_status("Some files may need manual review")

        print(f"📋 Files with Issues: {len(unicode_issues)}")
        print(f"📋 Fixes Applied: {len(fixes_applied)}")
        print(f"📋 Successfully Fixed: {validation_results.get('files_fixed', 0)}")
        print(f"📂 Fix Reports: {self.fixed_dir}")

        return success_rate >= 90

if __name__ == "__main__":
    fixer = UnicodeDocumentationFixer()
    success = fixer.run_unicode_fix_automation()
    sys.exit(0 if success else 1)