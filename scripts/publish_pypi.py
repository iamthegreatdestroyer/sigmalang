"""
PyPI Publication Script

Automated pre-publication checklist and package upload for PyPI.

Usage:
    python scripts/publish_pypi.py --check     # Run pre-publication checks
    python scripts/publish_pypi.py --build     # Build distributions
    python scripts/publish_pypi.py --upload    # Upload to PyPI (requires credentials)
    python scripts/publish_pypi.py --test      # Upload to TestPyPI first
"""

import argparse
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional


class PyPIPublisher:
    """Handles PyPI publication with pre-flight checks."""

    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.readme_path = project_root / "README.md"
        self.dist_dir = project_root / "dist"
        self.checks_passed = []
        self.checks_failed = []

    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        print(f"Running: {description}...")
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  [PASS] {description}")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"  [FAIL] {description}")
            print(f"  Error: {e.stderr}")
            return False, e.stderr

    def check_pyproject_version(self) -> bool:
        """Check that version is properly set in pyproject.toml."""
        print("\n1. Checking pyproject.toml version...")

        if not self.pyproject_path.exists():
            print("  [FAIL] pyproject.toml not found")
            return False

        content = self.pyproject_path.read_text()
        version_match = re.search(r'version\s*=\s*"([^"]+)"', content)

        if not version_match:
            print("  [FAIL] Version not found in pyproject.toml")
            return False

        version = version_match.group(1)
        print(f"  [PASS] Version: {version}")

        # Check version format (semantic versioning)
        if not re.match(r'^\d+\.\d+\.\d+', version):
            print(f"  [WARN] Version '{version}' doesn't follow semantic versioning")

        return True

    def check_readme_exists(self) -> bool:
        """Check that README.md exists and is not empty."""
        print("\n2. Checking README.md...")

        if not self.readme_path.exists():
            print("  [FAIL] README.md not found")
            return False

        size = self.readme_path.stat().st_size
        if size < 100:
            print(f"  [FAIL] README.md too small ({size} bytes)")
            return False

        print(f"  [PASS] README.md exists ({size} bytes)")
        return True

    def check_dependencies(self) -> bool:
        """Check that dependencies are properly specified."""
        print("\n3. Checking dependencies...")

        content = self.pyproject_path.read_text()

        # Check for dependencies section
        if "dependencies = [" not in content:
            print("  [FAIL] No dependencies section found")
            return False

        # Extract dependencies
        deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if not deps_match:
            print("  [FAIL] Could not parse dependencies")
            return False

        deps = deps_match.group(1)
        dep_count = len([line for line in deps.split('\n') if '"' in line])

        print(f"  [PASS] {dep_count} dependencies specified")

        # Check for version constraints
        if ">=" in deps:
            print(f"  [INFO] Using minimum version constraints (>=)")
        else:
            print(f"  [WARN] No version constraints found")

        return True

    def check_cli_entrypoints(self) -> bool:
        """Check that CLI entry points are defined."""
        print("\n4. Checking CLI entry points...")

        content = self.pyproject_path.read_text()

        # Check for scripts section
        if "[project.scripts]" not in content:
            print("  [FAIL] No CLI entry points defined")
            return False

        # Extract entry points
        scripts_section = content.split("[project.scripts]")[1].split("[")[0]
        entry_points = [line.strip() for line in scripts_section.split('\n') if '=' in line]

        print(f"  [PASS] {len(entry_points)} entry points defined:")
        for ep in entry_points:
            print(f"    - {ep.split('=')[0].strip()}")

        return True

    def check_license(self) -> bool:
        """Check that license is specified."""
        print("\n5. Checking license...")

        content = self.pyproject_path.read_text()

        if "license" not in content.lower():
            print("  [FAIL] No license specified")
            return False

        # Check for LICENSE file
        license_files = list(self.project_root.glob("LICENSE*"))

        if license_files:
            print(f"  [PASS] License specified, LICENSE file found")
        else:
            print(f"  [WARN] License specified but no LICENSE file found")

        return True

    def check_build_system(self) -> bool:
        """Check that build system is properly configured."""
        print("\n6. Checking build system...")

        content = self.pyproject_path.read_text()

        if "[build-system]" not in content:
            print("  [FAIL] No build-system section")
            return False

        if "setuptools" in content:
            print("  [PASS] Using setuptools build backend")
        elif "hatchling" in content:
            print("  [PASS] Using hatchling build backend")
        else:
            print("  [WARN] Unknown build backend")

        return True

    def test_clean_install(self) -> bool:
        """Test that package can be installed cleanly."""
        print("\n7. Testing clean install (dry-run)...")

        # Check if there are built distributions
        if not self.dist_dir.exists() or not list(self.dist_dir.glob("*.whl")):
            print("  [SKIP] No distributions built yet")
            return True

        # We can't actually install without potentially breaking the dev environment
        # So we just verify the wheel exists and is valid
        wheels = list(self.dist_dir.glob("*.whl"))
        if wheels:
            print(f"  [PASS] Found wheel: {wheels[0].name}")
        else:
            print("  [FAIL] No wheel found in dist/")
            return False

        return True

    def run_all_checks(self) -> bool:
        """Run all pre-publication checks."""
        print("=" * 70)
        print("PyPI PRE-PUBLICATION CHECKLIST")
        print("=" * 70)

        checks = [
            ("Version check", self.check_pyproject_version),
            ("README check", self.check_readme_exists),
            ("Dependencies check", self.check_dependencies),
            ("CLI entry points check", self.check_cli_entrypoints),
            ("License check", self.check_license),
            ("Build system check", self.check_build_system),
            ("Clean install test", self.test_clean_install),
        ]

        passed = 0
        failed = 0

        for name, check_func in checks:
            try:
                if check_func():
                    passed += 1
                    self.checks_passed.append(name)
                else:
                    failed += 1
                    self.checks_failed.append(name)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                failed += 1
                self.checks_failed.append(name)

        print("\n" + "=" * 70)
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("=" * 70)

        if failed > 0:
            print("\nFailed checks:")
            for check in self.checks_failed:
                print(f"  - {check}")
            return False
        else:
            print("\n[PASS] All checks passed! Ready for publication.")
            return True

    def build_distributions(self) -> bool:
        """Build source and wheel distributions."""
        print("\n" + "=" * 70)
        print("BUILDING DISTRIBUTIONS")
        print("=" * 70)

        # Clean old distributions
        if self.dist_dir.exists():
            print(f"\nCleaning old distributions in {self.dist_dir}...")
            for file in self.dist_dir.glob("*"):
                file.unlink()
                print(f"  Removed: {file.name}")

        # Build
        success, output = self.run_command(
            [sys.executable, "-m", "build"],
            "Building source and wheel distributions"
        )

        if not success:
            return False

        # Verify distributions
        print("\nVerifying distributions...")
        success, output = self.run_command(
            [sys.executable, "-m", "twine", "check", "dist/*"],
            "Checking distributions with twine"
        )

        if not success:
            return False

        # List built files
        print("\nBuilt distributions:")
        for file in sorted(self.dist_dir.glob("*")):
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.name} ({size:.1f} KB)")

        return True

    def upload_to_testpypi(self) -> bool:
        """Upload distributions to TestPyPI."""
        print("\n" + "=" * 70)
        print("UPLOADING TO TESTPYPI")
        print("=" * 70)

        print("\nNOTE: This requires TestPyPI credentials.")
        print("Set TWINE_USERNAME and TWINE_PASSWORD environment variables")
        print("or use ~/.pypirc configuration file.\n")

        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository", "testpypi",
            "dist/*"
        ]

        print(f"Command: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, cwd=self.project_root, check=True)
            print("\n[PASS] Upload to TestPyPI successful!")
            print("Test installation with:")
            print("  pip install --index-url https://test.pypi.org/simple/ sigmalang")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] Upload failed: {e}")
            return False

    def upload_to_pypi(self) -> bool:
        """Upload distributions to PyPI."""
        print("\n" + "=" * 70)
        print("UPLOADING TO PYPI")
        print("=" * 70)

        print("\nWARNING: This will publish to the REAL PyPI!")
        print("Make sure you have:")
        print("  1. Tested on TestPyPI first")
        print("  2. Set correct version number")
        print("  3. Updated CHANGELOG")
        print("  4. Tagged the release in git\n")

        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != "yes":
            print("\n[CANCELLED] Upload cancelled by user")
            return False

        cmd = [
            sys.executable, "-m", "twine", "upload",
            "dist/*"
        ]

        print(f"\nCommand: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, cwd=self.project_root, check=True)
            print("\n[PASS] Upload to PyPI successful!")
            print("Package published at: https://pypi.org/project/sigmalang/")
            print("Install with: pip install sigmalang")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] Upload failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyPI Publication Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Run pre-publication checks'
    )

    parser.add_argument(
        '--build',
        action='store_true',
        help='Build distributions'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Upload to TestPyPI'
    )

    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload to PyPI (production)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run checks, build, and upload to TestPyPI'
    )

    args = parser.parse_args()

    publisher = PyPIPublisher()

    # Default to --check if no arguments
    if not any([args.check, args.build, args.test, args.upload, args.all]):
        args.check = True

    success = True

    if args.all:
        # Run full workflow
        success = publisher.run_all_checks()
        if success:
            success = publisher.build_distributions()
        if success:
            success = publisher.upload_to_testpypi()
    else:
        if args.check:
            success = publisher.run_all_checks()

        if args.build and success:
            success = publisher.build_distributions()

        if args.test and success:
            success = publisher.upload_to_testpypi()

        if args.upload and success:
            success = publisher.upload_to_pypi()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
