#!/usr/bin/env python3
"""Run test suite and generate a summary report."""

import subprocess
import sys

def run_tests():
    """Run the test suite and capture results."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_roundtrip.py",
        "tests/test_sigmalang.py",
        "-v", "--tb=short", "-q"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
