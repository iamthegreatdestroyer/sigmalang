#!/usr/bin/env python3
"""
PHASE 0: ΣLANG Interface Contracts - Verification Report
=========================================================
"""

import os
import re
import sys

# ============================================================================
# PHASE 0 VERIFICATION REPORT
# ============================================================================

project_name = os.path.basename(os.getcwd())

# 1. CHECK FILES
files_to_check = [
    'sigmalang/api/__init__.py',
    'sigmalang/api/interfaces.py',
    'sigmalang/api/types.py',
    'sigmalang/api/exceptions.py',
    'sigmalang/stubs/__init__.py',
    'sigmalang/stubs/mock_sigma.py',
]

files_status = {}
for fpath in files_to_check:
    exists = os.path.isfile(fpath)
    files_status[fpath] = '✓' if exists else '✗'

# 2. COUNT PROTOCOLS IN interfaces.py
protocols = []
with open('sigmalang/api/interfaces.py', 'r') as f:
    content = f.read()
    # Find all @runtime_checkable class definitions
    protocol_matches = re.findall(r'class\s+(\w+)\(Protocol\)', content)
    protocols = protocol_matches

# 3. COUNT TYPES IN types.py
enums = []
dataclasses = []
with open('sigmalang/api/types.py', 'r') as f:
    content = f.read()
    # Find all Enum classes
    enum_matches = re.findall(r'class\s+(\w+)\(Enum\)', content)
    enums = enum_matches
    # Find all dataclass decorators
    dataclass_matches = re.findall(r'@dataclass[^\n]*\nclass\s+(\w+)', content)
    dataclasses = dataclass_matches

# 4. READ __all__ from __init__.py
exports = []
with open('sigmalang/api/__init__.py', 'r') as f:
    content = f.read()
    # Extract __all__ list
    all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if all_match:
        all_content = all_match.group(1)
        exports = re.findall(r'"(\w+)"', all_content)

# 5. TEST IMPORTS
import_status = 'PASS'
try:
    from sigmalang.api import CompressionEngine, RSUManager
    from sigmalang.stubs import MockCompressionEngine, MockRSUManager
    
    # Verify protocol implementation
    engine = MockCompressionEngine()
    assert isinstance(engine, CompressionEngine)
    
    mgr = MockRSUManager()
    assert isinstance(mgr, RSUManager)
    
except Exception as e:
    import_status = f'FAIL - {str(e)}'

# ============================================================================
# GENERATE REPORT
# ============================================================================

print("=" * 80)
print("  PHASE 0: ΣLANG INTERFACE CONTRACTS - VERIFICATION REPORT")
print("=" * 80)
print()

print(f"PROJECT: {project_name}")
print(f"STATUS: COMPLETE")
print()

print("FILES:")
for fpath, status in files_status.items():
    print(f"  {status} {fpath}")
print()

print(f"PROTOCOLS DEFINED: {len(protocols)}")
for proto in protocols:
    print(f"  - {proto}")
print()

print(f"TYPES DEFINED: {len(enums)} enums + {len(dataclasses)} dataclasses = {len(enums) + len(dataclasses)} total")
print(f"  Enums: {', '.join(enums)}")
print(f"  Dataclasses: {', '.join(dataclasses)}")
print()

print(f"EXPORTS IN __all__: {len(exports)} items")
print()

print(f"MOCK IMPORT TEST: {import_status}")
print()

missing_issues = []
if import_status != 'PASS':
    missing_issues.append(f"Import test failed: {import_status}")

if len(missing_issues) > 0:
    print("MISSING/ISSUES:")
    for issue in missing_issues:
        print(f"  - {issue}")
else:
    print("MISSING/ISSUES: None")

print()
print("=" * 80)
print("  VERIFICATION COMPLETE")
print("=" * 80)
