#!/usr/bin/env python3
"""
Fix all imports in test files to use 'sigmalang.core' instead of 'core'
"""

import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single test file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Skip files with encoding issues
        return False
    
    original = content
    
    # Replace "from core." with "from sigmalang.core."
    content = re.sub(r'^from core\.', 'from sigmalang.core.', content, flags=re.MULTILINE)
    
    # Replace "import core." with "import sigmalang.core."
    content = re.sub(r'^import core\.', 'import sigmalang.core.', content, flags=re.MULTILINE)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """Fix all test files."""
    test_dir = Path('tests')
    test_files = sorted(test_dir.glob('test_*.py'))
    
    updated = 0
    for test_file in test_files:
        if fix_imports_in_file(test_file):
            print(f"✓ Updated: {test_file.name}")
            updated += 1
        else:
            print(f"  No changes: {test_file.name}")
    
    print(f"\nTotal updated: {updated}/{len(test_files)} files")

if __name__ == '__main__':
    main()
