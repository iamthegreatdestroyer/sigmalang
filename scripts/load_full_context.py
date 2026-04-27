"""Load entire ΣLANG codebase into Claude 1M context."""
import json
from pathlib import Path


def collect_source_files(root_dir, extensions=['.py', '.md']):
    """Collect all source files."""
    root = Path(root_dir)
    files = []

    include_dirs = ['core', 'sigmalang', 'tests', 'scripts', 'docs']

    for inc_dir in include_dirs:
        dir_path = root / inc_dir
        if dir_path.exists():
            for ext in extensions:
                files.extend(dir_path.rglob(f'*{ext}'))

    return sorted(files)

def generate_snapshot(root_dir='S:/sigmalang'):
    """Generate full context snapshot."""
    snapshot = {
        'project': 'ΣLANG',
        'version': '0.95',
        'status': 'Phase 2 Active',
        'files': {}
    }

    source_files = collect_source_files(root_dir)
    print(f"[*] Collecting {len(source_files)} files...")

    for file_path in source_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            rel_path = file_path.relative_to(root_dir)
            snapshot['files'][str(rel_path)] = {
                'content': content,
                'size': len(content),
                'lines': content.count('\n') + 1
            }
        except Exception as e:
            print(f"[!] Skipped {file_path}: {e}")

    total_chars = sum(f['size'] for f in snapshot['files'].values())
    total_lines = sum(f['lines'] for f in snapshot['files'].values())

    snapshot['stats'] = {
        'total_files': len(snapshot['files']),
        'total_characters': total_chars,
        'total_lines': total_lines,
        'estimated_tokens': total_chars // 4
    }

    print("\n[OK] Snapshot Complete:")
    print(f"   Files: {snapshot['stats']['total_files']}")
    print(f"   Lines: {snapshot['stats']['total_lines']:,}")
    print(f"   Est. Tokens: {snapshot['stats']['estimated_tokens']:,}")

    output_path = Path(root_dir) / 'full_context_snapshot.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n[OK] Saved: {output_path}")
    return snapshot

if __name__ == '__main__':
    generate_snapshot()
