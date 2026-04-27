"""Manual security review for local development."""
from pathlib import Path

crypto_files = [
    'core/crypto_primitives.py',
    'core/key_derivation.py',
    'core/secure_storage.py'
]

print("[*] Manual Security Review")
print("=" * 60)

for file_path in crypto_files:
    full_path = Path('S:/sigmalang') / file_path

    if not full_path.exists():
        print(f"[!] {file_path} - Not found")
        continue

    print(f"\n[*] {file_path}")
    print("-" * 40)

    with open(full_path, 'r') as f:
        content = f.read()

    issues = []

    # Check deprecated algorithms
    deprecated = ['MD5', 'SHA1', 'DES', 'RC4']
    for algo in deprecated:
        if algo in content:
            issues.append(f"[!] Deprecated algorithm: {algo}")

    # Check weak key sizes
    if 'RSA' in content and '1024' in content:
        issues.append("[FAIL] Weak RSA key: 1024 bits")

    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  [OK] No immediate issues")

print("\n" + "=" * 60)
print("For comprehensive review, use:")
print("  @CIPHER review S:\\sigmalang\\core\\crypto_primitives.py")
