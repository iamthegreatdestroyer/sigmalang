"""Quick ΣLANG validation for Claude Code Execution."""
import sys

sys.path.insert(0, 'S:/sigmalang')

print("[*] Quick ΣLANG Validation")
print("-" * 40)

# Test 1: Import check
try:
    from sigmalang.core import decoder, encoder
    print("[OK] Imports: OK")
except ImportError as e:
    print(f"[FAIL] Imports: FAILED - {e}")
    sys.exit(1)

# Test 2: Basic encode
try:
    enc = encoder.SigmaEncoder(dim=3072, quantization_ratio=192)
    test = "Hello ΣLANG"
    compressed = enc.encode(test)
    print(f"[OK] Encode: OK (compressed to {len(compressed.glyph_sequence)} glyphs)")
except Exception as e:
    print(f"[FAIL] Encode: FAILED - {e}")
    sys.exit(1)

# Test 3: Basic decode
try:
    dec = decoder.SigmaDecoder()
    reconstructed = dec.decode(compressed.glyph_sequence)
    print("[OK] Decode: OK")
except Exception as e:
    print(f"[FAIL] Decode: FAILED - {e}")
    sys.exit(1)

print("-" * 40)
print("[OK] All quick validation tests passed!")
