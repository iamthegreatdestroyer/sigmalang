"""ΣLANG Roundtrip Validation for Claude Code Execution."""
import sys
import os

sys.path.insert(0, os.path.abspath('S:/sigmalang'))

try:
    from sigmalang.core.encoder import SigmaEncoder
    from sigmalang.core.decoder import SigmaDecoder
    from sigmalang.core.metrics import semantic_similarity
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

def test_basic_encode_decode():
    """Test 1: Basic roundtrip."""
    if not IMPORTS_OK:
        return {'test': 'basic_encode_decode', 'status': 'SKIP', 'reason': IMPORT_ERROR}

    original = "ΣLANG compresses semantic meaning using hyperdimensional vectors"

    try:
        encoder = SigmaEncoder(dim=3072, quantization_ratio=192)
        compressed = encoder.encode(original)

        decoder = SigmaDecoder()
        reconstructed = decoder.decode(compressed.glyph_sequence)

        similarity = semantic_similarity(original, reconstructed)
        compression_ratio = len(compressed.glyph_sequence) / len(original)

        assert similarity > 0.85, f"Fidelity too low: {similarity:.3f}"
        assert compression_ratio < 0.4, f"Compression insufficient: {compression_ratio:.1%}"

        return {
            'test': 'basic_encode_decode',
            'status': 'PASS',
            'similarity': round(similarity, 3),
            'compression_ratio': round(compression_ratio, 3)
        }
    except Exception as e:
        return {'test': 'basic_encode_decode', 'status': 'FAIL', 'error': str(e)}

def test_preserve_critical_tokens():
    """Test 2: Critical token preservation."""
    if not IMPORTS_OK:
        return {'test': 'preserve_critical_tokens', 'status': 'SKIP'}

    original = "Use AES-256-GCM with ECDH-P384"
    critical = ["AES-256-GCM", "ECDH-P384"]

    try:
        encoder = SigmaEncoder(dim=3072, quantization_ratio=192)
        compressed = encoder.encode(original, preserve_tokens=critical)

        decoder = SigmaDecoder()
        reconstructed = decoder.decode(compressed.glyph_sequence)

        for token in critical:
            assert token in reconstructed, f"Token '{token}' was modified"

        return {'test': 'preserve_critical_tokens', 'status': 'PASS'}
    except Exception as e:
        return {'test': 'preserve_critical_tokens', 'status': 'FAIL', 'error': str(e)}

def run_all_tests():
    """Execute all tests."""
    print("=" * 60)
    print("ΣLANG CLAUDE CODE EXECUTION VALIDATION SUITE")
    print("=" * 60)

    tests = [test_basic_encode_decode, test_preserve_critical_tokens]
    results = []

    for test_func in tests:
        result = test_func()
        results.append(result)

        icon = "[PASS]" if result['status'] == 'PASS' else "[FAIL]" if result['status'] == 'FAIL' else "[SKIP]"
        print(f"{icon} {result['test']}: {result['status']}")

        if result['status'] == 'PASS':
            for k, v in result.items():
                if k not in ['test', 'status']:
                    print(f"   {k}: {v}")
        elif result['status'] == 'FAIL':
            print(f"   Error: {result.get('error')}")
        print()

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')

    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return results

if __name__ == '__main__':
    run_all_tests()
