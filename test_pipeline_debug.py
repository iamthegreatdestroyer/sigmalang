#!/usr/bin/env python3
"""Debug script to identify Bug 5A in full SigmaEncoder/SigmaDecoder pipeline."""

import sys
sys.path.insert(0, '.')

from core.encoder import SigmaEncoder, SigmaDecoder
from core.parser import SemanticParser

print('=== Full Pipeline Debug: Special Characters ===')
print()

# Create encoder/decoder - SigmaDecoder takes encoder as argument
encoder = SigmaEncoder()
decoder = SigmaDecoder(encoder)
parser = SemanticParser()

# Test data with special characters
test_data = {
    "key_with_newline\n": "value\nwith\nnewlines",
    "tab\tkey": "tab\tvalue",
    "quote\"key": "quote\"value"
}

print('Input data:')
for k, v in test_data.items():
    print(f'  {repr(k)}: {repr(v)}')

print()
print('=== Encoding ===')
encoded = encoder.encode(test_data)
print(f'Encoded bytes length: {len(encoded)}')
print(f'Encoded hex: {encoded.hex()[:200]}...')

print()
print('=== Decoding ===')
decoded = decoder.decode(encoded)
print(f'Decoded type: {type(decoded)}')

print()
print('=== Comparison ===')
for orig_key, orig_val in test_data.items():
    if orig_key in decoded:
        dec_val = decoded[orig_key]
        match = orig_val == dec_val
        print(f'Key {repr(orig_key)}:')
        print(f'  Original:  {repr(orig_val)}')
        print(f'  Decoded:   {repr(dec_val)}')
        print(f'  Match:     {match}')
        if not match:
            print(f'  MISMATCH DETECTED!')
    else:
        print(f'Key {repr(orig_key)}: NOT FOUND in decoded!')
        print(f'  Available keys: {[repr(k) for k in decoded.keys()]}')

print()
print('=== Full Decoded Structure ===')
print(decoded)
