#!/usr/bin/env python3
"""Debug script to identify edge case failures at Glyph level."""

import sys
sys.path.insert(0, '.')

from core.primitives import Glyph, GlyphType, ExistentialPrimitive, GlyphStream

print('=== Glyph Encoding/Decoding Test ===')

# Test a string with newlines
test_str = 'value\nwith\nnewlines'
payload = test_str.encode('utf-8')

# Create glyph
glyph = Glyph(GlyphType.PRIMITIVE, ExistentialPrimitive.ENTITY, payload)

# Encode to bytes
encoded = glyph.to_bytes()
print(f'Original: {repr(test_str)}')
print(f'Payload bytes: {payload.hex()}')
print(f'Payload length: {len(payload)}')
print(f'Encoded bytes: {encoded.hex()}')
print(f'Encoded length: {len(encoded)}')

# Decode back
decoded_glyph, consumed = Glyph.from_bytes(encoded)
print()
print(f'Decoded type: {decoded_glyph.glyph_type}')
print(f'Decoded primitive_id: {decoded_glyph.primitive_id}')
print(f'Decoded payload: {decoded_glyph.payload.hex() if decoded_glyph.payload else None}')
if decoded_glyph.payload:
    try:
        decoded_str = decoded_glyph.payload.decode('utf-8')
        print(f'Decoded as str: {repr(decoded_str)}')
        print(f'Match: {test_str == decoded_str}')
    except UnicodeDecodeError as e:
        print(f'Decode error: {e}')
print(f'Consumed: {consumed}')

print()
print('=== GlyphStream Test ===')

# Test through GlyphStream
stream = GlyphStream(glyphs=[glyph], context_id=0, version=1, flags=0)
stream_bytes = stream.to_bytes()
print(f'Stream bytes: {stream_bytes.hex()}')

# Decode stream
decoded_stream = GlyphStream.from_bytes(stream_bytes)
print(f'Decoded glyphs: {len(decoded_stream.glyphs)}')

if decoded_stream.glyphs:
    g = decoded_stream.glyphs[0]
    print(f'First glyph payload: {g.payload.hex() if g.payload else None}')
    if g.payload:
        try:
            s = g.payload.decode('utf-8')
            print(f'Decoded str: {repr(s)}')
            print(f'Match: {test_str == s}')
        except UnicodeDecodeError as e:
            print(f'Decode error: {e}')
