#!/usr/bin/env python3
"""
ΣLANG Basic Encoding Example
============================

Demonstrates basic text encoding and decoding with ΣLANG.
"""

import sys
sys.path.insert(0, '..')

from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
from sigmalang.core.primitives import ExistentialPrimitive


def main():
    """Run basic encoding examples."""
    print("=" * 60)
    print("ΣLANG Basic Encoding Example")
    print("=" * 60)
    
    # Initialize encoder
    print("\n1. Initializing encoder...")
    encoder = SigmaEncoder()
    print("   ✓ Encoder ready")
    
    # Basic encoding
    print("\n2. Basic text encoding:")
    text = "Hello, world!"
    result = encoder.encode(text)
    
    print(f"   Input: '{text}'")
    print(f"   Original size: {len(text)} bytes")
    print(f"   Vector dimensions: {len(result.vector) if hasattr(result, 'vector') else 'N/A'}")
    
    # Encoding code-related text
    print("\n3. Code-related encoding:")
    code_text = "Create a Python function that sorts a list in descending order"
    result = encoder.encode(code_text)
    
    print(f"   Input: '{code_text}'")
    print(f"   Encoding type: {result.encoding_type if hasattr(result, 'encoding_type') else 'semantic'}")
    
    # Batch encoding
    print("\n4. Batch encoding:")
    texts = [
        "Machine learning fundamentals",
        "Deep neural networks",
        "Natural language processing",
        "Computer vision algorithms"
    ]
    
    for text in texts:
        result = encoder.encode(text)
        ratio = result.compression_ratio if hasattr(result, 'compression_ratio') else 1.0
        print(f"   '{text[:35]:35}' -> {ratio:.1f}x compression")
    
    # Decoding (if decoder available)
    print("\n5. Decoding vectors:")
    try:
        decoder = SigmaDecoder(encoder)
        encoded = encoder.encode("Hello, ΣLANG!")
        decoded = decoder.decode(encoded.vector)
        print(f"   Original: 'Hello, ΣLANG!'")
        print(f"   Decoded: '{decoded}'")
    except Exception as e:
        print(f"   Decoder not fully available: {e}")
    
    # Explore primitives
    print("\n6. Semantic primitives:")
    print("   Existential primitives (Tier 0):")
    for prim in list(ExistentialPrimitive)[:8]:
        print(f"      Σ[{prim.name}] = 0x{prim.value:02X}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
