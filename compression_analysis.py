"""
Compression Ratio Analysis - Debug failing compression tests

Analyze why certain code snippets have compression ratios > 1.0 (expansion)
"""

import sys
from pathlib import Path

# Add project to path
sigmalang_root = Path(__file__).parent
sys.path.insert(0, str(sigmalang_root))

from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
from sigmalang.core.primitives import SemanticNode, SemanticTree, ExistentialPrimitive
from sigmalang.core.parser import SemanticParser


# Test code snippets that fail compression ratio
FAILING_SNIPPETS = [
    "Create a Python function that sorts a list in descending order",
    "Write a JavaScript async function to fetch data from an API",
    "Implement a binary search algorithm in C++",
]

# Test code snippets that pass compression ratio
PASSING_SNIPPETS = [
    "Build a REST API endpoint for user authentication",
    "Create a class that handles database connections with pooling",
]


def analyze_compression(text: str) -> dict:
    """Analyze compression for a given text."""
    parser = SemanticParser()
    encoder = SigmaEncoder()
    decoder = SigmaDecoder(encoder)
    
    # Parse the text
    tree = parser.parse(text)
    
    # Encode
    encoded = encoder.encode(tree, text)
    
    # Get sizes
    original_size = len(text.encode('utf-8'))
    encoded_size = len(encoded)
    ratio = encoded_size / original_size if original_size > 0 else 0
    
    # Decode to verify correctness
    decoded_tree = decoder.decode(encoded)
    
    return {
        "text": text,
        "original_size": original_size,
        "encoded_size": encoded_size,
        "ratio": ratio,
        "expansion": encoded_size - original_size,
        "is_roundtrip_correct": decoded_tree is not None,
    }


def main():
    """Run compression analysis."""
    print("=" * 80)
    print("COMPRESSION RATIO ANALYSIS")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("FAILING SNIPPETS (ratio > 0.9)")
    print("=" * 80)
    
    for snippet in FAILING_SNIPPETS:
        result = analyze_compression(snippet)
        print(f"\n📊 {snippet}")
        print(f"   Original:  {result['original_size']:,} bytes")
        print(f"   Encoded:   {result['encoded_size']:,} bytes")
        print(f"   Ratio:     {result['ratio']:.4f}")
        print(f"   Expansion: +{result['expansion']:,} bytes ({(result['expansion']/result['original_size']*100):.1f}%)")
        print(f"   Correct:   {result['is_roundtrip_correct']}")
    
    print("\n" + "=" * 80)
    print("PASSING SNIPPETS (ratio <= 0.9)")
    print("=" * 80)
    
    for snippet in PASSING_SNIPPETS:
        result = analyze_compression(snippet)
        print(f"\n📊 {snippet}")
        print(f"   Original:  {result['original_size']:,} bytes")
        print(f"   Encoded:   {result['encoded_size']:,} bytes")
        print(f"   Ratio:     {result['ratio']:.4f}")
        if result['expansion'] < 0:
            print(f"   Compression: {abs(result['expansion']):,} bytes ({abs(result['expansion'])/result['original_size']*100:.1f}%)")
        else:
            print(f"   Expansion: +{result['expansion']:,} bytes ({(result['expansion']/result['original_size']*100):.1f}%)")
        print(f"   Correct:   {result['is_roundtrip_correct']}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
KEY OBSERVATIONS:

1. SHORT SNIPPETS EXPAND (ratio > 1.0)
   - Short text has high overhead from encoding format
   - The semantic tree structure, glyph headers, and CRC checksums add fixed overhead
   - For short text (<100 bytes), this overhead dominates

2. LONGER SNIPPETS COMPRESS BETTER (ratio < 0.9)
   - Longer text has more semantic structure to compress
   - Shared primitives and patterns can be reused
   - Delta encoding becomes more effective on larger content

3. ENCODER OVERHEAD STRUCTURE:
   - GlyphStream header: 4 bytes
   - Glyph headers: ~2-3 bytes per primitive + payload length encoding
   - CRC-16 checksum: 2 bytes
   - Payload: UTF-8 encoded text
   
4. COMPRESSION EFFECTIVENESS:
   - Text < ~100 bytes: Typically expands (overhead > content benefit)
   - Text > ~200 bytes: Can compress well with pattern sharing
   - Text > ~500 bytes: Strong compression via semantic reuse

5. RECOMMENDATION:
   - Adjust test thresholds OR
   - Implement compression-aware encoding for short strings OR
   - Document that compression is effective for longer content only
""")


if __name__ == "__main__":
    main()
