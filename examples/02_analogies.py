#!/usr/bin/env python3
"""
ΣLANG Analogies Example
=======================

Demonstrates solving word analogies with the ΣLANG analogy engine.
"""

import sys
sys.path.insert(0, '..')


def main():
    """Run analogy examples."""
    print("=" * 60)
    print("ΣLANG Analogies Example")
    print("=" * 60)
    
    # Import analogy engine
    print("\n1. Initializing analogy engine...")
    try:
        from sigmalang.core.semantic_analogy_engine import SemanticAnalogyEngine
        engine = SemanticAnalogyEngine()
        
        # Register candidate words
        candidates = [
            "woman", "man", "king", "queen", "prince", "princess",
            "boy", "girl", "father", "mother", "son", "daughter",
            "husband", "wife", "brother", "sister",
            "dog", "cat", "puppy", "kitten",
            "big", "small", "hot", "cold", "fast", "slow",
            "happy", "sad", "good", "bad", "rich", "poor",
            "Tokyo", "Japan", "Paris", "France", "Berlin", "Germany",
            "London", "England", "Rome", "Italy", "Madrid", "Spain"
        ]
        engine.register_candidates(candidates)
        print("   ✓ Analogy engine ready")
        engine_available = True
    except ImportError as e:
        print(f"   ⚠ SemanticAnalogyEngine not available: {e}")
        print("   Using fallback demonstration...")
        engine_available = False
    
    # Define test analogies
    test_analogies = [
        ("king", "queen", "man"),           # Gender relationship
        ("Paris", "France", "Tokyo"),       # Capital-country relationship
        ("big", "small", "hot"),            # Antonym relationship
        ("dog", "puppy", "cat"),            # Adult-young relationship
    ]
    
    print("\n2. Solving analogies (A:B::C:?):")
    print("-" * 50)
    
    for a, b, c in test_analogies:
        print(f"\n   {a} : {b} :: {c} : ?")
        
        if engine_available:
            try:
                result = engine.solve_analogy(a, b, c, top_k=3)
                
                if hasattr(result, 'solutions'):
                    for i, sol in enumerate(result.solutions[:3]):
                        answer = sol.answer if hasattr(sol, 'answer') else str(sol)
                        conf = sol.confidence if hasattr(sol, 'confidence') else 0.5
                        print(f"      {i+1}. {answer} (confidence: {conf:.1%})")
                elif isinstance(result, list):
                    for i, item in enumerate(result[:3]):
                        if isinstance(item, tuple):
                            print(f"      {i+1}. {item[0]} (confidence: {item[1]:.1%})")
                        else:
                            print(f"      {i+1}. {item}")
            except Exception as e:
                print(f"      Error: {e}")
        else:
            # Fallback: show expected answers
            expected = {
                ("king", "queen", "man"): "woman",
                ("Paris", "France", "Tokyo"): "Japan",
                ("big", "small", "hot"): "cold",
                ("dog", "puppy", "cat"): "kitten"
            }
            answer = expected.get((a, b, c), "?")
            print(f"      Expected: {answer}")
    
    # Explain analogies
    print("\n3. Explaining analogies:")
    print("-" * 50)
    
    if engine_available and hasattr(engine, 'explain_analogy'):
        try:
            explanation = engine.explain_analogy("Paris", "France", "Tokyo", "Japan")
            if hasattr(explanation, 'explanation'):
                print(f"\n   Paris:France::Tokyo:Japan")
                print(f"   Explanation: {explanation.explanation}")
                print(f"   Similarity: {explanation.similarity_score:.1%}")
        except Exception as e:
            print(f"   Explanation not available: {e}")
    else:
        print("\n   Paris:France::Tokyo:Japan")
        print("   Explanation: Capital city to country relationship")
        print("   Paris is the capital of France, Tokyo is the capital of Japan")
    
    # Analogy types
    print("\n4. Analogy types supported:")
    print("-" * 50)
    
    analogy_types = [
        ("Semantic", "king:queen::man:woman", "Meaning-based relationships"),
        ("Structural", "A:AB::B:BC", "Pattern-based relationships"),
        ("Proportional", "1:2::3:6", "Numeric proportions"),
        ("Causal", "rain:wet::fire:burn", "Cause-effect relationships"),
        ("Functional", "pen:write::knife:cut", "Purpose-based relationships"),
    ]
    
    for atype, example, desc in analogy_types:
        print(f"   {atype:15} | {example:20} | {desc}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
