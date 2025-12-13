#!/usr/bin/env python3
"""
ΣLANG Semantic Search Example
=============================

Demonstrates semantic search capabilities.
"""

import sys
sys.path.insert(0, '..')


def main():
    """Run semantic search examples."""
    print("=" * 60)
    print("ΣLANG Semantic Search Example")
    print("=" * 60)
    
    # Sample corpus
    corpus = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
        "Reinforcement learning trains agents through reward signals",
        "Supervised learning uses labeled training data",
        "Unsupervised learning finds patterns in unlabeled data",
        "Transfer learning applies knowledge from one task to another",
        "Feature engineering creates meaningful input variables",
        "Model validation ensures generalization to new data",
        "Gradient descent optimizes neural network weights",
        "Backpropagation calculates gradients for training",
        "Convolutional networks excel at image recognition",
        "Recurrent networks process sequential data",
        "Transformers use attention mechanisms for sequences",
    ]
    
    print(f"\n1. Corpus: {len(corpus)} documents")
    print("-" * 50)
    for i, doc in enumerate(corpus[:5]):
        print(f"   {i+1}. {doc[:55]}...")
    print(f"   ... and {len(corpus)-5} more documents")
    
    # Initialize search engine
    print("\n2. Initializing search engine...")
    try:
        from sigmalang.core.semantic_search import SemanticSearchEngine
        search_engine = SemanticSearchEngine()
        search_engine.index_corpus(corpus)
        print("   ✓ Search engine ready")
        engine_available = True
    except ImportError as e:
        print(f"   ⚠ SemanticSearchEngine not available: {e}")
        print("   Using fallback keyword search...")
        engine_available = False
    
    # Search queries
    queries = [
        "AI systems that understand human language",
        "How to train machine learning models",
        "Image recognition with neural networks",
        "Learning without labeled data",
    ]
    
    print("\n3. Semantic search results:")
    print("-" * 50)
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        
        if engine_available:
            try:
                results = search_engine.search(query, top_k=3)
                for i, result in enumerate(results):
                    if isinstance(result, tuple):
                        text, score = result[0], result[1]
                    elif hasattr(result, 'text'):
                        text, score = result.text, result.score
                    else:
                        text, score = str(result), 0.5
                    print(f"      {i+1}. [{score:.3f}] {text[:50]}...")
            except Exception as e:
                print(f"      Error: {e}")
        else:
            # Fallback: simple keyword matching
            query_words = set(query.lower().split())
            scored = []
            for doc in corpus:
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                scored.append((doc, overlap / len(query_words)))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            for i, (doc, score) in enumerate(scored[:3]):
                print(f"      {i+1}. [{score:.3f}] {doc[:50]}...")
    
    # Search modes
    print("\n4. Available search modes:")
    print("-" * 50)
    
    modes = [
        ("exact", "Exact keyword matching"),
        ("semantic", "Meaning-based search (default)"),
        ("hybrid", "Combined keyword + semantic"),
        ("fuzzy", "Approximate string matching"),
    ]
    
    for mode, desc in modes:
        print(f"   {mode:10} | {desc}")
    
    # Advanced features
    print("\n5. Advanced search features:")
    print("-" * 50)
    print("   • Threshold filtering (minimum similarity score)")
    print("   • Highlighted matches in results")
    print("   • Query expansion for better recall")
    print("   • Cross-lingual search (multilingual models)")
    print("   • Batch search for multiple queries")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
