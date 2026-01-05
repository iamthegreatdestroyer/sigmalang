#!/usr/bin/env python3
"""
ΣLANG Training Pipeline
=======================

Complete training script for the ΣLANG learned codebook.

Usage:
    # Train from scratch with sample data
    python train.py --mode bootstrap
    
    # Train from JSONL corpus
    python train.py --mode batch --corpus data/training_corpus.jsonl
    
    # Continue training with online learning
    python train.py --mode online --codebook models/codebook.json

Copyright 2025 - Ryot LLM Project
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.primitives import SemanticTree, SemanticNode, ExistentialPrimitive
from core.parser import SemanticParser
from core.encoder import SigmaEncoder, SigmaDecoder
from training.codebook import (
    LearnedCodebook, CodebookTrainer, TrainingConfig, TrainingCorpusBuilder
)


# ============================================================================
# SAMPLE TRAINING DATA
# ============================================================================

BOOTSTRAP_EXAMPLES = [
    # Code requests
    "Create a Python function that sorts a list in descending order",
    "Write a JavaScript function to validate email addresses",
    "Build a class that handles database connections",
    "Implement a binary search algorithm in Python",
    "Create a function that filters out null values from an array",
    "Write a method to convert JSON to CSV format",
    "Build a REST API endpoint for user authentication",
    "Create a Python script that reads from a file and processes each line",
    "Write a function to merge two sorted arrays",
    "Implement a cache with LRU eviction policy",
    
    # Queries
    "What is the difference between a list and a tuple?",
    "How do I handle exceptions in Python?",
    "What are the best practices for API design?",
    "Explain how async/await works in JavaScript",
    "What is dependency injection?",
    "How does garbage collection work?",
    "What is the time complexity of quicksort?",
    "Explain the singleton pattern",
    "What is the difference between SQL and NoSQL?",
    "How do I optimize database queries?",
    
    # Modifications
    "Fix the bug in this function that causes infinite recursion",
    "Refactor this code to use async/await",
    "Optimize this algorithm for better performance",
    "Add error handling to this function",
    "Update this code to use the new API",
    "Convert this class to use dependency injection",
    "Modify this function to support pagination",
    "Change this synchronous code to asynchronous",
    "Fix the memory leak in this component",
    "Update this code to handle edge cases",
    
    # Explanations
    "Explain how this algorithm works step by step",
    "Describe the architecture of this system",
    "Walk me through this code",
    "Explain the trade-offs of this approach",
    "Describe what this function does",
    "Explain why this pattern is useful",
    "Walk me through the authentication flow",
    "Describe the data flow in this system",
    "Explain the purpose of this design pattern",
    "Describe how caching improves performance",
    
    # Project-specific patterns (these should become learned)
    "In the context of RYZEN-LLM, update the encoder module",
    "For the NEXUS project, implement a new agent",
    "Add a new primitive to the ΣLANG system",
    "Create a benchmark for the Token Recycler",
    "Update the Sigma Bank storage layer",
    "Implement a new compression strategy for the codebook",
    "Add AVX-512 optimization to the kernel",
    "Create a test suite for the semantic parser",
    "Build a visualization for compression metrics",
    "Implement delta encoding for context references",
]


def generate_variations(base_examples: List[str], multiplier: int = 5) -> List[str]:
    """Generate variations of base examples for richer training."""
    variations = list(base_examples)
    
    # Language variations
    languages = ['Python', 'JavaScript', 'TypeScript', 'Rust', 'Go', 'Java']
    
    # Action variations
    actions = [
        ('Create', 'Write'), ('Create', 'Build'), ('Create', 'Implement'),
        ('Write', 'Build'), ('Write', 'Create'), ('Write', 'Make'),
        ('Fix', 'Debug'), ('Fix', 'Repair'), ('Fix', 'Correct'),
        ('Explain', 'Describe'), ('Explain', 'Tell me about'),
    ]
    
    for _ in range(multiplier - 1):
        for example in base_examples:
            # Language substitution
            for lang in languages:
                if 'Python' in example:
                    variations.append(example.replace('Python', lang))
                elif 'JavaScript' in example:
                    variations.append(example.replace('JavaScript', lang))
            
            # Action substitution
            for old, new in actions:
                if example.startswith(old):
                    variations.append(example.replace(old, new, 1))
    
    return list(set(variations))  # Remove duplicates


# ============================================================================
# TRAINING MODES
# ============================================================================

def bootstrap_training(output_dir: Path, epochs: int = 5):
    """
    Bootstrap training with built-in example data.
    Use this to initialize a new codebook.
    """
    print("=" * 60)
    print("ΣLANG BOOTSTRAP TRAINING")
    print("=" * 60)
    
    # Generate training data
    examples = generate_variations(BOOTSTRAP_EXAMPLES, multiplier=10)
    print(f"Generated {len(examples)} training examples")
    
    # Initialize components
    parser = SemanticParser()
    codebook = LearnedCodebook()
    
    config = TrainingConfig(
        epochs=epochs,
        promotion_threshold=15,  # Lower for bootstrap
        save_interval=200,
        checkpoint_dir=output_dir / "checkpoints"
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = CodebookTrainer(codebook, config)
    
    # Build corpus
    print("\nBuilding training corpus...")
    builder = TrainingCorpusBuilder(parser)
    builder.add_texts(examples)
    
    # Split data
    (train_corpus, train_sizes), (test_corpus, test_sizes) = builder.split(0.9)
    print(f"Training set: {len(train_corpus)} examples")
    print(f"Test set: {len(test_corpus)} examples")
    
    # Train
    print("\nStarting training...")
    trainer.batch_train(train_corpus, train_sizes)
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(test_corpus, test_sizes)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save final codebook
    codebook_path = output_dir / "codebook.json"
    codebook.save(codebook_path)
    print(f"\nCodebook saved to: {codebook_path}")
    
    # Print report
    print("\n" + trainer.get_training_report())
    
    return codebook


def batch_training(corpus_path: Path, output_dir: Path, 
                   existing_codebook: Path = None, epochs: int = 10):
    """
    Batch training from a JSONL corpus file.
    """
    print("=" * 60)
    print("ΣLANG BATCH TRAINING")
    print("=" * 60)
    print(f"Corpus: {corpus_path}")
    
    # Initialize components
    parser = SemanticParser()
    
    if existing_codebook and existing_codebook.exists():
        print(f"Loading existing codebook: {existing_codebook}")
        codebook = LearnedCodebook(existing_codebook)
    else:
        codebook = LearnedCodebook()
    
    config = TrainingConfig(
        epochs=epochs,
        promotion_threshold=20,
        save_interval=500,
        checkpoint_dir=output_dir / "checkpoints"
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = CodebookTrainer(codebook, config)
    
    # Build corpus
    print("\nLoading corpus...")
    builder = TrainingCorpusBuilder(parser)
    builder.load_from_jsonl(corpus_path)
    
    corpus, sizes = builder.get_corpus()
    print(f"Loaded {len(corpus)} examples")
    
    # Split data
    (train_corpus, train_sizes), (test_corpus, test_sizes) = builder.split(0.9)
    
    # Train
    print("\nStarting training...")
    trainer.batch_train(train_corpus, train_sizes)
    
    # Evaluate
    metrics = trainer.evaluate(test_corpus, test_sizes)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save
    codebook_path = output_dir / "codebook.json"
    codebook.save(codebook_path)
    print(f"\nCodebook saved to: {codebook_path}")
    
    print("\n" + trainer.get_training_report())
    
    return codebook


def online_training(codebook_path: Path):
    """
    Interactive online training mode.
    Processes input in real-time and learns patterns.
    """
    print("=" * 60)
    print("ΣLANG ONLINE TRAINING MODE")
    print("=" * 60)
    print("Enter text to train on. Type 'quit' to exit, 'stats' for statistics.")
    print()
    
    # Initialize
    parser = SemanticParser()
    codebook = LearnedCodebook(codebook_path if codebook_path.exists() else None)
    encoder = SigmaEncoder(codebook)
    
    config = TrainingConfig(
        promotion_threshold=10,
        save_interval=50
    )
    trainer = CodebookTrainer(codebook, config)
    
    while True:
        try:
            text = # SECURITY: input() should be validated
validated_input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not text:
            continue
        
        if text.lower() == 'quit':
            break
        
        if text.lower() == 'stats':
            print(trainer.get_training_report())
            continue
        
        if text.lower() == 'save':
            codebook.save(codebook_path)
            print(f"Codebook saved to {codebook_path}")
            continue
        
        # Parse and encode
        tree = parser.parse(text)
        original_size = len(text.encode('utf-8'))
        
        # Observe for training
        trainer.observe(tree, original_size)
        
        # Encode to show compression
        encoded = encoder.encode(tree, text)
        compressed_size = len(encoded)
        ratio = original_size / compressed_size
        
        # Show results
        print(f"  Original: {original_size} bytes")
        print(f"  Compressed: {compressed_size} bytes")
        print(f"  Ratio: {ratio:.2f}x")
        
        pattern_id = codebook.match(tree)
        if pattern_id is not None:
            print(f"  Matched pattern: Σ_{pattern_id}")
    
    # Save on exit
    codebook.save(codebook_path)
    print(f"\nCodebook saved to {codebook_path}")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_compression():
    """
    Demonstrate the ΣLANG compression system.
    """
    print("=" * 60)
    print("ΣLANG COMPRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize
    parser = SemanticParser()
    codebook = LearnedCodebook()
    encoder = SigmaEncoder(codebook)
    decoder = SigmaDecoder(encoder)
    
    # Train on some examples first
    print("\n1. Training codebook on example patterns...")
    config = TrainingConfig(promotion_threshold=3, epochs=2)
    trainer = CodebookTrainer(codebook, config)
    
    # Repeat patterns to trigger learning
    training_data = [
        "Create a Python function that sorts a list",
        "Create a Python function that filters a list",
        "Create a Python function that maps a list",
        "Create a Python function that reduces a list",
        "Create a Python function that validates a list",
    ] * 10
    
    for text in training_data:
        tree = parser.parse(text)
        trainer.observe(tree, len(text.encode('utf-8')))
    
    print(f"   Learned {len(codebook.patterns)} patterns")
    
    # Now demonstrate compression
    print("\n2. Testing compression on new inputs...")
    
    test_inputs = [
        "Create a Python function that sorts a list",  # Should match pattern
        "Write a JavaScript function to validate email",  # Similar structure
        "What is the time complexity of quicksort?",  # Query pattern
        "Fix the bug in the authentication module",  # Modification pattern
        "Create a Python function that compresses data",  # Should match pattern!
    ]
    
    print("\n   Results:")
    print("   " + "-" * 56)
    
    total_original = 0
    total_compressed = 0
    
    for text in test_inputs:
        tree = parser.parse(text)
        original_size = len(text.encode('utf-8'))
        
        encoded = encoder.encode(tree, text)
        compressed_size = len(encoded)
        
        total_original += original_size
        total_compressed += compressed_size
        
        ratio = original_size / compressed_size
        pattern_id = codebook.match(tree)
        
        pattern_str = f"Σ_{pattern_id}" if pattern_id else "none"
        print(f"   {text[:40]:<40} {original_size:>3}B → {compressed_size:>2}B ({ratio:>5.1f}x) [{pattern_str}]")
    
    print("   " + "-" * 56)
    overall_ratio = total_original / total_compressed
    print(f"   {'TOTAL':<40} {total_original:>3}B → {total_compressed:>2}B ({overall_ratio:>5.1f}x)")
    
    # Show encoder stats
    print("\n3. Encoder Statistics:")
    stats = encoder.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ΣLANG Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  bootstrap   Initialize new codebook with built-in examples
  batch       Train from JSONL corpus file
  online      Interactive training mode
  demo        Demonstrate compression system

Examples:
  python train.py --mode bootstrap --output models/
  python train.py --mode batch --corpus data/training.jsonl
  python train.py --mode online --codebook models/codebook.json
  python train.py --mode demo
        """
    )
    
    parser.add_argument('--mode', choices=['bootstrap', 'batch', 'online', 'demo'],
                        default='demo', help='Training mode')
    parser.add_argument('--corpus', type=Path, help='Path to JSONL corpus file')
    parser.add_argument('--codebook', type=Path, help='Path to existing codebook')
    parser.add_argument('--output', type=Path, default=Path('models'),
                        help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'bootstrap':
        bootstrap_training(args.output, args.epochs)
    
    elif args.mode == 'batch':
        if not args.corpus:
            parser.error("--corpus required for batch mode")
        batch_training(args.corpus, args.output, args.codebook, args.epochs)
    
    elif args.mode == 'online':
        codebook_path = args.codebook or args.output / 'codebook.json'
        online_training(codebook_path)
    
    elif args.mode == 'demo':
        demonstrate_compression()


if __name__ == '__main__':
    main()
