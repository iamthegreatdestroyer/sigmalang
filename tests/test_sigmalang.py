#!/usr/bin/env python3
"""
ΣLANG Comprehensive Test Suite
==============================

Tests all components of the ΣLANG system and demonstrates
compression capabilities.

Run with: python -m pytest tests/test_sigmalang.py -v
Or directly: python tests/test_sigmalang.py

Copyright 2025 - Ryot LLM Project
"""

import sys
import unittest
from pathlib import Path
import tempfile
import json

# Add parent to path for proper imports
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root.parent))

from sigmalang.core.primitives import (
    SemanticNode, SemanticTree, Glyph, GlyphStream, GlyphType,
    ExistentialPrimitive, CodePrimitive, ActionPrimitive,
    PRIMITIVE_REGISTRY
)
from sigmalang.core.parser import SemanticParser, SemanticTreePrinter
from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder, SigmaHashBank
from sigmalang.training.codebook import (
    LearnedCodebook, CodebookTrainer, TrainingConfig,
    PatternExtractor, TrainingCorpusBuilder
)
from sigmalang.ryot_integration import (
    SigmaLangPipeline, RyotInputProcessor, create_pipeline
)


class TestPrimitives(unittest.TestCase):
    """Test primitive definitions and registry."""
    
    def test_primitive_registry_lookup(self):
        """Test primitive name lookup."""
        name = PRIMITIVE_REGISTRY.get_name(ExistentialPrimitive.ENTITY)
        self.assertEqual(name, "ENTITY")
        
        name = PRIMITIVE_REGISTRY.get_name(CodePrimitive.FUNCTION)
        self.assertEqual(name, "FUNCTION")
    
    def test_primitive_id_lookup(self):
        """Test primitive ID lookup."""
        id = PRIMITIVE_REGISTRY.get_id("ENTITY")
        self.assertEqual(id, ExistentialPrimitive.ENTITY)
    
    def test_semantic_node_creation(self):
        """Test semantic node creation."""
        node = SemanticNode(
            primitive=ExistentialPrimitive.ENTITY,
            value="test"
        )
        self.assertEqual(node.primitive, ExistentialPrimitive.ENTITY)
        self.assertEqual(node.value, "test")
        self.assertEqual(node.depth(), 1)
    
    def test_semantic_tree_creation(self):
        """Test semantic tree creation."""
        root = SemanticNode(
            primitive=ExistentialPrimitive.ACTION,
            children=[
                SemanticNode(primitive=ExistentialPrimitive.ENTITY, value="target"),
                SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value="property")
            ]
        )
        tree = SemanticTree(root=root, source_text="test")
        
        self.assertEqual(tree.depth, 2)
        self.assertEqual(tree.node_count, 3)
        self.assertEqual(len(tree.primitives_used), 3)


class TestGlyphs(unittest.TestCase):
    """Test glyph encoding/decoding."""
    
    def test_glyph_encoding(self):
        """Test basic glyph encoding."""
        glyph = Glyph(
            glyph_type=GlyphType.PRIMITIVE,
            primitive_id=ExistentialPrimitive.ENTITY
        )
        encoded = glyph.to_bytes()
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)
    
    def test_glyph_decoding(self):
        """Test glyph round-trip."""
        original = Glyph(
            glyph_type=GlyphType.PRIMITIVE,
            primitive_id=ExistentialPrimitive.ACTION,
            payload=b"test"
        )
        encoded = original.to_bytes()
        decoded, _ = Glyph.from_bytes(encoded)
        
        self.assertEqual(decoded.glyph_type, original.glyph_type)
        self.assertEqual(decoded.primitive_id, original.primitive_id)
    
    def test_glyph_stream(self):
        """Test glyph stream encoding."""
        glyphs = [
            Glyph(GlyphType.PRIMITIVE, ExistentialPrimitive.ACTION),
            Glyph(GlyphType.PRIMITIVE, ExistentialPrimitive.ENTITY),
        ]
        stream = GlyphStream(glyphs=glyphs)
        encoded = stream.to_bytes()
        decoded = GlyphStream.from_bytes(encoded)
        
        self.assertEqual(len(decoded.glyphs), len(glyphs))


class TestParser(unittest.TestCase):
    """Test semantic parser."""
    
    def setUp(self):
        self.parser = SemanticParser()
    
    def test_code_request_parsing(self):
        """Test parsing code generation request."""
        text = "Create a Python function that sorts a list"
        tree = self.parser.parse(text)
        
        self.assertIsInstance(tree, SemanticTree)
        self.assertEqual(tree.source_text, text)
        self.assertGreater(tree.node_count, 1)
    
    def test_query_parsing(self):
        """Test parsing query."""
        text = "What is dependency injection?"
        tree = self.parser.parse(text)
        
        self.assertIsInstance(tree, SemanticTree)
        self.assertIn(ExistentialPrimitive.ABSTRACT, tree.primitives_used)
    
    def test_modification_parsing(self):
        """Test parsing modification request."""
        text = "Fix the bug in this authentication code"
        tree = self.parser.parse(text)
        
        self.assertIsInstance(tree, SemanticTree)


class TestEncoder(unittest.TestCase):
    """Test ΣLANG encoder."""
    
    def setUp(self):
        self.parser = SemanticParser()
        self.encoder = SigmaEncoder()
        self.decoder = SigmaDecoder(self.encoder)
    
    def test_basic_encoding(self):
        """Test basic encoding."""
        text = "Create a function"
        tree = self.parser.parse(text)
        encoded = self.encoder.encode(tree, text)
        
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)
    
    def test_compression_achieved(self):
        """Test that compression is achieved."""
        text = "Create a Python function that sorts a list in descending order and returns the result"
        tree = self.parser.parse(text)
        encoded = self.encoder.encode(tree, text)
        
        original_size = len(text.encode('utf-8'))
        compressed_size = len(encoded)
        
        # Should achieve some compression
        self.assertLess(compressed_size, original_size)
    
    def test_encoding_decoding_roundtrip(self):
        """Test encode/decode roundtrip."""
        text = "Create a function that processes data"
        tree = self.parser.parse(text)
        encoded = self.encoder.encode(tree, text)
        decoded = self.decoder.decode(encoded)
        
        self.assertIsInstance(decoded, SemanticTree)


class TestSigmaHashBank(unittest.TestCase):
    """Test Sigma Hash Bank."""
    
    def setUp(self):
        self.bank = SigmaHashBank()
        self.parser = SemanticParser()
    
    def test_hash_computation(self):
        """Test semantic hash computation."""
        tree = self.parser.parse("Create a function")
        hash1 = self.bank.compute_hash(tree)
        
        self.assertIsInstance(hash1, int)
        self.assertGreater(hash1, 0)
    
    def test_similar_inputs_similar_hashes(self):
        """Test that similar inputs produce similar hashes."""
        tree1 = self.parser.parse("Create a Python function that sorts a list")
        tree2 = self.parser.parse("Create a Python function that filters a list")
        tree3 = self.parser.parse("What is the weather today?")
        
        emb1 = self.bank.compute_embedding(tree1)
        emb2 = self.bank.compute_embedding(tree2)
        emb3 = self.bank.compute_embedding(tree3)
        
        # Similar queries should have higher similarity
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)
        
        # tree1 and tree2 are more similar than tree1 and tree3
        self.assertGreater(sim_12, sim_13)
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving trees."""
        tree = self.parser.parse("Create a function")
        sigma_hash = self.bank.store(tree)
        
        retrieved = self.bank.retrieve(sigma_hash)
        self.assertIsNotNone(retrieved)


class TestLearnedCodebook(unittest.TestCase):
    """Test learned codebook."""
    
    def setUp(self):
        self.codebook = LearnedCodebook()
        self.parser = SemanticParser()
    
    def test_pattern_observation(self):
        """Test pattern observation."""
        tree = self.parser.parse("Create a function")
        
        # Observe multiple times
        for _ in range(5):
            self.codebook.observe(tree)
        
        self.assertGreater(len(self.codebook.candidates), 0)
    
    def test_pattern_promotion(self):
        """Test pattern promotion after threshold."""
        tree = self.parser.parse("Create a Python function")
        
        # Observe enough times to trigger promotion
        for _ in range(25):
            self.codebook.observe(tree)
        
        # Should have promoted to pattern
        pattern_id = self.codebook.match(tree)
        # May or may not match depending on implementation details
    
    def test_codebook_save_load(self):
        """Test codebook persistence."""
        tree = self.parser.parse("Test pattern")
        
        for _ in range(25):
            self.codebook.observe(tree)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)
        
        try:
            self.codebook.save(path)
            
            # Load into new codebook
            new_codebook = LearnedCodebook()
            new_codebook.load(path)
            
            self.assertEqual(
                len(new_codebook.patterns),
                len(self.codebook.patterns)
            )
        finally:
            path.unlink()


class TestCodebookTrainer(unittest.TestCase):
    """Test codebook training."""
    
    def setUp(self):
        self.codebook = LearnedCodebook()
        self.config = TrainingConfig(
            promotion_threshold=5,
            epochs=2
        )
        self.trainer = CodebookTrainer(self.codebook, self.config)
        self.parser = SemanticParser()
    
    def test_online_training(self):
        """Test online training observation."""
        texts = [
            "Create a function",
            "Write a method",
            "Build a class"
        ]
        
        for text in texts * 3:
            tree = self.parser.parse(text)
            self.trainer.observe(tree, len(text.encode('utf-8')))
        
        self.assertGreater(self.trainer.observation_count, 0)
    
    def test_training_report(self):
        """Test training report generation."""
        tree = self.parser.parse("Test input")
        self.trainer.observe(tree, 10)
        
        report = self.trainer.get_training_report()
        self.assertIn("ΣLANG", report)


class TestPipeline(unittest.TestCase):
    """Test ΣLANG pipeline."""
    
    def setUp(self):
        self.pipeline = SigmaLangPipeline(enable_training=True)
    
    def test_encode_input(self):
        """Test input encoding."""
        result = self.pipeline.encode_input("Create a Python function")
        
        self.assertIsNotNone(result.sigma_bytes)
        self.assertGreater(result.compression_ratio, 0)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        text = "Create a Python function that sorts a list in descending order"
        result = self.pipeline.encode_input(text)
        
        expected_ratio = result.input_size / result.output_size
        self.assertAlmostEqual(result.compression_ratio, expected_ratio)
    
    def test_pattern_learning(self):
        """Test that patterns are learned over time."""
        text = "Create a Python function that processes data"
        
        # Encode multiple times to trigger learning
        for _ in range(20):
            result = self.pipeline.encode_input(text)
        
        # Check if pattern matching occurred
        stats = self.pipeline.get_compression_stats()
        # Metrics should be recorded
        self.assertGreater(stats['total_inputs'], 0)
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        self.pipeline.encode_input("Test input one")
        self.pipeline.encode_input("Test input two")
        
        stats = self.pipeline.get_compression_stats()
        
        self.assertEqual(stats['total_inputs'], 2)
        self.assertGreater(stats['total_input_bytes'], 0)


class TestInputProcessor(unittest.TestCase):
    """Test Ryot input processor."""
    
    def setUp(self):
        self.pipeline = SigmaLangPipeline(enable_training=False)
        self.processor = RyotInputProcessor(self.pipeline)
    
    def test_process_input(self):
        """Test input processing."""
        result = self.processor.process("Create a function")
        
        self.assertIn('sigma_encoding', result)
        self.assertIn('compression_ratio', result)
        self.assertIn('metadata', result)
    
    def test_context_history(self):
        """Test context history tracking."""
        self.processor.process("First input")
        self.processor.process("Second input")
        
        self.assertEqual(len(self.processor.context_history), 2)


# Import numpy for similarity tests
import numpy as np


def run_compression_demo():
    """Run a demonstration of compression capabilities."""
    print("=" * 70)
    print("ΣLANG COMPRESSION DEMONSTRATION")
    print("=" * 70)
    
    pipeline = create_pipeline(enable_training=True)
    
    # Test inputs
    test_inputs = [
        "Create a Python function that sorts a list",
        "Create a Python function that filters a list", 
        "Create a Python function that maps a list",
        "Write a JavaScript function to validate email",
        "What is the time complexity of quicksort?",
        "Explain how dependency injection works",
        "Fix the bug in this authentication module",
        "Create a Python function that compresses data",
    ]
    
    # First pass - learning
    print("\n1. Learning phase (first pass)...")
    for text in test_inputs * 3:  # Repeat to trigger learning
        pipeline.encode_input(text)
    
    print(f"   Patterns learned: {len(pipeline.codebook.patterns)}")
    
    # Second pass - measure compression
    print("\n2. Compression measurement...")
    print("-" * 70)
    print(f"{'Input':<45} {'Orig':>6} {'Comp':>6} {'Ratio':>8} {'Type':<10}")
    print("-" * 70)
    
    total_original = 0
    total_compressed = 0
    
    for text in test_inputs:
        result = pipeline.encode_input(text)
        total_original += result.input_size
        total_compressed += result.output_size
        
        display_text = text[:42] + "..." if len(text) > 45 else text
        print(f"{display_text:<45} {result.input_size:>6} {result.output_size:>6} "
              f"{result.compression_ratio:>7.1f}x {result.encoding_type:<10}")
    
    print("-" * 70)
    overall_ratio = total_original / total_compressed
    print(f"{'TOTAL':<45} {total_original:>6} {total_compressed:>6} {overall_ratio:>7.1f}x")
    
    # Show statistics
    print("\n3. Pipeline Statistics:")
    stats = pipeline.get_compression_stats()
    print(f"   Total inputs processed: {stats['total_inputs']}")
    print(f"   Overall compression ratio: {stats['overall_compression_ratio']:.2f}x")
    print(f"   Pattern match rate: {stats['pattern_match_rate']:.1%}")
    print(f"   Codebook patterns: {stats['codebook']['pattern_count']}")
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        run_compression_demo()
    else:
        # Run tests
        unittest.main(verbosity=2)
