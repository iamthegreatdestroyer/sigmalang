"""
ΣLANG Integration for Ryot LLM
==============================

Complete integration layer that provides the ΣLANG compression pipeline
for the Ryot LLM inference engine.

This module provides:
1. SigmaLangPipeline: End-to-end encoding/decoding with automatic training
2. RyotInputProcessor: Pre-processes human input for the LLM
3. RyotOutputProcessor: Post-processes LLM output for human delivery
4. CompressionMetrics: Real-time compression statistics

Usage:
    from sigmalang.ryot_integration import SigmaLangPipeline
    
    pipeline = SigmaLangPipeline(codebook_path="models/codebook.json")
    
    # Encode human input for LLM processing
    sigma_bytes, metadata = pipeline.encode_input("Create a Python function...")
    
    # Decode back to human-readable (if needed)
    reconstructed = pipeline.decode_output(sigma_bytes)

Copyright 2025 - Ryot LLM Project
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
import numpy as np

from .core.primitives import SemanticTree, SemanticNode, ExistentialPrimitive
from .core.parser import SemanticParser, SemanticTreePrinter
from .core.encoder import SigmaEncoder, SigmaDecoder
from .training.codebook import LearnedCodebook, CodebookTrainer, TrainingConfig


# ============================================================================
# COMPRESSION METRICS
# ============================================================================

@dataclass
class CompressionMetrics:
    """Real-time compression statistics."""
    total_inputs: int = 0
    total_input_bytes: int = 0
    total_output_bytes: int = 0
    pattern_matches: int = 0
    reference_hits: int = 0
    delta_encodings: int = 0
    full_encodings: int = 0
    
    # Rolling window for recent performance
    recent_ratios: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def overall_compression_ratio(self) -> float:
        if self.total_output_bytes == 0:
            return 1.0
        return self.total_input_bytes / self.total_output_bytes
    
    @property
    def average_recent_ratio(self) -> float:
        if not self.recent_ratios:
            return 1.0
        return np.mean(self.recent_ratios)
    
    @property
    def pattern_match_rate(self) -> float:
        if self.total_inputs == 0:
            return 0.0
        return self.pattern_matches / self.total_inputs
    
    def record(self, input_size: int, output_size: int, encoding_type: str):
        """Record a compression event."""
        self.total_inputs += 1
        self.total_input_bytes += input_size
        self.total_output_bytes += output_size
        
        ratio = input_size / output_size if output_size > 0 else 1.0
        self.recent_ratios.append(ratio)
        
        if encoding_type == 'pattern':
            self.pattern_matches += 1
        elif encoding_type == 'reference':
            self.reference_hits += 1
        elif encoding_type == 'delta':
            self.delta_encodings += 1
        else:
            self.full_encodings += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_inputs': self.total_inputs,
            'total_input_bytes': self.total_input_bytes,
            'total_output_bytes': self.total_output_bytes,
            'overall_compression_ratio': self.overall_compression_ratio,
            'average_recent_ratio': self.average_recent_ratio,
            'pattern_match_rate': self.pattern_match_rate,
            'encoding_breakdown': {
                'pattern_matches': self.pattern_matches,
                'reference_hits': self.reference_hits,
                'delta_encodings': self.delta_encodings,
                'full_encodings': self.full_encodings
            }
        }


# ============================================================================
# ENCODING RESULT
# ============================================================================

@dataclass
class EncodingResult:
    """Result of ΣLANG encoding."""
    sigma_bytes: bytes
    semantic_tree: SemanticTree
    encoding_type: str  # 'pattern', 'reference', 'delta', 'full'
    input_size: int
    output_size: int
    compression_ratio: float
    pattern_id: Optional[int] = None
    sigma_hash: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_compressed(self) -> bool:
        return self.compression_ratio > 1.0


# ============================================================================
# SIGMA LANG PIPELINE
# ============================================================================

class SigmaLangPipeline:
    """
    End-to-end ΣLANG compression pipeline for Ryot LLM.
    
    Provides seamless encoding of human input to compressed ΣLANG format
    and decoding back to semantic structures or human-readable text.
    
    Features:
    - Automatic pattern learning during usage
    - Context-aware delta compression
    - Real-time compression metrics
    - Persistent codebook storage
    """
    
    def __init__(self, 
                 codebook_path: Optional[Path] = None,
                 enable_training: bool = True,
                 training_config: Optional[TrainingConfig] = None):
        """
        Initialize the ΣLANG pipeline.
        
        Args:
            codebook_path: Path to load/save learned codebook
            enable_training: Whether to learn patterns during usage
            training_config: Configuration for pattern learning
        """
        self.codebook_path = Path(codebook_path) if codebook_path else None
        self.enable_training = enable_training
        
        # Initialize components
        self.parser = SemanticParser()
        self.codebook = LearnedCodebook(self.codebook_path)
        self.encoder = SigmaEncoder(self.codebook)
        self.decoder = SigmaDecoder(self.encoder)
        
        # Training
        if enable_training:
            config = training_config or TrainingConfig(
                promotion_threshold=15,
                save_interval=100
            )
            self.trainer = CodebookTrainer(self.codebook, config)
        else:
            self.trainer = None
        
        # Metrics
        self.metrics = CompressionMetrics()
        
        # Operation counter for auto-save
        self._operation_count = 0
        self._save_interval = 100
    
    def encode_input(self, text: str) -> EncodingResult:
        """
        Encode human text input to ΣLANG format.
        
        Args:
            text: Human natural language input
            
        Returns:
            EncodingResult with compressed bytes and metadata
        """
        input_size = len(text.encode('utf-8'))
        
        # Parse to semantic tree
        tree = self.parser.parse(text)
        
        # Check for pattern match
        pattern_id = self.codebook.match(tree)
        
        # Encode
        sigma_bytes = self.encoder.encode(tree, text)
        output_size = len(sigma_bytes)
        
        # Determine encoding type
        if pattern_id is not None:
            encoding_type = 'pattern'
        elif output_size <= 6:  # Reference encoding is ~5-6 bytes
            encoding_type = 'reference'
        elif output_size < input_size * 0.3:  # Significant compression
            encoding_type = 'delta'
        else:
            encoding_type = 'full'
        
        # Calculate compression
        compression_ratio = input_size / output_size if output_size > 0 else 1.0
        
        # Record metrics
        self.metrics.record(input_size, output_size, encoding_type)
        
        # Train if enabled
        if self.trainer:
            self.trainer.observe(tree, input_size)
        
        # Auto-save periodically
        self._operation_count += 1
        if self._operation_count % self._save_interval == 0:
            self._auto_save()
        
        return EncodingResult(
            sigma_bytes=sigma_bytes,
            semantic_tree=tree,
            encoding_type=encoding_type,
            input_size=input_size,
            output_size=output_size,
            compression_ratio=compression_ratio,
            pattern_id=pattern_id,
            sigma_hash=self.encoder.sigma_bank.compute_hash(tree),
            metadata={
                'primitives_used': list(tree.primitives_used),
                'tree_depth': tree.depth,
                'node_count': tree.node_count
            }
        )
    
    def decode_output(self, sigma_bytes: bytes) -> SemanticTree:
        """
        Decode ΣLANG bytes back to semantic tree.
        
        Args:
            sigma_bytes: ΣLANG encoded bytes
            
        Returns:
            Reconstructed SemanticTree
        """
        return self.decoder.decode(sigma_bytes)
    
    def encode_batch(self, texts: List[str]) -> List[EncodingResult]:
        """Encode multiple texts."""
        return [self.encode_input(text) for text in texts]
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get current compression statistics."""
        return {
            **self.metrics.to_dict(),
            'codebook': self.codebook.get_stats(),
            'encoder': self.encoder.get_stats()
        }
    
    def _auto_save(self):
        """Auto-save codebook periodically."""
        if self.codebook_path:
            self.codebook.save(self.codebook_path)
    
    def save(self, path: Optional[Path] = None):
        """Manually save codebook."""
        save_path = path or self.codebook_path
        if save_path:
            self.codebook.save(save_path)
    
    def load(self, path: Path):
        """Load codebook from path."""
        self.codebook.load(path)


# ============================================================================
# RYOT INPUT PROCESSOR
# ============================================================================

class RyotInputProcessor:
    """
    Processes human input for Ryot LLM consumption.
    
    Responsibilities:
    1. Convert human text to ΣLANG encoding
    2. Attach context from RSU Vector Bank
    3. Prepare optimized context window
    4. Track input patterns for learning
    """
    
    def __init__(self, pipeline: SigmaLangPipeline, 
                 max_context_tokens: int = 4096):
        self.pipeline = pipeline
        self.max_context_tokens = max_context_tokens
        self.context_history: List[EncodingResult] = []
    
    def process(self, human_input: str) -> Dict[str, Any]:
        """
        Process human input for LLM.
        
        Args:
            human_input: Raw human text input
            
        Returns:
            Dict containing:
            - sigma_encoding: Compressed ΣLANG bytes
            - context_references: Relevant prior context
            - metadata: Processing metadata
        """
        # Encode input
        result = self.pipeline.encode_input(human_input)
        
        # Get relevant context from history
        context_refs = self._get_relevant_context(result.semantic_tree)
        
        # Add to history
        self.context_history.append(result)
        if len(self.context_history) > 50:  # Keep last 50
            self.context_history.pop(0)
        
        return {
            'sigma_encoding': result.sigma_bytes,
            'semantic_tree': result.semantic_tree,
            'context_references': context_refs,
            'compression_ratio': result.compression_ratio,
            'encoding_type': result.encoding_type,
            'metadata': {
                'original_size': result.input_size,
                'compressed_size': result.output_size,
                'pattern_id': result.pattern_id,
                **result.metadata
            }
        }
    
    def _get_relevant_context(self, tree: SemanticTree, k: int = 5) -> List[int]:
        """Find relevant prior context via semantic similarity."""
        # Use sigma bank's similarity search
        return self.pipeline.encoder.sigma_bank.search_similar(tree, k)


# ============================================================================
# RYOT OUTPUT PROCESSOR
# ============================================================================

class RyotOutputProcessor:
    """
    Processes Ryot LLM output for human delivery.
    
    Note: In the full Ryot LLM architecture, the model operates
    internally on ΣLANG representations. This processor converts
    output back to human-readable format when needed.
    """
    
    def __init__(self, pipeline: SigmaLangPipeline):
        self.pipeline = pipeline
    
    def process(self, sigma_output: bytes) -> str:
        """
        Convert ΣLANG output to human-readable text.
        
        Args:
            sigma_output: ΣLANG encoded output from LLM
            
        Returns:
            Human-readable text
        """
        tree = self.pipeline.decode_output(sigma_output)
        return self._tree_to_text(tree)
    
    def _tree_to_text(self, tree: SemanticTree) -> str:
        """Convert semantic tree to readable text."""
        # For now, use the stored source text or template
        if tree.source_text and tree.source_text != "[decoded]":
            return tree.source_text
        
        # Generate from tree structure
        return self._generate_text(tree.root)
    
    def _generate_text(self, node: SemanticNode) -> str:
        """Generate text from semantic node."""
        parts = []
        
        # Add node value if present
        if node.value:
            parts.append(str(node.value))
        
        # Process children
        for child in node.children:
            child_text = self._generate_text(child)
            if child_text:
                parts.append(child_text)
        
        return ' '.join(parts)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_pipeline(codebook_path: str = "models/sigma_codebook.json",
                   enable_training: bool = True) -> SigmaLangPipeline:
    """
    Create a configured ΣLANG pipeline.
    
    Args:
        codebook_path: Path to codebook file
        enable_training: Enable pattern learning
        
    Returns:
        Configured SigmaLangPipeline
    """
    path = Path(codebook_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SigmaLangPipeline(path, enable_training)


def quick_encode(text: str, pipeline: Optional[SigmaLangPipeline] = None) -> bytes:
    """
    Quick encode text to ΣLANG.
    
    Args:
        text: Text to encode
        pipeline: Optional existing pipeline (creates temporary if None)
        
    Returns:
        ΣLANG encoded bytes
    """
    if pipeline is None:
        pipeline = SigmaLangPipeline(enable_training=False)
    return pipeline.encode_input(text).sigma_bytes


def quick_decode(sigma_bytes: bytes, 
                 pipeline: Optional[SigmaLangPipeline] = None) -> SemanticTree:
    """
    Quick decode ΣLANG to semantic tree.
    
    Args:
        sigma_bytes: ΣLANG bytes
        pipeline: Optional existing pipeline
        
    Returns:
        Decoded SemanticTree
    """
    if pipeline is None:
        pipeline = SigmaLangPipeline(enable_training=False)
    return pipeline.decode_output(sigma_bytes)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for ΣLANG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ΣLANG Pipeline CLI")
    parser.add_argument('--codebook', default='models/codebook.json',
                       help='Path to codebook')
    parser.add_argument('--encode', type=str, help='Text to encode')
    parser.add_argument('--stats', action='store_true', 
                       help='Show compression statistics')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    pipeline = create_pipeline(args.codebook)
    
    if args.encode:
        result = pipeline.encode_input(args.encode)
        print(f"Input: {args.encode}")
        print(f"Compressed: {result.output_size} bytes (was {result.input_size})")
        print(f"Ratio: {result.compression_ratio:.2f}x")
        print(f"Type: {result.encoding_type}")
        print(f"Hex: {result.sigma_bytes.hex()}")
    
    elif args.stats:
        stats = pipeline.get_compression_stats()
        print(json.dumps(stats, indent=2, default=str))
    
    elif args.interactive:
        print("ΣLANG Interactive Mode (type 'quit' to exit)")
        processor = RyotInputProcessor(pipeline)
        
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            result = processor.process(text)
            print(f"  Compressed: {result['metadata']['compressed_size']} bytes")
            print(f"  Ratio: {result['compression_ratio']:.2f}x")
            print(f"  Type: {result['encoding_type']}")
        
        pipeline.save()
        print("\nCodebook saved.")


if __name__ == '__main__':
    main()
