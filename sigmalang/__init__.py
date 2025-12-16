"""
ΣLANG - Sub-Linear Algorithmic Neural Glyph Language
=====================================================

A novel compression language designed for internal LLM representation,
achieving 10-50x compression through semantic primitives, logarithmic
addressing, and learned pattern matching.

Core Components:
- primitives: Σ-primitive definitions and semantic tree structures
- parser: Natural language to semantic tree conversion
- encoder: Semantic tree to ΣLANG binary compression
- decoder: ΣLANG binary to semantic tree reconstruction

Training Components:
- codebook: Learned pattern storage and matching
- trainer: Online and batch training procedures

Copyright 2025 - Ryot LLM Project
"""

__version__ = "1.0.0"
__author__ = "Ryot LLM Project"

from .core.primitives import (
    SemanticNode,
    SemanticTree,
    Glyph,
    GlyphStream,
    GlyphType,
    ExistentialPrimitive,
    CodePrimitive,
    MathPrimitive,
    LogicPrimitive,
    EntityPrimitive,
    ActionPrimitive,
    CommunicationPrimitive,
    StructurePrimitive,
    LearnedPrimitive,
    PRIMITIVE_REGISTRY
)

from .core.parser import (
    SemanticParser,
    SemanticTreePrinter,
    IntentType
)

from .core.encoder import (
    SigmaEncoder,
    SigmaDecoder,
    SigmaHashBank,
    ContextStack
)

from .training.codebook import (
    LearnedCodebook,
    CodebookTrainer,
    TrainingConfig,
    TrainingCorpusBuilder,
    PatternSignature
)

# Phase 2A: Adapter imports for Ryot LLM integration
try:
    from .adapters import (
        SigmaCompressionAdapter,
        create_ryot_compression_adapter,
        RyotTokenSequence,
        RyotSigmaEncodedContext,
    )
except ImportError:
    # Adapters may not be available in all contexts
    pass

__all__ = [
    # Core structures
    'SemanticNode',
    'SemanticTree',
    'Glyph',
    'GlyphStream',
    'GlyphType',
    
    # Primitives
    'ExistentialPrimitive',
    'CodePrimitive',
    'MathPrimitive',
    'LogicPrimitive',
    'EntityPrimitive',
    'ActionPrimitive',
    'CommunicationPrimitive',
    'StructurePrimitive',
    'LearnedPrimitive',
    'PRIMITIVE_REGISTRY',
    
    # Parser
    'SemanticParser',
    'SemanticTreePrinter',
    'IntentType',
    
    # Encoder/Decoder
    'SigmaEncoder',
    'SigmaDecoder',
    'SigmaHashBank',
    'ContextStack',
    
    # Training
    'LearnedCodebook',
    'CodebookTrainer',
    'TrainingConfig',
    'TrainingCorpusBuilder',
    'PatternSignature',
    
    # Adapters (Phase 2A: Ryot Integration)
    'SigmaCompressionAdapter',
    'create_ryot_compression_adapter',
    'RyotTokenSequence',
    'RyotSigmaEncodedContext',
]
