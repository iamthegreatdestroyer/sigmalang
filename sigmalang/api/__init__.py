"""
ΣLANG Public API
================
"""

from .interfaces import (
    CompressionEngine,
    RSUManager,
    CodebookProtocol,
    StorageBackend,
    SigmaFactory,
)

from .types import (
    GlyphTier, EncodingMode, ProcessingMode, StorageTier, CompressionQuality,
    SemanticGlyph, GlyphModifier, EncodedGlyph, GlyphSequence,
    SigmaEncodedContext, DecodedContext,
    RSUMetadata, RSUEntry, RSUReference, RSUChain,
    EncodingResult, DecodingResult, CompressionStatistics,
    CodebookEntry, CodebookMetadata, CodebookState,
)

from .exceptions import (
    SigmaError, CodebookNotLoadedError, EncodingError, DecodingError,
    RSUNotFoundError, RSUStorageError, SemanticHashCollisionError, CompressionQualityError,
)

__all__ = [
    "CompressionEngine", "RSUManager", "CodebookProtocol", "StorageBackend", "SigmaFactory",
    "GlyphTier", "EncodingMode", "ProcessingMode", "StorageTier", "CompressionQuality",
    "SemanticGlyph", "GlyphModifier", "EncodedGlyph", "GlyphSequence",
    "SigmaEncodedContext", "DecodedContext",
    "RSUMetadata", "RSUEntry", "RSUReference", "RSUChain",
    "EncodingResult", "DecodingResult", "CompressionStatistics",
    "CodebookEntry", "CodebookMetadata", "CodebookState",
    "SigmaError", "CodebookNotLoadedError", "EncodingError", "DecodingError",
    "RSUNotFoundError", "RSUStorageError", "SemanticHashCollisionError", "CompressionQualityError",
]

__version__ = "0.1.0"
