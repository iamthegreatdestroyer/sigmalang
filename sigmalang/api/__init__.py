"""
ΣLANG Public API
================
"""

from .exceptions import (
    CodebookNotLoadedError,
    CompressionQualityError,
    DecodingError,
    EncodingError,
    RSUNotFoundError,
    RSUStorageError,
    SemanticHashCollisionError,
    SigmaError,
)
from .interfaces import (
    CodebookProtocol,
    CompressionEngine,
    RSUManager,
    SigmaFactory,
    StorageBackend,
)
from .types import (
    CodebookEntry,
    CodebookMetadata,
    CodebookState,
    CompressionQuality,
    CompressionStatistics,
    DecodedContext,
    DecodingResult,
    EncodedGlyph,
    EncodingMode,
    EncodingResult,
    GlyphModifier,
    GlyphSequence,
    GlyphTier,
    ProcessingMode,
    RSUChain,
    RSUEntry,
    RSUMetadata,
    RSUReference,
    SemanticGlyph,
    SigmaEncodedContext,
    StorageTier,
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

__version__ = "2.0.0"
