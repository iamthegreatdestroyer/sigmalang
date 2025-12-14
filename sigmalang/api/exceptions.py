"""
ΣLANG Custom Exceptions
=======================
"""

from typing import Optional


class SigmaError(Exception):
    """Base exception for all ΣLANG errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SIGMA_ERROR",
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.is_retryable = is_retryable


class CodebookNotLoadedError(SigmaError):
    """Raised when encoding attempted without loaded codebook."""
    
    def __init__(self, codebook_path: Optional[str] = None):
        message = "No codebook loaded"
        if codebook_path:
            message = f"Failed to load codebook from: {codebook_path}"
        super().__init__(message, "CODEBOOK_NOT_LOADED", is_retryable=True)


class EncodingError(SigmaError):
    """Raised when encoding fails."""
    
    def __init__(self, message: str, token_position: Optional[int] = None):
        super().__init__(message, "ENCODING_ERROR", is_retryable=True)
        self.token_position = token_position


class DecodingError(SigmaError):
    """Raised when decoding fails."""
    
    def __init__(self, message: str, glyph_position: Optional[int] = None):
        super().__init__(message, "DECODING_ERROR", is_retryable=False)
        self.glyph_position = glyph_position


class RSUNotFoundError(SigmaError):
    """Raised when RSU lookup fails."""
    
    def __init__(self, rsu_id: Optional[str] = None, semantic_hash: Optional[int] = None):
        if rsu_id:
            message = f"RSU not found: {rsu_id}"
        elif semantic_hash:
            message = f"No RSU for hash: {semantic_hash:016x}"
        else:
            message = "RSU not found"
        super().__init__(message, "RSU_NOT_FOUND", is_retryable=False)
        self.rsu_id = rsu_id
        self.semantic_hash = semantic_hash


class RSUStorageError(SigmaError):
    """Raised when RSU storage operations fail."""
    
    def __init__(self, message: str, operation: str = "unknown"):
        super().__init__(message, "RSU_STORAGE_ERROR", is_retryable=True)
        self.operation = operation


class SemanticHashCollisionError(SigmaError):
    """Raised when semantic hash collision detected."""
    
    def __init__(self, hash_value: int):
        message = f"Semantic hash collision: {hash_value:016x}"
        super().__init__(message, "HASH_COLLISION", is_retryable=False)
        self.hash_value = hash_value


class CompressionQualityError(SigmaError):
    """Raised when compression quality below threshold."""
    
    def __init__(self, achieved: float, required: float):
        message = f"Compression quality {achieved:.2%} below required {required:.2%}"
        super().__init__(message, "QUALITY_ERROR", is_retryable=True)
        self.achieved = achieved
        self.required = required
