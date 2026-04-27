"""
ΣLANG Contracts Module
======================

Re-exports core interface protocols from sigmalang.api.interfaces
for convenient access. All contract definitions live in the API module.

Usage:
    from sigmalang.contracts import CompressionEngine, CodebookProtocol
"""

from sigmalang.api.interfaces import (
    CodebookProtocol,
    CompressionEngine,
    RSUManager,
    StorageBackend,
)

__all__ = [
    "CompressionEngine",
    "RSUManager",
    "CodebookProtocol",
    "StorageBackend",
]
