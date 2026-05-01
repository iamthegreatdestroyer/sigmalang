"""
Storage backends for the Token Recycler.

Provides InMemoryStorage (LRU + TTL) and RedisStorage (namespaced keys,
gzip-compressed payloads, Redis TTL).  Use get_storage() to instantiate.
"""

from __future__ import annotations

__all__ = [
    "RecyclerStorage",
    "InMemoryStorage",
    "RedisStorage",
    "get_storage",
]


class RecyclerStorage:
    """Placeholder — implemented in Stage 4."""


class InMemoryStorage(RecyclerStorage):
    """Placeholder — implemented in Stage 4."""


class RedisStorage(RecyclerStorage):
    """Placeholder — implemented in Stage 4."""


def get_storage(backend: str = "memory", **kwargs) -> RecyclerStorage:
    """Placeholder — implemented in Stage 4."""
    raise NotImplementedError
