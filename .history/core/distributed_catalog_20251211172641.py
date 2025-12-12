"""\
Distributed Catalog Utilities - Phase 2A.5

This module provides a minimal, dependency-free synchronization layer for
`EnhancedAnalogyCatalog` instances.

Design goals:
- No networking assumptions (callers can transport JSON however they want)
- Deterministic merge behavior (conflict policy)
- Compatible with existing persistence format/semantics

Key concepts:
- A *delta* is a JSON payload containing patterns + metadata.
- Deltas can be created from a full catalog, a subset of pattern IDs, or
  incrementally since an ISO timestamp.

Note:
Patterns are serialized the same way as CatalogPersistence:
- If a pattern has `to_dict()`, that is used.
- Otherwise, the string representation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import logging

from .pattern_persistence import EnhancedAnalogyCatalog, PatternMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogDelta:
    """Serializable delta payload for catalog synchronization."""

    version: str
    created_at: str
    patterns: Dict[str, Any]
    metadata: Dict[str, Dict[str, Any]]

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            {
                "version": self.version,
                "created_at": self.created_at,
                "pattern_count": len(self.patterns),
                "patterns": self.patterns,
                "metadata": self.metadata,
            },
            indent=indent,
        )

    @staticmethod
    def from_json(delta_json: str) -> "CatalogDelta":
        data = json.loads(delta_json)
        return CatalogDelta(
            version=str(data.get("version", "1.0")),
            created_at=str(data.get("created_at", "")),
            patterns=dict(data.get("patterns", {})),
            metadata=dict(data.get("metadata", {})),
        )


class CatalogSynchronizer:
    """Create/apply deltas to synchronize EnhancedAnalogyCatalog instances."""

    VERSION = "1.0"

    @staticmethod
    def create_delta(
        catalog: EnhancedAnalogyCatalog,
        *,
        pattern_ids: Optional[Iterable[str]] = None,
    ) -> CatalogDelta:
        """Create a delta payload from the given catalog.

        Args:
            catalog: Source catalog
            pattern_ids: Optional subset of pattern IDs to include. If omitted,
                includes all patterns.

        Returns:
            CatalogDelta
        """
        if pattern_ids is None:
            selected_ids = list(catalog.patterns.keys())
        else:
            selected_ids = [pid for pid in pattern_ids if pid in catalog.patterns]

        patterns_payload: Dict[str, Any] = {}
        metadata_payload: Dict[str, Dict[str, Any]] = {}

        for pid in selected_ids:
            pattern = catalog.patterns[pid]
            meta = catalog.metadata.get(pid)

            if hasattr(pattern, "to_dict"):
                try:
                    patterns_payload[pid] = pattern.to_dict()
                except Exception:
                    patterns_payload[pid] = str(pattern)
            else:
                patterns_payload[pid] = str(pattern)

            if meta is not None:
                metadata_payload[pid] = meta.to_dict()

        return CatalogDelta(
            version=CatalogSynchronizer.VERSION,
            created_at=datetime.now().isoformat(),
            patterns=patterns_payload,
            metadata=metadata_payload,
        )

    @staticmethod
    def create_delta_since(
        catalog: EnhancedAnalogyCatalog,
        *,
        since_iso: str,
    ) -> CatalogDelta:
        """Create a delta containing patterns created at/after `since_iso`."""
        since_dt = datetime.fromisoformat(since_iso)

        def _is_newer(pid: str) -> bool:
            meta = catalog.metadata.get(pid)
            if meta is None:
                return False
            try:
                created_dt = datetime.fromisoformat(meta.created_at)
            except ValueError:
                return False
            return created_dt >= since_dt

        return CatalogSynchronizer.create_delta(
            catalog,
            pattern_ids=[pid for pid in catalog.patterns.keys() if _is_newer(pid)],
        )

    @staticmethod
    def apply_delta(
        catalog: EnhancedAnalogyCatalog,
        delta_json: str,
        *,
        conflict: str = "skip",
    ) -> Tuple[int, int]:
        """Apply a delta to a target catalog.

        Args:
            catalog: Target catalog to mutate
            delta_json: JSON produced by CatalogDelta.to_json
            conflict: What to do when a pattern ID already exists in the target.
                - "skip": keep existing, ignore incoming
                - "overwrite": replace existing

        Returns:
            (added_count, updated_count)
        """
        if conflict not in {"skip", "overwrite"}:
            raise ValueError(f"Invalid conflict policy: {conflict}")

        delta = CatalogDelta.from_json(delta_json)

        added = 0
        updated = 0

        for pid, pattern_payload in delta.patterns.items():
            incoming_meta_dict = delta.metadata.get(pid)

            exists = pid in catalog.patterns
            if exists and conflict == "skip":
                continue

            # Upsert pattern
            catalog.patterns[pid] = pattern_payload

            # Upsert metadata
            if incoming_meta_dict is not None:
                try:
                    meta_obj = PatternMetadata.from_dict(incoming_meta_dict)
                except Exception:
                    # Fall back to minimal metadata if payload is malformed
                    meta_obj = PatternMetadata(
                        pattern_id=pid,
                        created_at=datetime.now().isoformat(),
                    )
            else:
                meta_obj = PatternMetadata(
                    pattern_id=pid,
                    created_at=datetime.now().isoformat(),
                )

            catalog.metadata[pid] = meta_obj

            # Ensure index contains the new/updated entry.
            # For overwrite we clear old entry first to avoid stale term mappings.
            if exists:
                catalog.index.remove_pattern(pid)

            catalog.index.add_pattern(pid, catalog.patterns[pid], meta_obj)

            if exists:
                updated += 1
            else:
                added += 1

        CatalogSynchronizer._reconcile_counter(catalog)

        logger.info(
            "Applied catalog delta: added=%d updated=%d version=%s",
            added,
            updated,
            delta.version,
        )

        return added, updated

    @staticmethod
    def _reconcile_counter(catalog: EnhancedAnalogyCatalog) -> None:
        """Update the catalog's internal counter to avoid ID collisions."""
        max_idx = -1
        for pid in catalog.patterns.keys():
            if pid.startswith("pattern_"):
                suffix = pid[len("pattern_") :]
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
        if max_idx >= 0:
            # Next auto-generated ID should be greater than any existing ID.
            catalog._pattern_counter = max(catalog._pattern_counter, max_idx + 1)
