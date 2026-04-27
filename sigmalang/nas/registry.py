"""
Architecture Registry - Phase 7 Track 3

Stores, retrieves, and compares discovered architectures.
Supports JSON persistence, ranking, and replay.

Usage:
    registry = ArchitectureRegistry("architectures.json")
    registry.register(config, evaluation_result)
    best = registry.get_best(metric='pareto_score')
    registry.save()
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sigmalang.nas.evaluator import EvaluationResult
from sigmalang.nas.search_space import ArchitectureConfig

logger = logging.getLogger(__name__)


class ArchitectureRegistry:
    """
    Persistent registry for discovered architectures.

    Stores architecture configs with their evaluation results.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = Path(filepath) if filepath else None
        self._entries: Dict[str, Dict[str, Any]] = {}

        # Auto-load if file exists
        if self.filepath and self.filepath.exists():
            self.load()

    def register(
        self,
        config: ArchitectureConfig,
        evaluation: Optional[EvaluationResult] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register an architecture in the registry.

        Returns the architecture ID.
        """
        arch_id = config.architecture_id or f"arch-{len(self._entries)}"
        config.architecture_id = arch_id

        entry = {
            'config': config.to_dict(),
            'evaluation': evaluation.to_dict() if evaluation else None,
            'tags': tags or [],
            'registered_at': time.time(),
        }

        self._entries[arch_id] = entry
        logger.info(f"Registered architecture: {arch_id}")
        return arch_id

    def get(self, arch_id: str) -> Optional[Dict[str, Any]]:
        """Get a registered architecture by ID."""
        return self._entries.get(arch_id)

    def get_config(self, arch_id: str) -> Optional[ArchitectureConfig]:
        """Get architecture config by ID."""
        entry = self._entries.get(arch_id)
        if entry is None:
            return None
        return ArchitectureConfig.from_dict(entry['config'])

    def get_best(
        self,
        metric: str = 'pareto_score',
        top_n: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get top-N architectures by a specific metric.

        Args:
            metric: evaluation metric to rank by
            top_n: number of results
        """
        scored = []
        for arch_id, entry in self._entries.items():
            if entry.get('evaluation'):
                score = entry['evaluation'].get(metric, 0.0)
                scored.append((score, arch_id, entry))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [
            {'arch_id': aid, 'score': s, **entry}
            for s, aid, entry in scored[:top_n]
        ]

    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered architectures."""
        return [
            {'arch_id': aid, **entry}
            for aid, entry in self._entries.items()
        ]

    def compare(self, id_a: str, id_b: str) -> Dict[str, Any]:
        """Compare two architectures side by side."""
        a = self._entries.get(id_a)
        b = self._entries.get(id_b)
        if not a or not b:
            return {'error': 'One or both architectures not found'}

        comparison = {'architecture_a': id_a, 'architecture_b': id_b, 'differences': {}}

        config_a = a['config']
        config_b = b['config']
        for key in config_a:
            if key in ('architecture_id', 'parent_id', 'generation'):
                continue
            va = config_a.get(key)
            vb = config_b.get(key)
            if va != vb:
                comparison['differences'][key] = {'a': va, 'b': vb}

        # Compare evaluations if available
        if a.get('evaluation') and b.get('evaluation'):
            eval_a = a['evaluation']
            eval_b = b['evaluation']
            comparison['evaluation_comparison'] = {}
            for key in eval_a:
                if key == 'architecture_id':
                    continue
                va = eval_a.get(key, 0)
                vb = eval_b.get(key, 0)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    comparison['evaluation_comparison'][key] = {
                        'a': va, 'b': vb,
                        'winner': 'a' if va > vb else 'b' if vb > va else 'tie'
                    }

        return comparison

    def remove(self, arch_id: str) -> bool:
        """Remove an architecture from the registry."""
        if arch_id in self._entries:
            del self._entries[arch_id]
            return True
        return False

    def filter_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get architectures with a specific tag."""
        return [
            {'arch_id': aid, **entry}
            for aid, entry in self._entries.items()
            if tag in entry.get('tags', [])
        ]

    def save(self, filepath: Optional[str] = None) -> None:
        """Save registry to JSON file."""
        path = Path(filepath) if filepath else self.filepath
        if path is None:
            logger.warning("No filepath specified for save")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._entries, f, indent=2)

        logger.info(f"Saved {len(self._entries)} architectures to {path}")

    def load(self, filepath: Optional[str] = None) -> None:
        """Load registry from JSON file."""
        path = Path(filepath) if filepath else self.filepath
        if path is None or not path.exists():
            return

        with open(path, 'r', encoding='utf-8') as f:
            self._entries = json.load(f)

        logger.info(f"Loaded {len(self._entries)} architectures from {path}")

    @property
    def count(self) -> int:
        return len(self._entries)

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        summary = {
            'total_architectures': self.count,
            'filepath': str(self.filepath) if self.filepath else None,
        }

        if self._entries:
            best = self.get_best(metric='pareto_score', top_n=1)
            if best:
                summary['best_pareto_score'] = best[0].get('score', 0)
                summary['best_arch_id'] = best[0].get('arch_id', '')

        return summary
