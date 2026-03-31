"""
ΣLANG Pattern Learning - Dynamic Codebook Adaptation

Learns compression patterns from usage data and adapts the codebook
dynamically. Bridges the gap between static Tier 1 primitives and
the online learning pipeline (sigmalang.training.online_learner).

This module provides lightweight, synchronous pattern learning
suitable for embedding in the encode/decode hot path.

For asynchronous background learning, see:
    - sigmalang.training.online_learner (continuous learning)
    - sigmalang.training.adaptive_pruner (codebook pruning)
    - sigmalang.training.ab_tester (A/B testing)
"""

import hashlib
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Dynamic pattern learner for SigmaLang codebook adaptation.

    Observes text patterns during encoding and tracks candidates
    for Tier 2 primitive promotion.
    """

    def __init__(self, max_patterns: int = 1024, promotion_threshold: int = 5):
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.codebook: Dict[str, int] = {}
        self._frequency: Counter = Counter()
        self._max_patterns = max_patterns
        self._promotion_threshold = promotion_threshold
        self._total_observations = 0

    def learn_pattern(self, text: str) -> Dict[str, Any]:
        """
        Learn compression pattern from text.

        Extracts n-gram patterns, tracks frequencies, and identifies
        candidates for codebook promotion.

        Args:
            text: Input text to analyze

        Returns:
            Learned pattern information
        """
        self._total_observations += 1

        # Extract word n-grams (2-4 grams)
        words = text.split()
        extracted = []
        for n in range(2, min(5, len(words) + 1)):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                self._frequency[gram] += 1
                extracted.append(gram)

        # Identify new candidates above threshold
        new_candidates = 0
        for gram in extracted:
            if self._frequency[gram] >= self._promotion_threshold:
                key = self._pattern_key(gram)
                if key not in self.patterns:
                    self.patterns[key] = {
                        'pattern': gram,
                        'frequency': self._frequency[gram],
                        'first_seen': self._total_observations,
                        'promoted': False,
                    }
                    new_candidates += 1
                else:
                    self.patterns[key]['frequency'] = self._frequency[gram]

        # Evict low-frequency patterns if over capacity
        if len(self._frequency) > self._max_patterns * 2:
            self._evict_stale()

        return {
            'pattern_type': self._classify_pattern(text),
            'complexity': len(words),
            'n_grams_extracted': len(extracted),
            'new_candidates': new_candidates,
            'total_patterns': len(self.patterns),
            'learned': True,
        }

    def adapt_codebook(self, new_patterns: Dict[str, Any]) -> bool:
        """
        Adapt codebook with new patterns.

        Args:
            new_patterns: New patterns to incorporate

        Returns:
            Success status
        """
        for key, value in new_patterns.items():
            self.patterns[key] = value
            if isinstance(value, dict) and 'pattern' in value:
                pid = len(self.codebook) + 128  # Tier 2 range
                if pid <= 255:
                    self.codebook[value['pattern']] = pid
        return True

    def get_promotion_candidates(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top patterns ready for Tier 2 promotion.

        Returns patterns sorted by frequency that haven't been promoted yet.
        """
        candidates = [
            p for p in self.patterns.values()
            if not p.get('promoted', False) and p.get('frequency', 0) >= self._promotion_threshold
        ]
        candidates.sort(key=lambda p: p.get('frequency', 0), reverse=True)
        return candidates[:top_n]

    def promote_pattern(self, pattern_key: str) -> Optional[int]:
        """
        Promote a pattern to a Tier 2 primitive.

        Returns assigned primitive ID (128-255) or None if full.
        """
        if pattern_key not in self.patterns:
            return None

        pattern = self.patterns[pattern_key]
        if pattern.get('promoted'):
            return self.codebook.get(pattern['pattern'])

        pid = len(self.codebook) + 128
        if pid > 255:
            logger.warning("Tier 2 codebook full (128 entries)")
            return None

        self.codebook[pattern['pattern']] = pid
        pattern['promoted'] = True
        pattern['primitive_id'] = pid
        logger.info(f"Promoted pattern '{pattern['pattern']}' to primitive {pid}")
        return pid

    def primitives_used(self) -> int:
        """Return number of Tier 2 primitives allocated."""
        return len(self.codebook)

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        promoted = sum(1 for p in self.patterns.values() if p.get('promoted'))
        return {
            'total_observations': self._total_observations,
            'tracked_patterns': len(self.patterns),
            'frequency_entries': len(self._frequency),
            'promoted_primitives': promoted,
            'codebook_size': len(self.codebook),
            'codebook_capacity': 128,
            'codebook_utilization': round(len(self.codebook) / 128, 3),
        }

    def _pattern_key(self, text: str) -> str:
        """Generate a stable key for a pattern."""
        return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:12]

    def _classify_pattern(self, text: str) -> str:
        """Classify the dominant pattern type in text."""
        words = text.split()
        if not words:
            return 'empty'

        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.3:
            return 'repetitive'
        elif unique_ratio > 0.9:
            return 'unique'
        elif any(c in text for c in '{}()[];='):
            return 'code'
        elif any(c in text for c in '+-*/=<>'):
            return 'mathematical'
        else:
            return 'natural_language'

    def _evict_stale(self) -> None:
        """Evict lowest-frequency entries to stay within capacity."""
        if len(self._frequency) <= self._max_patterns:
            return
        most_common = self._frequency.most_common(self._max_patterns)
        self._frequency = Counter(dict(most_common))
