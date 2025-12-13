"""
ΣLANG Adaptive Compression Selector
====================================

Intelligent algorithm selection based on input characteristics.
Detects data patterns and selects optimal compression strategy.

Features:
- Lightweight pattern detection (< 1ms)
- Entropy-based analysis
- Distribution characteristic detection
- Repetition pattern analysis
- Decision tree for algorithm selection
- Per-pattern performance tracking

Copyright 2025 - Ryot LLM Project
"""

import struct
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Available compression strategies."""
    PATTERN = auto()       # Best for repetitive, structured data
    REFERENCE = auto()     # Best for similar/reference data
    DELTA = auto()         # Best for incremental changes
    LOSSLESS = auto()      # Fallback, guaranteed correctness
    RAW = auto()           # No compression, minimal overhead


@dataclass
class DataCharacteristics:
    """Detected characteristics of input data."""
    
    # Entropy metrics
    entropy: float = 0.0                      # Shannon entropy (0-8 for bytes)
    local_entropy: float = 0.0                # Entropy of first 256 bytes
    
    # Distribution
    unique_bytes: int = 0                     # Number of unique byte values
    repetition_ratio: float = 0.0             # Fraction of repeated bytes
    max_run_length: int = 0                   # Longest run of same byte
    avg_run_length: float = 0.0               # Average run length
    
    # Structure
    has_patterns: bool = False                # Detected repeating patterns
    pattern_coverage: float = 0.0             # % of data in patterns
    ascii_density: float = 0.0                # Fraction of ASCII bytes
    
    # Temporal
    locality: float = 0.0                     # Temporal locality score
    delta_entropy: float = 0.0                # Entropy of deltas
    
    # Classification
    data_type: str = "unknown"                # Detected data type
    estimated_compression_ratio: float = 1.0 # Estimated ratio after compression
    
    detection_time_ms: float = 0.0            # Time to detect


@dataclass
class CompressionDecision:
    """Decision made by adaptive selector."""
    
    strategy: CompressionStrategy
    confidence: float = 0.0                   # 0-1 confidence in decision
    characteristics: DataCharacteristics = field(default_factory=DataCharacteristics)
    reasoning: str = ""
    decision_time_ms: float = 0.0


class PatternDetector:
    """Detects repetitive patterns in binary data."""
    
    MIN_PATTERN_LENGTH = 4
    MAX_PATTERN_LENGTH = 256
    SAMPLE_SIZE = 2048
    
    @staticmethod
    def detect_patterns(data: bytes, max_patterns: int = 10) -> Tuple[List[bytes], float]:
        """
        Detect repeating patterns in data.
        
        Returns:
            (patterns, coverage_ratio)
        """
        if len(data) < PatternDetector.MIN_PATTERN_LENGTH:
            return [], 0.0
        
        # Sample first part for speed
        sample_size = min(len(data), PatternDetector.SAMPLE_SIZE)
        sample = data[:sample_size]
        
        patterns = {}
        
        # Find patterns from length 4 to min(32, sample_size/4)
        max_len = min(32, sample_size // 4)
        for pattern_len in range(PatternDetector.MIN_PATTERN_LENGTH, max_len + 1):
            for i in range(sample_size - pattern_len):
                pattern = sample[i:i + pattern_len]
                
                # Only count if appears 2+ times
                count = patterns.get(pattern, 0)
                if count == 0:
                    # Check if pattern exists elsewhere
                    rest = sample[i + pattern_len:]
                    if pattern in rest:
                        patterns[pattern] = 2
                elif count > 0:
                    patterns[pattern] = count + 1
        
        # Sort by frequency * length (best compression)
        ranked = sorted(
            patterns.items(),
            key=lambda x: x[1] * len(x[0]),
            reverse=True
        )[:max_patterns]
        
        found_patterns = [p[0] for p in ranked]
        
        # Estimate coverage
        coverage = 0
        data_copy = sample
        for pattern in found_patterns:
            parts = data_copy.split(pattern)
            coverage += (len(parts) - 1) * len(pattern)
            data_copy = parts[0]  # For next iteration, only use first part
        
        coverage_ratio = min(1.0, coverage / len(sample)) if sample else 0.0
        
        return found_patterns, coverage_ratio
    
    @staticmethod
    def get_run_lengths(data: bytes) -> Tuple[int, float]:
        """
        Analyze run-length encoding potential.
        
        Returns:
            (max_run_length, avg_run_length)
        """
        if not data:
            return 0, 0.0
        
        runs = []
        current_run = 1
        
        for i in range(1, min(len(data), 1024)):
            if data[i] == data[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        
        if current_run > 1:
            runs.append(current_run)
        
        if not runs:
            return 1, 1.0
        
        max_run = max(runs)
        avg_run = sum(runs) / len(runs)
        
        return max_run, avg_run


class EntropyAnalyzer:
    """Analyzes information entropy of data."""
    
    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        """
        Calculate Shannon entropy of data.
        
        Returns:
            Entropy value (0-8 for bytes)
        """
        if not data:
            return 0.0
        
        counts = Counter(data)
        entropy = 0.0
        data_len = len(data)
        
        for count in counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    @staticmethod
    def calculate_local_entropy(data: bytes, window_size: int = 256) -> float:
        """Calculate entropy of first window."""
        if not data:
            return 0.0
        
        window = data[:min(len(data), window_size)]
        return EntropyAnalyzer.calculate_entropy(window)
    
    @staticmethod
    def calculate_delta_entropy(data: bytes) -> float:
        """
        Calculate entropy of deltas (differences).
        Low delta entropy suggests good delta encoding potential.
        """
        if len(data) < 2:
            return 0.0
        
        deltas = bytes(data[i] ^ data[i - 1] for i in range(1, min(len(data), 512)))
        return EntropyAnalyzer.calculate_entropy(deltas)
    
    @staticmethod
    def estimate_compression_ratio(entropy: float, data_len: int) -> float:
        """
        Estimate compression ratio based on entropy.
        
        Theoretical lower bound: entropy / 8 (entropy bits per byte)
        Practical estimate: add overhead
        """
        if data_len < 64:
            return 1.0  # Very small data, compression overhead dominates
        
        # Theoretical: entropy / 8 bytes per byte
        theoretical = entropy / 8.0
        
        # Add overhead: ~5% + 32 bytes per KB
        overhead = 0.05 + (32 / data_len) * 100
        
        estimated = theoretical + overhead
        return max(0.1, min(1.0, estimated))


class DataTypeClassifier:
    """Classifies data type for strategy selection."""
    
    @staticmethod
    def classify(characteristics: DataCharacteristics) -> Tuple[str, float]:
        """
        Classify data type and return confidence.
        
        Returns:
            (data_type, confidence)
        """
        entropy = characteristics.entropy
        local_entropy = characteristics.local_entropy
        unique_bytes = characteristics.unique_bytes
        repetition_ratio = characteristics.repetition_ratio
        ascii_density = characteristics.ascii_density
        max_run = characteristics.max_run_length
        
        # High repetition, low entropy = highly compressible
        if repetition_ratio > 0.5 and entropy < 2.0:
            return "highly_repetitive", 0.95
        
        # High unique bytes, high entropy = low compressibility
        if unique_bytes > 200 and entropy > 6.5:
            return "random_or_binary", 0.90
        
        # Moderate entropy, reasonable unique = mixed
        if 3.0 <= entropy <= 5.0 and 50 <= unique_bytes <= 150:
            return "mixed_structured", 0.85
        
        # Text-like (high ASCII, moderate entropy)
        if ascii_density > 0.7 and entropy < 5.5:
            return "text_like", 0.80
        
        # Delta-friendly (high locality, low delta entropy)
        if characteristics.delta_entropy < characteristics.entropy * 0.6:
            return "delta_friendly", 0.75
        
        # RLE-friendly (high max run length relative to total)
        if max_run > 10 and max_run > characteristics.avg_run_length * 3:
            return "rle_friendly", 0.70
        
        return "unknown", 0.0


class AdaptiveCompressionSelector:
    """
    Intelligent compression algorithm selector.
    
    Decision process:
    1. Analyze input characteristics (< 1ms)
    2. Classify data type
    3. Predict compression ratios for each strategy
    4. Select best strategy
    5. Return decision with confidence and reasoning
    
    Lightweight - designed for real-time use.
    """
    
    def __init__(self, enable_tracking: bool = True):
        """
        Initialize selector.
        
        Args:
            enable_tracking: Track performance by pattern for learning
        """
        self.enable_tracking = enable_tracking
        self.pattern_performance: Dict[str, Dict[str, Any]] = {}
        self.strategy_stats: Dict[CompressionStrategy, Dict[str, float]] = {
            strategy: {
                'uses': 0,
                'avg_compression_ratio': 1.0,
                'avg_time_ms': 0.0,
            }
            for strategy in CompressionStrategy
        }
    
    def select(self, data: bytes) -> CompressionDecision:
        """
        Select best compression strategy for data.
        
        Args:
            data: Input bytes to compress
            
        Returns:
            CompressionDecision with strategy and reasoning
        """
        start_time = time.time()
        
        # Step 1: Analyze characteristics
        characteristics = self._analyze_characteristics(data)
        
        # Step 2: Classify data
        data_type, confidence = DataTypeClassifier.classify(characteristics)
        characteristics.data_type = data_type
        
        # Step 3: Select strategy
        strategy = self._select_strategy(data, characteristics, data_type)
        
        # Step 4: Generate reasoning
        reasoning = self._generate_reasoning(
            characteristics, data_type, strategy, confidence
        )
        
        decision_time_ms = (time.time() - start_time) * 1000
        
        decision = CompressionDecision(
            strategy=strategy,
            confidence=confidence,
            characteristics=characteristics,
            reasoning=reasoning,
            decision_time_ms=decision_time_ms
        )
        
        # Track for learning
        if self.enable_tracking:
            self._track_decision(decision)
        
        return decision
    
    def _analyze_characteristics(self, data: bytes) -> DataCharacteristics:
        """Analyze input data characteristics."""
        characteristics = DataCharacteristics()
        
        if not data:
            return characteristics
        
        # Entropy
        characteristics.entropy = EntropyAnalyzer.calculate_entropy(data)
        characteristics.local_entropy = EntropyAnalyzer.calculate_local_entropy(data)
        characteristics.delta_entropy = EntropyAnalyzer.calculate_delta_entropy(data)
        
        # Distribution
        characteristics.unique_bytes = len(set(data))
        sample = data[:min(len(data), 1024)]
        if sample:
            repeat_count = len(sample) - len(set(sample))
            characteristics.repetition_ratio = repeat_count / len(sample)
        
        # Runs
        max_run, avg_run = PatternDetector.get_run_lengths(data)
        characteristics.max_run_length = max_run
        characteristics.avg_run_length = avg_run
        
        # Patterns
        patterns, coverage = PatternDetector.detect_patterns(data)
        characteristics.has_patterns = len(patterns) > 0
        characteristics.pattern_coverage = coverage
        
        # ASCII
        ascii_count = sum(1 for b in data[:min(len(data), 512)] if 32 <= b < 127)
        characteristics.ascii_density = ascii_count / min(len(data), 512) if data else 0.0
        
        # Locality (consecutive byte differences)
        if len(data) > 1:
            diffs = sum(
                1 for i in range(1, min(len(data), 256))
                if abs(data[i] - data[i - 1]) <= 16
            )
            characteristics.locality = diffs / min(len(data) - 1, 255)
        
        # Estimated compression ratio
        characteristics.estimated_compression_ratio = \
            EntropyAnalyzer.estimate_compression_ratio(
                characteristics.entropy, len(data)
            )
        
        return characteristics
    
    def _select_strategy(
        self,
        data: bytes,
        characteristics: DataCharacteristics,
        data_type: str
    ) -> CompressionStrategy:
        """
        Select best strategy based on characteristics.
        
        Decision tree:
        - high_entropy (>6.5) → RAW (incompressible)
        - high_repetition (>0.5) → PATTERN or REFERENCE
        - low_entropy (<2.0) → LOSSLESS (guaranteed correctness)
        - text/structured → REFERENCE (good baseline)
        - delta_friendly → DELTA
        - default → REFERENCE
        """
        entropy = characteristics.entropy
        rep_ratio = characteristics.repetition_ratio
        
        # Very small data: use RAW
        if len(data) < 64:
            return CompressionStrategy.RAW
        
        # Incompressible (high entropy)
        if entropy > 6.8:
            return CompressionStrategy.RAW
        
        # Highly repetitive
        if rep_ratio > 0.6 and characteristics.has_patterns:
            return CompressionStrategy.PATTERN
        
        # Very low entropy (highly compressible)
        if entropy < 1.5:
            return CompressionStrategy.PATTERN
        
        # Delta-friendly
        if characteristics.delta_entropy < entropy * 0.5 and characteristics.locality > 0.6:
            return CompressionStrategy.DELTA
        
        # RLE-friendly (high run lengths)
        if characteristics.max_run_length > 20:
            return CompressionStrategy.PATTERN
        
        # Default to REFERENCE (good all-around)
        return CompressionStrategy.REFERENCE
    
    def _generate_reasoning(
        self,
        characteristics: DataCharacteristics,
        data_type: str,
        strategy: CompressionStrategy,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for decision."""
        entropy = characteristics.entropy
        rep_ratio = characteristics.repetition_ratio
        
        reasons = []
        reasons.append(f"Data type: {data_type} (confidence: {confidence:.0%})")
        reasons.append(f"Entropy: {entropy:.2f}/8.0")
        reasons.append(f"Repetition: {rep_ratio:.1%}")
        
        if characteristics.has_patterns:
            reasons.append(f"Patterns detected: {characteristics.pattern_coverage:.1%} coverage")
        
        if characteristics.delta_entropy < entropy * 0.6:
            reasons.append(f"Delta-friendly (Δ entropy: {characteristics.delta_entropy:.2f})")
        
        reasons.append(f"→ Strategy: {strategy.name}")
        
        return " | ".join(reasons)
    
    def _track_decision(self, decision: CompressionDecision):
        """Track decision for learning."""
        pattern_key = decision.characteristics.data_type
        
        if pattern_key not in self.pattern_performance:
            self.pattern_performance[pattern_key] = {
                'uses': 0,
                'strategies': {},
                'avg_compression_ratio': 1.0,
            }
        
        self.pattern_performance[pattern_key]['uses'] += 1
        
        strategy_key = decision.strategy.name
        if strategy_key not in self.pattern_performance[pattern_key]['strategies']:
            self.pattern_performance[pattern_key]['strategies'][strategy_key] = {
                'uses': 0,
                'total_ratio': 0.0,
            }
        
        self.pattern_performance[pattern_key]['strategies'][strategy_key]['uses'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        total_patterns = sum(
            p['uses'] for p in self.pattern_performance.values()
        )
        
        pattern_dist = {
            k: v['uses'] for k, v in self.pattern_performance.items()
        } if self.pattern_performance else {}
        
        return {
            'total_selections': total_patterns,
            'pattern_distribution': pattern_dist,
            'pattern_performance': self.pattern_performance,
            'strategy_stats': {
                strategy.name: stats
                for strategy, stats in self.strategy_stats.items()
            }
        }


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def select_compression_strategy(data: bytes) -> Tuple[CompressionStrategy, CompressionDecision]:
    """
    Convenience function for one-shot compression strategy selection.
    
    Args:
        data: Input bytes
        
    Returns:
        (CompressionStrategy, CompressionDecision)
    """
    selector = AdaptiveCompressionSelector()
    decision = selector.select(data)
    return decision.strategy, decision


# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================

def analyze_data_patterns(data: bytes) -> Dict[str, Any]:
    """
    Comprehensive analysis of data patterns.
    Useful for debugging and understanding compression behavior.
    """
    selector = AdaptiveCompressionSelector(enable_tracking=False)
    decision = selector.select(data)
    
    characteristics = decision.characteristics
    
    return {
        'data_length': len(data),
        'unique_bytes': characteristics.unique_bytes,
        'entropy': characteristics.entropy,
        'local_entropy': characteristics.local_entropy,
        'delta_entropy': characteristics.delta_entropy,
        'repetition_ratio': characteristics.repetition_ratio,
        'max_run_length': characteristics.max_run_length,
        'avg_run_length': characteristics.avg_run_length,
        'has_patterns': characteristics.has_patterns,
        'pattern_coverage': characteristics.pattern_coverage,
        'ascii_density': characteristics.ascii_density,
        'locality': characteristics.locality,
        'data_type': characteristics.data_type,
        'estimated_compression_ratio': characteristics.estimated_compression_ratio,
        'recommended_strategy': decision.strategy.name,
        'confidence': decision.confidence,
        'reasoning': decision.reasoning,
        'detection_time_ms': decision.decision_time_ms,
    }
