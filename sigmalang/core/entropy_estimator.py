"""
Entropy Estimator & Compression Efficiency - Phase 7 Track 10

Calculates theoretical compression bounds for input text using information
theory, then reports how close SigmaLang's actual compression is to the
theoretical optimum.

Theory:
    Shannon's Source Coding Theorem states that no lossless compression
    can achieve a compressed size smaller than the entropy of the source.

    H(X) = -sum(p(x) * log2(p(x)))  for all symbols x

    The compression efficiency is:
        eta = H(X) / actual_bits_per_symbol

    Where eta = 1.0 means optimal (impossible in practice), and higher
    values of eta indicate better compression.

Research Basis:
    - "Language Modeling Is Compression" (Sep 2023) - 84 HF upvotes
    - "Compression Represents Intelligence Linearly" (Apr 2024) - 28 upvotes
    - "Information Capacity" (Nov 2025) - compression as LLM efficiency metric

Usage:
    from sigmalang.core.entropy_estimator import EntropyAnalyzer

    analyzer = EntropyAnalyzer()
    report = analyzer.analyze("The quick brown fox jumps over the lazy dog")
    print(f"Entropy: {report.entropy_bpc:.3f} bits/char")
    print(f"Theoretical minimum: {report.theoretical_min_bytes} bytes")
    print(f"SigmaLang efficiency: {report.efficiency_pct:.1f}%")
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Entropy Models
# =============================================================================

class EntropyOrder(Enum):
    """Order of entropy estimation."""
    ZEROTH = 0  # Character frequency only (memoryless source)
    FIRST = 1   # Bigram conditional entropy
    SECOND = 2  # Trigram conditional entropy


@dataclass
class EntropyReport:
    """Complete entropy analysis report for a text input."""

    # Input characteristics
    input_length: int = 0
    unique_symbols: int = 0
    alphabet_size: int = 0

    # Entropy measurements (bits per character)
    entropy_h0: float = 0.0  # Zeroth-order (character frequency)
    entropy_h1: float = 0.0  # First-order (bigram conditional)
    entropy_h2: float = 0.0  # Second-order (trigram conditional)

    # Derived metrics
    entropy_bpc: float = 0.0  # Best entropy estimate (lowest order available)
    redundancy: float = 0.0   # 1 - (entropy / max_entropy)

    # Compression bounds
    theoretical_min_bits: int = 0  # Shannon lower bound
    theoretical_min_bytes: int = 0

    # SigmaLang comparison (filled when actual compression data provided)
    actual_compressed_bytes: int = 0
    actual_bits_per_char: float = 0.0
    efficiency_pct: float = 0.0  # How close to theoretical optimum (0-100%)
    excess_bits_pct: float = 0.0  # How many extra bits beyond theoretical

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_length': self.input_length,
            'unique_symbols': self.unique_symbols,
            'entropy_h0_bpc': round(self.entropy_h0, 4),
            'entropy_h1_bpc': round(self.entropy_h1, 4),
            'entropy_h2_bpc': round(self.entropy_h2, 4),
            'best_entropy_bpc': round(self.entropy_bpc, 4),
            'redundancy': round(self.redundancy, 4),
            'theoretical_min_bytes': self.theoretical_min_bytes,
            'actual_compressed_bytes': self.actual_compressed_bytes,
            'efficiency_pct': round(self.efficiency_pct, 1),
            'excess_bits_pct': round(self.excess_bits_pct, 1),
        }


# =============================================================================
# Entropy Calculators
# =============================================================================

def zeroth_order_entropy(text: str) -> float:
    """
    Calculate zeroth-order (unigram) Shannon entropy.

    H0 = -sum(p(c) * log2(p(c))) for each character c

    This is the simplest model, treating each character as independent.
    """
    if not text:
        return 0.0

    freq = Counter(text)
    n = len(text)
    entropy = 0.0

    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def first_order_entropy(text: str) -> float:
    """
    Calculate first-order (bigram) conditional entropy.

    H1 = -sum_a( p(a) * sum_b( p(b|a) * log2(p(b|a)) ) )

    Models the probability of each character given the previous one.
    Always <= H0.
    """
    if len(text) < 2:
        return zeroth_order_entropy(text)

    # Count bigrams and unigrams
    unigram_counts = Counter(text[:-1])  # Context characters
    bigram_counts = Counter(zip(text[:-1], text[1:]))
    n = len(text) - 1

    entropy = 0.0
    for (a, b), ab_count in bigram_counts.items():
        p_ab = ab_count / n
        p_b_given_a = ab_count / unigram_counts[a]
        if p_b_given_a > 0:
            entropy -= p_ab * math.log2(p_b_given_a)

    return entropy


def second_order_entropy(text: str) -> float:
    """
    Calculate second-order (trigram) conditional entropy.

    H2 = -sum_{a,b}( p(a,b) * sum_c( p(c|a,b) * log2(p(c|a,b)) ) )

    Models probability of each character given two previous ones.
    Always <= H1 <= H0.
    """
    if len(text) < 3:
        return first_order_entropy(text)

    # Count trigrams and bigram contexts
    bigram_counts = Counter()
    trigram_counts = Counter()

    for i in range(len(text) - 2):
        bigram = (text[i], text[i + 1])
        trigram = (text[i], text[i + 1], text[i + 2])
        bigram_counts[bigram] += 1
        trigram_counts[trigram] += 1

    n = len(text) - 2
    entropy = 0.0

    for (a, b, c), abc_count in trigram_counts.items():
        p_abc = abc_count / n
        p_c_given_ab = abc_count / bigram_counts[(a, b)]
        if p_c_given_ab > 0:
            entropy -= p_abc * math.log2(p_c_given_ab)

    return entropy


def byte_entropy(data: bytes) -> float:
    """Calculate zeroth-order entropy of raw bytes."""
    if not data:
        return 0.0

    freq = Counter(data)
    n = len(data)
    entropy = 0.0

    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


# =============================================================================
# Symbol Distribution Analysis
# =============================================================================

def symbol_distribution(text: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Analyze the symbol frequency distribution.

    Returns the top-N most frequent symbols with their probabilities
    and self-information (surprise) values.
    """
    freq = Counter(text)
    n = len(text)
    result = []

    for char, count in freq.most_common(top_n):
        p = count / n
        self_info = -math.log2(p) if p > 0 else 0.0

        # Display representation
        if char == ' ':
            display = '<SPACE>'
        elif char == '\n':
            display = '<NL>'
        elif char == '\t':
            display = '<TAB>'
        elif ord(char) < 32:
            display = f'<0x{ord(char):02X}>'
        else:
            display = char

        result.append({
            'symbol': display,
            'count': count,
            'probability': round(p, 6),
            'self_information_bits': round(self_info, 3),
            'contribution_bits': round(p * self_info, 6),
        })

    return result


# =============================================================================
# Main Analyzer
# =============================================================================

class EntropyAnalyzer:
    """
    Comprehensive entropy analysis and compression efficiency estimation.

    Computes multi-order entropy estimates for input text and compares
    against SigmaLang's actual compression performance.

    Usage:
        analyzer = EntropyAnalyzer()

        # Basic entropy analysis
        report = analyzer.analyze("Hello world...")
        print(f"Entropy: {report.entropy_bpc:.3f} bpc")
        print(f"Theoretical min: {report.theoretical_min_bytes} bytes")

        # With actual compression comparison
        report = analyzer.analyze(text, actual_compressed_size=len(compressed))
        print(f"Efficiency: {report.efficiency_pct:.1f}%")

        # Analyze raw encoded bytes
        byte_report = analyzer.analyze_bytes(encoded_bytes)
    """

    def __init__(self, max_order: int = 2):
        """
        Args:
            max_order: Maximum entropy order to compute (0, 1, or 2).
                Higher orders are more accurate but slower for large inputs.
        """
        self.max_order = min(max_order, 2)

    def analyze(
        self,
        text: str,
        actual_compressed_size: int = 0
    ) -> EntropyReport:
        """
        Perform full entropy analysis on text.

        Args:
            text: Input text to analyze
            actual_compressed_size: Optional actual compressed size in bytes
                for efficiency comparison

        Returns:
            EntropyReport with all metrics
        """
        if not text:
            return EntropyReport()

        report = EntropyReport()
        report.input_length = len(text)
        report.unique_symbols = len(set(text))
        report.alphabet_size = report.unique_symbols

        # Maximum possible entropy (uniform distribution)
        max_entropy = math.log2(report.unique_symbols) if report.unique_symbols > 1 else 0

        # Zeroth-order entropy (always computed)
        report.entropy_h0 = zeroth_order_entropy(text)

        # First-order (conditional on previous character)
        if self.max_order >= 1 and len(text) >= 2:
            report.entropy_h1 = first_order_entropy(text)
        else:
            report.entropy_h1 = report.entropy_h0

        # Second-order (conditional on two previous characters)
        if self.max_order >= 2 and len(text) >= 3:
            report.entropy_h2 = second_order_entropy(text)
        else:
            report.entropy_h2 = report.entropy_h1

        # Best (lowest) entropy estimate
        report.entropy_bpc = min(report.entropy_h0, report.entropy_h1, report.entropy_h2)

        # Redundancy: how far from maximum entropy
        if max_entropy > 0:
            report.redundancy = 1 - (report.entropy_bpc / max_entropy)

        # Theoretical minimum compressed size
        report.theoretical_min_bits = math.ceil(report.entropy_bpc * len(text))
        report.theoretical_min_bytes = math.ceil(report.theoretical_min_bits / 8)

        # Efficiency comparison
        if actual_compressed_size > 0:
            report.actual_compressed_bytes = actual_compressed_size
            report.actual_bits_per_char = (actual_compressed_size * 8) / max(1, len(text))

            if report.actual_bits_per_char > 0:
                report.efficiency_pct = min(
                    100.0,
                    (report.entropy_bpc / report.actual_bits_per_char) * 100
                )

            if report.theoretical_min_bytes > 0:
                report.excess_bits_pct = max(
                    0,
                    (actual_compressed_size - report.theoretical_min_bytes)
                    / report.theoretical_min_bytes * 100
                )

        return report

    def analyze_bytes(self, data: bytes) -> Dict[str, Any]:
        """
        Analyze entropy of raw byte data (e.g., encoded output).

        Useful for measuring how much room remains for meta-token
        compression on already-encoded streams.
        """
        if not data:
            return {'entropy_bpb': 0.0, 'size': 0}

        h0 = byte_entropy(data)
        max_entropy = 8.0  # Maximum for byte data

        theoretical_min = math.ceil(h0 * len(data) / 8)

        return {
            'size_bytes': len(data),
            'entropy_bpb': round(h0, 4),  # bits per byte
            'max_entropy_bpb': max_entropy,
            'redundancy': round(1 - h0 / max_entropy, 4),
            'theoretical_min_bytes': theoretical_min,
            'compressible_pct': round((1 - theoretical_min / max(1, len(data))) * 100, 1),
            'unique_bytes': len(set(data)),
        }

    def compare_encodings(
        self,
        text: str,
        encodings: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Compare multiple encoding sizes against theoretical bounds.

        Args:
            text: Original text
            encodings: Dict of {encoding_name: compressed_size_bytes}

        Returns:
            Comparison report with efficiency for each encoding
        """
        report = self.analyze(text)

        results = {
            'input_length': len(text),
            'entropy_bpc': round(report.entropy_bpc, 4),
            'theoretical_min_bytes': report.theoretical_min_bytes,
            'encodings': {}
        }

        for name, size in encodings.items():
            bpc = (size * 8) / max(1, len(text))
            efficiency = min(100.0, (report.entropy_bpc / bpc) * 100) if bpc > 0 else 0
            ratio = len(text.encode('utf-8')) / max(1, size)

            results['encodings'][name] = {
                'compressed_bytes': size,
                'bits_per_char': round(bpc, 4),
                'compression_ratio': round(ratio, 2),
                'efficiency_pct': round(efficiency, 1),
                'excess_over_optimal_pct': round(
                    max(0, (size - report.theoretical_min_bytes)
                        / max(1, report.theoretical_min_bytes) * 100), 1
                ),
            }

        # Rank by efficiency
        ranked = sorted(
            results['encodings'].items(),
            key=lambda x: x[1]['efficiency_pct'],
            reverse=True
        )
        results['ranking'] = [name for name, _ in ranked]

        return results

    def get_distribution(self, text: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get symbol frequency distribution analysis."""
        return symbol_distribution(text, top_n)


# =============================================================================
# Global Analyzer Instance
# =============================================================================

_global_analyzer: Optional[EntropyAnalyzer] = None


def get_entropy_analyzer(max_order: int = 2) -> EntropyAnalyzer:
    """Get or create the global entropy analyzer."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = EntropyAnalyzer(max_order=max_order)
    return _global_analyzer


def estimate_entropy(text: str) -> float:
    """Quick entropy estimate in bits per character."""
    return get_entropy_analyzer().analyze(text).entropy_bpc


def compression_efficiency(text: str, compressed_size: int) -> float:
    """Quick compression efficiency percentage (0-100%)."""
    report = get_entropy_analyzer().analyze(text, actual_compressed_size=compressed_size)
    return report.efficiency_pct
