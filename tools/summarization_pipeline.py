"""
Automated Summarization Pipeline - Phase 4 Task 4.4

Compress, store, and retrieve document summaries on demand.
Combines SigmaLang compression with extractive summarization
for efficient knowledge management.

Architecture:
    Input Document --> Sentence Scorer --> Top Sentences
                                              |
                                       Summary Assembler
                                              |
                                       SigmaLang Compressor
                                              |
                                       Summary Store
                                              |
                                       Retrieval API

Features:
- Extractive summarization (no external LLM needed)
- Configurable compression ratio
- Persistent summary store
- Semantic retrieval over summaries
- Incremental updates
"""

import sys
import json
import time
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root))

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SummarizationConfig:
    """Configuration for the summarization pipeline."""

    # Summarization parameters
    target_ratio: float = 0.2  # Target summary length as fraction of original
    min_summary_sentences: int = 3
    max_summary_sentences: int = 50
    sentence_min_length: int = 20  # Min chars for a sentence to be considered
    sentence_max_length: int = 500

    # Scoring weights
    position_weight: float = 0.3  # Earlier sentences score higher
    length_weight: float = 0.1  # Prefer medium-length sentences
    keyword_weight: float = 0.4  # Sentences with key terms score higher
    uniqueness_weight: float = 0.2  # Unique content scores higher

    # Storage
    store_dir: str = ".sigma-summaries"


# =============================================================================
# Sentence Scoring
# =============================================================================

class SentenceScorer:
    """Scores sentences for extractive summarization."""

    def __init__(self, config: SummarizationConfig):
        self.config = config

    def score_sentences(
        self,
        sentences: List[str],
        document: str
    ) -> List[Tuple[int, float, str]]:
        """
        Score sentences for inclusion in summary.

        Returns:
            List of (index, score, sentence) tuples, sorted by score desc
        """
        if not sentences:
            return []

        # Extract keywords
        keywords = self._extract_keywords(document)

        scored = []
        for i, sentence in enumerate(sentences):
            if len(sentence) < self.config.sentence_min_length:
                continue
            if len(sentence) > self.config.sentence_max_length:
                continue

            score = self._score_sentence(sentence, i, len(sentences), keywords)
            scored.append((i, score, sentence))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total_sentences: int,
        keywords: Dict[str, float]
    ) -> float:
        """Calculate composite score for a sentence."""
        score = 0.0

        # Position score (earlier = better)
        position_score = 1.0 - (position / max(1, total_sentences))
        score += self.config.position_weight * position_score

        # Length score (prefer medium length)
        ideal_length = 150  # chars
        length_diff = abs(len(sentence) - ideal_length) / ideal_length
        length_score = max(0, 1.0 - length_diff)
        score += self.config.length_weight * length_score

        # Keyword score
        sentence_words = set(sentence.lower().split())
        keyword_overlap = sum(
            keywords.get(word, 0) for word in sentence_words
        )
        max_keyword_score = sum(keywords.values()) if keywords else 1
        keyword_score = keyword_overlap / max(1, max_keyword_score)
        score += self.config.keyword_weight * keyword_score

        # Uniqueness score (presence of named entities, numbers, etc.)
        has_numbers = bool(re.search(r'\d+', sentence))
        has_quotes = '"' in sentence or "'" in sentence
        uniqueness_score = 0.5 + (0.25 if has_numbers else 0) + (0.25 if has_quotes else 0)
        score += self.config.uniqueness_weight * uniqueness_score

        return score

    def _extract_keywords(self, document: str, top_k: int = 30) -> Dict[str, float]:
        """Extract keywords with TF-IDF-like scores."""
        words = re.findall(r'\w+', document.lower())

        if not words:
            return {}

        # Stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'this', 'that', 'and', 'or', 'not', 'no', 'but', 'if',
            'so', 'as', 'we', 'he', 'she', 'they', 'them', 'their', 'its'
        }

        # Count frequencies
        word_freq: Dict[str, int] = defaultdict(int)
        for word in words:
            if len(word) > 2 and word not in stopwords:
                word_freq[word] += 1

        # Normalize to TF scores
        max_freq = max(word_freq.values()) if word_freq else 1
        keywords = {
            word: freq / max_freq
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
        }

        return keywords


# =============================================================================
# Text Splitter
# =============================================================================

class SentenceSplitter:
    """Splits text into sentences."""

    def split(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on ., !, ?
        # Handle common abbreviations
        text = text.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')
        text = text.replace('Dr.', 'Dr').replace('etc.', 'etc')
        text = text.replace('e.g.', 'eg').replace('i.e.', 'ie')
        text = text.replace('vs.', 'vs').replace('St.', 'St')

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean up
        cleaned = []
        for s in sentences:
            s = s.strip()
            if s:
                cleaned.append(s)

        return cleaned


# =============================================================================
# Summary Models
# =============================================================================

@dataclass
class Summary:
    """A generated summary."""

    summary_id: str
    source_id: str  # Hash of original document
    title: str
    summary_text: str
    original_length: int  # chars
    summary_length: int  # chars
    compression_ratio: float
    sentence_count: int
    top_keywords: List[str]
    created_at: str = field(default_factory=lambda: __import__('datetime').datetime.now(
        __import__('datetime').timezone.utc).isoformat())
    sigma_compressed: Optional[bytes] = None
    sigma_compressed_size: int = 0

    @property
    def total_compression_ratio(self) -> float:
        """Get total compression (summarization + SigmaLang)."""
        if self.sigma_compressed_size > 0:
            return self.original_length / max(1, self.sigma_compressed_size)
        return self.compression_ratio

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('sigma_compressed', None)
        return d


# =============================================================================
# Summarization Pipeline
# =============================================================================

class SummarizationPipeline:
    """
    Main summarization pipeline.

    Extracts key sentences, compresses with SigmaLang, and stores
    for on-demand retrieval.
    """

    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self.splitter = SentenceSplitter()
        self.scorer = SentenceScorer(self.config)

        self.summaries: Dict[str, Summary] = {}

        # Storage
        self._store_dir = Path(self.config.store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'documents_summarized': 0,
            'total_original_chars': 0,
            'total_summary_chars': 0,
            'total_sigma_bytes': 0
        }

    def summarize(
        self,
        text: str,
        title: str = "",
        target_ratio: Optional[float] = None
    ) -> Summary:
        """
        Summarize a document.

        Args:
            text: Document text
            title: Optional title
            target_ratio: Override target summary ratio

        Returns:
            Generated Summary object
        """
        target_ratio = target_ratio or self.config.target_ratio

        # Split into sentences
        sentences = self.splitter.split(text)

        # Score sentences
        scored = self.scorer.score_sentences(sentences, text)

        # Select top sentences
        target_sentences = max(
            self.config.min_summary_sentences,
            min(
                self.config.max_summary_sentences,
                int(len(sentences) * target_ratio)
            )
        )

        selected = scored[:target_sentences]

        # Re-order by position (preserve document order)
        selected.sort(key=lambda x: x[0])

        # Assemble summary
        summary_text = ' '.join(s[2] for s in selected)

        # Extract keywords
        keywords = self.scorer._extract_keywords(text, top_k=10)
        top_keywords = list(keywords.keys())[:10]

        # Generate IDs
        source_id = hashlib.sha256(text[:1000].encode('utf-8')).hexdigest()[:16]
        summary_id = hashlib.sha256(
            (source_id + str(time.time())).encode()
        ).hexdigest()[:12]

        # Try SigmaLang compression
        sigma_compressed = None
        sigma_size = 0
        try:
            from sigmalang.core.parser import SemanticParser
            from sigmalang.core.encoder import SigmaEncoder

            parser = SemanticParser()
            encoder = SigmaEncoder()

            tree = parser.parse(summary_text)
            sigma_compressed = encoder.encode(tree)
            sigma_size = len(sigma_compressed)
        except Exception as e:
            logger.debug(f"SigmaLang compression skipped: {e}")

        # Create summary
        summary = Summary(
            summary_id=summary_id,
            source_id=source_id,
            title=title or f"Summary of {len(text)} chars",
            summary_text=summary_text,
            original_length=len(text),
            summary_length=len(summary_text),
            compression_ratio=round(len(text) / max(1, len(summary_text)), 2),
            sentence_count=len(selected),
            top_keywords=top_keywords,
            sigma_compressed=sigma_compressed,
            sigma_compressed_size=sigma_size
        )

        # Store
        self.summaries[summary_id] = summary
        self._save_summary(summary)

        # Update stats
        self.stats['documents_summarized'] += 1
        self.stats['total_original_chars'] += len(text)
        self.stats['total_summary_chars'] += len(summary_text)
        self.stats['total_sigma_bytes'] += sigma_size

        return summary

    def summarize_file(self, file_path: str, title: str = "") -> Optional[Summary]:
        """Summarize a file."""
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            content = path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None

        return self.summarize(content, title=title or path.name)

    def summarize_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None
    ) -> List[Summary]:
        """Summarize all eligible files in a directory."""
        path = Path(dir_path)
        extensions = extensions or ['.txt', '.md', '.py', '.js']

        summaries = []

        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                try:
                    summary = self.summarize_file(str(file_path))
                    if summary:
                        summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Failed to summarize {file_path}: {e}")

        return summaries

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant summaries for a query.

        Args:
            query: Search query
            top_k: Maximum results

        Returns:
            List of matching summaries with relevance scores
        """
        query_terms = set(re.findall(r'\w+', query.lower()))

        scored = []
        for summary in self.summaries.values():
            # Score by keyword overlap
            summary_terms = set(summary.summary_text.lower().split())
            keyword_overlap = len(query_terms & set(summary.top_keywords))
            content_overlap = len(query_terms & summary_terms)

            score = keyword_overlap * 2.0 + content_overlap * 0.1

            if score > 0:
                scored.append((summary, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for summary, score in scored[:top_k]:
            results.append({
                'summary_id': summary.summary_id,
                'title': summary.title,
                'summary_text': summary.summary_text,
                'score': round(score, 4),
                'compression_ratio': summary.compression_ratio,
                'total_compression_ratio': summary.total_compression_ratio,
                'top_keywords': summary.top_keywords
            })

        return results

    def _save_summary(self, summary: Summary) -> None:
        """Save summary to disk."""
        # Save metadata
        meta_path = self._store_dir / f"{summary.summary_id}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, indent=2)

        # Save compressed data if available
        if summary.sigma_compressed:
            sigma_path = self._store_dir / f"{summary.summary_id}.sigma"
            sigma_path.write_bytes(summary.sigma_compressed)

    def load_store(self) -> int:
        """Load summaries from disk store."""
        loaded = 0

        for meta_path in self._store_dir.glob('*.json'):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                summary = Summary(**data)
                self.summaries[summary.summary_id] = summary
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load {meta_path}: {e}")

        return loaded

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total_original = self.stats['total_original_chars']
        total_summary = self.stats['total_summary_chars']
        total_sigma = self.stats['total_sigma_bytes']

        return {
            **self.stats,
            'summaries_stored': len(self.summaries),
            'avg_compression_ratio': round(
                total_original / max(1, total_summary), 2
            ),
            'avg_total_compression': round(
                total_original / max(1, total_sigma), 2
            ) if total_sigma > 0 else None,
            'store_dir': str(self._store_dir)
        }


# =============================================================================
# Global Pipeline Instance
# =============================================================================

_global_pipeline: Optional[SummarizationPipeline] = None


def get_summarization_pipeline() -> SummarizationPipeline:
    """Get or create the global summarization pipeline."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = SummarizationPipeline()
    return _global_pipeline


def initialize_summarization(
    target_ratio: float = 0.2,
    store_dir: str = ".sigma-summaries"
) -> SummarizationPipeline:
    """
    Initialize the summarization pipeline.

    Usage:
        from tools.summarization_pipeline import initialize_summarization

        pipeline = initialize_summarization(target_ratio=0.2)

        # Summarize text
        summary = pipeline.summarize(long_document, title="My Document")
        print(f"Compression: {summary.total_compression_ratio:.1f}x")
        print(f"Keywords: {summary.top_keywords}")

        # Summarize file
        summary = pipeline.summarize_file("docs/paper.md")

        # Search summaries
        results = pipeline.retrieve("machine learning optimization")
    """
    global _global_pipeline
    config = SummarizationConfig(
        target_ratio=target_ratio,
        store_dir=store_dir
    )
    _global_pipeline = SummarizationPipeline(config)
    return _global_pipeline


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SigmaLang Summarization Pipeline")
    parser.add_argument('input', help='File or directory to summarize')
    parser.add_argument('--ratio', '-r', type=float, default=0.2, help='Summary ratio (default: 0.2)')
    parser.add_argument('--search', '-s', default=None, help='Search stored summaries')
    parser.add_argument('--output', '-o', default='.sigma-summaries', help='Output directory')

    args = parser.parse_args()

    pipeline = initialize_summarization(target_ratio=args.ratio, store_dir=args.output)

    if args.search:
        loaded = pipeline.load_store()
        print(f"Loaded {loaded} summaries")
        results = pipeline.retrieve(args.search)
        for r in results:
            print(f"\n[{r['score']:.4f}] {r['title']}")
            print(f"  {r['summary_text'][:200]}...")
    else:
        path = Path(args.input)
        if path.is_dir():
            summaries = pipeline.summarize_directory(str(path))
            print(f"\nSummarized {len(summaries)} files")
        else:
            summary = pipeline.summarize_file(str(path))
            if summary:
                print(f"\nTitle: {summary.title}")
                print(f"Compression: {summary.compression_ratio:.1f}x")
                print(f"Keywords: {', '.join(summary.top_keywords)}")
                print(f"\nSummary:\n{summary.summary_text}")

    stats = pipeline.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
