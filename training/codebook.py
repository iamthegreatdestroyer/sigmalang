"""
ΣLANG Learned Codebook
======================

Dynamically learns semantic patterns from usage, allocating Tier 2 primitives
(Σ₁₂₈ - Σ₂₅₅) to frequently occurring patterns for maximum compression.

Training Procedures:
1. Pattern Observation: Track semantic tree structures during normal usage
2. Frequency Analysis: Identify patterns exceeding threshold
3. Pattern Extraction: Extract minimal distinguishing features
4. Codebook Promotion: Allocate primitive IDs to high-value patterns
5. Continuous Refinement: Adapt to evolving usage patterns

Copyright 2025 - Ryot LLM Project
"""

import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from heapq import nlargest
import pickle

import sys
from pathlib import Path
# Ensure sigmalang is importable
_root = Path(__file__).parent.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from sigmalang.core.primitives import (
    SemanticNode, SemanticTree, LearnedPrimitive,
    ExistentialPrimitive, PRIMITIVE_REGISTRY,
    LEARNED_PRIMITIVE_START, LEARNED_PRIMITIVE_END, LEARNED_PRIMITIVE_COUNT
)


# ============================================================================
# PATTERN REPRESENTATION
# ============================================================================

@dataclass
class PatternSignature:
    """
    Unique signature for a semantic pattern.
    Captures structural and content features for matching.
    """
    # Structural features
    primitive_sequence: Tuple[int, ...]    # Ordered primitive IDs
    depth: int                              # Tree depth
    node_count: int                         # Total nodes
    branching_factor: float                 # Average children per node
    
    # Content features
    content_hash: bytes                     # Hash of value content
    has_values: bool                        # Whether pattern includes values
    value_types: Tuple[str, ...]           # Types of values (str, int, etc.)
    
    # Template features
    template_string: str                    # Human-readable template
    variable_positions: Tuple[int, ...]    # Positions of variable content
    
    def __hash__(self):
        return hash((self.primitive_sequence, self.content_hash))
    
    def __eq__(self, other):
        if not isinstance(other, PatternSignature):
            return False
        return (self.primitive_sequence == other.primitive_sequence and
                self.content_hash == other.content_hash)
    
    def to_bytes(self) -> bytes:
        """Serialize signature to bytes."""
        return hashlib.sha256(
            str(self.primitive_sequence).encode() + self.content_hash
        ).digest()[:16]
    
    def similarity(self, other: 'PatternSignature') -> float:
        """Compute similarity score with another signature."""
        # Primitive sequence similarity (Jaccard)
        set1 = set(self.primitive_sequence)
        set2 = set(other.primitive_sequence)
        jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
        
        # Structural similarity
        depth_sim = 1.0 - abs(self.depth - other.depth) / max(self.depth, other.depth, 1)
        node_sim = 1.0 - abs(self.node_count - other.node_count) / max(self.node_count, other.node_count, 1)
        
        # Weighted combination
        return 0.6 * jaccard + 0.2 * depth_sim + 0.2 * node_sim


@dataclass
class PatternCandidate:
    """A candidate pattern under observation."""
    signature: PatternSignature
    examples: List[SemanticTree] = field(default_factory=list)
    frequency: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    compression_potential: float = 0.0
    
    def update(self, tree: SemanticTree):
        """Update candidate with new observation."""
        self.frequency += 1
        self.last_seen = time.time()
        if len(self.examples) < 10:  # Keep limited examples
            self.examples.append(tree)
        
        # Estimate compression potential
        avg_size = np.mean([t.node_count for t in self.examples])
        self.compression_potential = avg_size * self.frequency


# ============================================================================
# PATTERN EXTRACTOR
# ============================================================================

class PatternExtractor:
    """
    Extracts pattern signatures from semantic trees.
    Identifies the structural and content features that define a pattern.
    """
    
    def extract(self, tree: SemanticTree) -> PatternSignature:
        """Extract pattern signature from a semantic tree."""
        # Get primitive sequence (pre-order traversal)
        primitives = self._get_primitive_sequence(tree.root)
        
        # Get structural features
        depth = tree.depth
        node_count = tree.node_count
        branching = self._compute_branching_factor(tree.root)
        
        # Get content features
        values = self._extract_values(tree.root)
        content_hash = self._hash_content(values)
        value_types = tuple(type(v).__name__ for v in values if v is not None)
        
        # Generate template
        template, var_positions = self._generate_template(tree.root)
        
        return PatternSignature(
            primitive_sequence=tuple(primitives),
            depth=depth,
            node_count=node_count,
            branching_factor=branching,
            content_hash=content_hash,
            has_values=bool(values),
            value_types=value_types,
            template_string=template,
            variable_positions=tuple(var_positions)
        )
    
    def _get_primitive_sequence(self, node: SemanticNode) -> List[int]:
        """Get ordered sequence of primitives."""
        result = [node.primitive]
        for child in node.children:
            result.extend(self._get_primitive_sequence(child))
        return result
    
    def _compute_branching_factor(self, node: SemanticNode) -> float:
        """Compute average branching factor."""
        total_children = 0
        total_nodes = 0
        
        def count(n):
            nonlocal total_children, total_nodes
            total_nodes += 1
            total_children += len(n.children)
            for child in n.children:
                count(child)
        
        count(node)
        return total_children / total_nodes if total_nodes > 0 else 0
    
    def _extract_values(self, node: SemanticNode) -> List[Any]:
        """Extract all values from tree."""
        values = []
        if node.value is not None:
            values.append(node.value)
        for child in node.children:
            values.extend(self._extract_values(child))
        return values
    
    def _hash_content(self, values: List[Any]) -> bytes:
        """Create content hash from values."""
        # Hash types and structure, not exact values (for template matching)
        type_string = '|'.join(type(v).__name__ for v in values)
        return hashlib.md5(type_string.encode()).digest()[:8]
    
    def _generate_template(self, node: SemanticNode, position: int = 0) -> Tuple[str, List[int]]:
        """Generate human-readable template with variable positions."""
        var_positions = []
        
        prim_name = PRIMITIVE_REGISTRY.get_name(node.primitive)
        
        if node.value is not None:
            var_positions.append(position)
            template = f"[{prim_name}:VAR]"
        else:
            template = f"[{prim_name}]"
        
        child_templates = []
        child_pos = position + 1
        for child in node.children:
            child_template, child_vars = self._generate_template(child, child_pos)
            child_templates.append(child_template)
            var_positions.extend(child_vars)
            child_pos += child.node_count()
        
        if child_templates:
            template += f"({', '.join(child_templates)})"
        
        return template, var_positions


# ============================================================================
# LEARNED CODEBOOK
# ============================================================================

class LearnedCodebook:
    """
    Manages learned semantic patterns and their mappings to Tier 2 primitives.
    
    The codebook dynamically allocates primitive IDs (Σ₁₂₈ - Σ₂₅₅) to
    frequently occurring patterns, achieving near-constant-time encoding
    for common usage patterns.
    """
    
    OBSERVATION_THRESHOLD = 5      # Min observations before consideration
    PROMOTION_THRESHOLD = 20       # Frequency required for promotion
    SIMILARITY_THRESHOLD = 0.85    # Similarity threshold for matching
    MAX_PATTERNS = LEARNED_PRIMITIVE_COUNT  # 128 patterns max
    
    def __init__(self, codebook_path: Optional[Path] = None):
        self.codebook_path = codebook_path
        
        # Pattern storage
        self.patterns: Dict[int, LearnedPrimitive] = {}  # ID -> Pattern
        self.signature_index: Dict[bytes, int] = {}       # Signature hash -> ID
        
        # Observation tracking
        self.candidates: Dict[bytes, PatternCandidate] = {}
        self.extractor = PatternExtractor()
        
        # Statistics
        self.match_count = 0
        self.miss_count = 0
        self.promotions = 0
        
        # Load existing codebook
        if codebook_path and codebook_path.exists():
            self.load(codebook_path)
    
    def observe(self, tree: SemanticTree):
        """
        Observe a semantic tree for pattern learning.
        Called during normal encoding to build pattern statistics.
        """
        signature = self.extractor.extract(tree)
        sig_hash = signature.to_bytes()
        
        # Check if already promoted
        if sig_hash in self.signature_index:
            pattern_id = self.signature_index[sig_hash]
            self.patterns[pattern_id].frequency += 1
            self.patterns[pattern_id].last_used = time.time()
            return
        
        # Update or create candidate
        if sig_hash in self.candidates:
            self.candidates[sig_hash].update(tree)
        else:
            self.candidates[sig_hash] = PatternCandidate(
                signature=signature,
                examples=[tree],
                frequency=1
            )
        
        # Check for promotion
        candidate = self.candidates[sig_hash]
        if candidate.frequency >= self.PROMOTION_THRESHOLD:
            self._promote_candidate(sig_hash, candidate)
    
    def match(self, tree: SemanticTree) -> Optional[int]:
        """
        Check if a tree matches a learned pattern.
        Returns pattern ID if match found, None otherwise.
        """
        signature = self.extractor.extract(tree)
        sig_hash = signature.to_bytes()
        
        # Exact match
        if sig_hash in self.signature_index:
            self.match_count += 1
            pattern_id = self.signature_index[sig_hash]
            self.patterns[pattern_id].frequency += 1
            self.patterns[pattern_id].last_used = time.time()
            return pattern_id
        
        # Approximate match
        best_match = None
        best_similarity = 0.0
        
        for pattern_id, pattern in self.patterns.items():
            stored_sig = self._get_signature_from_pattern(pattern)
            if stored_sig:
                sim = signature.similarity(stored_sig)
                if sim > best_similarity and sim >= self.SIMILARITY_THRESHOLD:
                    best_similarity = sim
                    best_match = pattern_id
        
        if best_match is not None:
            self.match_count += 1
            return best_match
        
        self.miss_count += 1
        return None
    
    def expand(self, pattern_id: int) -> SemanticTree:
        """
        Expand a pattern ID back to a semantic tree template.
        The template may have variable slots for specific content.
        """
        if pattern_id not in self.patterns:
            raise ValueError(f"Unknown pattern ID: {pattern_id}")
        
        pattern = self.patterns[pattern_id]
        
        # Reconstruct from template
        # For now, return a placeholder tree
        root = SemanticNode(
            primitive=ExistentialPrimitive.ABSTRACT,
            value=pattern.expansion_template
        )
        
        return SemanticTree(root=root, source_text=pattern.expansion_template)
    
    def _promote_candidate(self, sig_hash: bytes, candidate: PatternCandidate):
        """Promote a candidate to a full pattern."""
        if len(self.patterns) >= self.MAX_PATTERNS:
            # Evict least valuable pattern
            self._evict_least_valuable()
        
        # Allocate new ID
        pattern_id = self._allocate_id()
        
        # Create learned primitive
        embedding = self._compute_pattern_embedding(candidate)
        
        primitive = LearnedPrimitive(
            id=pattern_id,
            pattern_signature=sig_hash,
            expansion_template=candidate.signature.template_string,
            semantic_embedding=embedding,
            frequency=candidate.frequency,
            compression_ratio=candidate.compression_potential / max(candidate.frequency, 1),
            created_at=candidate.first_seen,
            last_used=time.time()
        )
        
        # Store
        self.patterns[pattern_id] = primitive
        self.signature_index[sig_hash] = pattern_id
        
        # Remove from candidates
        del self.candidates[sig_hash]
        
        self.promotions += 1
    
    def _allocate_id(self) -> int:
        """Allocate next available primitive ID."""
        used_ids = set(self.patterns.keys())
        for i in range(LEARNED_PRIMITIVE_START, LEARNED_PRIMITIVE_END + 1):
            if i not in used_ids:
                return i
        raise RuntimeError("No available primitive IDs")
    
    def _evict_least_valuable(self):
        """Evict the least valuable pattern."""
        if not self.patterns:
            return
        
        # Score by frequency * recency
        def value_score(p: LearnedPrimitive) -> float:
            recency = 1.0 / (time.time() - p.last_used + 1)
            return p.frequency * recency
        
        least_valuable_id = min(self.patterns.keys(), 
                                key=lambda k: value_score(self.patterns[k]))
        
        pattern = self.patterns[least_valuable_id]
        del self.signature_index[pattern.pattern_signature]
        del self.patterns[least_valuable_id]
    
    def _compute_pattern_embedding(self, candidate: PatternCandidate) -> np.ndarray:
        """Compute embedding vector for a pattern."""
        # Use primitive frequencies
        embedding = np.zeros(256)
        for prim in candidate.signature.primitive_sequence:
            if prim < 256:
                embedding[prim] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _get_signature_from_pattern(self, pattern: LearnedPrimitive) -> Optional[PatternSignature]:
        """Reconstruct signature from pattern (for similarity matching)."""
        # This is a simplified version - full implementation would store more
        return None  # For now, rely on exact hash matching
    
    def save(self, path: Optional[Path] = None):
        """Save codebook to disk."""
        path = path or self.codebook_path
        if not path:
            return
        
        data = {
            'patterns': {
                str(k): v.to_dict() for k, v in self.patterns.items()
            },
            'signature_index': {
                k.hex(): v for k, v in self.signature_index.items()
            },
            'statistics': {
                'match_count': self.match_count,
                'miss_count': self.miss_count,
                'promotions': self.promotions
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """Load codebook from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.patterns = {
            int(k): LearnedPrimitive.from_dict(v) 
            for k, v in data.get('patterns', {}).items()
        }
        
        self.signature_index = {
            bytes.fromhex(k): v 
            for k, v in data.get('signature_index', {}).items()
        }
        
        stats = data.get('statistics', {})
        self.match_count = stats.get('match_count', 0)
        self.miss_count = stats.get('miss_count', 0)
        self.promotions = stats.get('promotions', 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get codebook statistics."""
        total = self.match_count + self.miss_count
        return {
            'pattern_count': len(self.patterns),
            'candidate_count': len(self.candidates),
            'match_count': self.match_count,
            'miss_count': self.miss_count,
            'match_rate': self.match_count / total if total > 0 else 0,
            'promotions': self.promotions
        }


# ============================================================================
# TRAINING PROCEDURES
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for codebook training."""
    # Training parameters
    min_observations: int = 5          # Minimum observations before consideration
    promotion_threshold: int = 20      # Frequency required for promotion
    similarity_threshold: float = 0.85 # Similarity threshold for matching
    
    # Batch training
    batch_size: int = 100              # Training batch size
    epochs: int = 10                   # Training epochs for embedding refinement
    learning_rate: float = 0.01        # Learning rate for embeddings
    
    # Pruning
    prune_interval: int = 1000         # Prune candidates every N observations
    candidate_ttl: float = 86400       # Time-to-live for candidates (seconds)
    min_compression_ratio: float = 2.0 # Minimum compression ratio to keep
    
    # Output
    save_interval: int = 500           # Save codebook every N observations
    checkpoint_dir: Optional[Path] = None


class CodebookTrainer:
    """
    Trains and refines the learned codebook through various procedures.
    
    Training modes:
    1. Online learning: Observe patterns during normal usage
    2. Batch training: Process corpus of historical data
    3. Reinforcement: Adjust based on compression performance
    """
    
    def __init__(self, codebook: LearnedCodebook, config: TrainingConfig):
        self.codebook = codebook
        self.config = config
        
        # Training state
        self.observation_count = 0
        self.batch_buffer: List[SemanticTree] = []
        
        # Metrics
        self.compression_history: List[float] = []
        self.pattern_usage: Counter = Counter()
    
    def observe(self, tree: SemanticTree, original_size: int = 0):
        """
        Online observation for continuous learning.
        
        Args:
            tree: Semantic tree to observe
            original_size: Original input size in bytes
        """
        # Observe in codebook
        self.codebook.observe(tree)
        
        self.observation_count += 1
        
        # Track compression if we matched
        pattern_id = self.codebook.match(tree)
        if pattern_id is not None:
            self.pattern_usage[pattern_id] += 1
            if original_size > 0:
                compressed_size = 5  # Pattern reference is ~5 bytes
                ratio = original_size / compressed_size
                self.compression_history.append(ratio)
        
        # Periodic maintenance
        if self.observation_count % self.config.prune_interval == 0:
            self._prune_stale_candidates()
        
        if self.observation_count % self.config.save_interval == 0:
            self._checkpoint()
    
    def batch_train(self, corpus: List[SemanticTree], 
                    original_sizes: Optional[List[int]] = None):
        """
        Batch training on a corpus of semantic trees.
        
        Args:
            corpus: List of semantic trees
            original_sizes: Optional list of original sizes for compression tracking
        """
        if original_sizes is None:
            original_sizes = [0] * len(corpus)
        
        print(f"Starting batch training on {len(corpus)} examples...")
        
        for epoch in range(self.config.epochs):
            epoch_compression = []
            
            for i, (tree, size) in enumerate(zip(corpus, original_sizes)):
                self.observe(tree, size)
                
                # Track epoch metrics
                if self.compression_history:
                    epoch_compression.append(self.compression_history[-1])
                
                # Progress
                if (i + 1) % 100 == 0:
                    avg_ratio = np.mean(epoch_compression) if epoch_compression else 0
                    print(f"  Epoch {epoch+1}/{self.config.epochs}, "
                          f"Example {i+1}/{len(corpus)}, "
                          f"Avg compression: {avg_ratio:.2f}x")
            
            # End of epoch
            avg_compression = np.mean(epoch_compression) if epoch_compression else 0
            print(f"Epoch {epoch+1} complete. "
                  f"Patterns: {len(self.codebook.patterns)}, "
                  f"Avg compression: {avg_compression:.2f}x")
            
            self._refine_embeddings()
        
        print("Batch training complete.")
        self._checkpoint()
    
    def _refine_embeddings(self):
        """Refine pattern embeddings based on usage statistics."""
        # Adjust embeddings to better cluster similar patterns
        # This is a simplified version - full implementation would use
        # contrastive learning or other embedding optimization
        pass
    
    def _prune_stale_candidates(self):
        """Remove candidates that are too old or low-value."""
        current_time = time.time()
        to_remove = []
        
        for sig_hash, candidate in self.codebook.candidates.items():
            # Check age
            age = current_time - candidate.first_seen
            if age > self.config.candidate_ttl:
                # Check if frequent enough to keep
                frequency_rate = candidate.frequency / age * 3600  # Per hour
                if frequency_rate < 0.1:  # Less than 0.1 per hour
                    to_remove.append(sig_hash)
        
        for sig_hash in to_remove:
            del self.codebook.candidates[sig_hash]
        
        if to_remove:
            print(f"Pruned {len(to_remove)} stale candidates")
    
    def _checkpoint(self):
        """Save training checkpoint."""
        if self.config.checkpoint_dir:
            checkpoint_path = self.config.checkpoint_dir / f"codebook_{self.observation_count}.json"
            self.codebook.save(checkpoint_path)
    
    def evaluate(self, test_corpus: List[SemanticTree], 
                 original_sizes: List[int]) -> Dict[str, Any]:
        """
        Evaluate codebook performance on test data.
        
        Returns metrics including compression ratio, match rate, etc.
        """
        matches = 0
        total_original = 0
        total_compressed = 0
        
        for tree, size in zip(test_corpus, original_sizes):
            pattern_id = self.codebook.match(tree)
            
            total_original += size
            
            if pattern_id is not None:
                matches += 1
                total_compressed += 5  # Pattern reference size
            else:
                # Estimate full encoding size
                total_compressed += tree.node_count * 2  # ~2 bytes per node
        
        return {
            'test_size': len(test_corpus),
            'match_count': matches,
            'match_rate': matches / len(test_corpus) if test_corpus else 0,
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'compression_ratio': total_original / total_compressed if total_compressed > 0 else 1,
            'pattern_count': len(self.codebook.patterns)
        }
    
    def get_training_report(self) -> str:
        """Generate a training report."""
        stats = self.codebook.get_stats()
        
        # Top patterns by usage
        top_patterns = self.pattern_usage.most_common(10)
        
        report = [
            "=" * 60,
            "ΣLANG CODEBOOK TRAINING REPORT",
            "=" * 60,
            f"Total observations: {self.observation_count}",
            f"Patterns learned: {stats['pattern_count']}",
            f"Candidates pending: {stats['candidate_count']}",
            f"Match rate: {stats['match_rate']:.2%}",
            f"Promotions: {stats['promotions']}",
            "",
            "Top 10 Patterns by Usage:",
        ]
        
        for pattern_id, count in top_patterns:
            if pattern_id in self.codebook.patterns:
                pattern = self.codebook.patterns[pattern_id]
                report.append(f"  Σ_{pattern_id}: {count} uses - {pattern.expansion_template[:50]}...")
        
        if self.compression_history:
            report.extend([
                "",
                f"Average compression ratio: {np.mean(self.compression_history):.2f}x",
                f"Max compression ratio: {np.max(self.compression_history):.2f}x",
            ])
        
        return "\n".join(report)


# ============================================================================
# CORPUS BUILDER
# ============================================================================

class TrainingCorpusBuilder:
    """
    Builds training corpus from various sources.
    Converts raw data into semantic trees for codebook training.
    """
    
    def __init__(self, parser):
        self.parser = parser
        self.corpus: List[SemanticTree] = []
        self.sizes: List[int] = []
    
    def add_text(self, text: str):
        """Add a text example to corpus."""
        tree = self.parser.parse(text)
        self.corpus.append(tree)
        self.sizes.append(len(text.encode('utf-8')))
    
    def add_texts(self, texts: List[str]):
        """Add multiple text examples."""
        for text in texts:
            self.add_text(text)
    
    def add_conversation(self, messages: List[Dict[str, str]]):
        """Add a conversation (list of message dicts)."""
        for msg in messages:
            if 'content' in msg:
                self.add_text(msg['content'])
    
    def load_from_jsonl(self, path: Path):
        """Load examples from JSONL file."""
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if 'text' in data:
                    self.add_text(data['text'])
                elif 'content' in data:
                    self.add_text(data['content'])
                elif 'messages' in data:
                    self.add_conversation(data['messages'])
    
    def get_corpus(self) -> Tuple[List[SemanticTree], List[int]]:
        """Get the built corpus."""
        return self.corpus, self.sizes
    
    def split(self, train_ratio: float = 0.8) -> Tuple[
        Tuple[List[SemanticTree], List[int]],
        Tuple[List[SemanticTree], List[int]]
    ]:
        """Split corpus into train/test sets."""
        n = len(self.corpus)
        split_idx = int(n * train_ratio)
        
        indices = np.random.permutation(n)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        train_corpus = [self.corpus[i] for i in train_idx]
        train_sizes = [self.sizes[i] for i in train_idx]
        
        test_corpus = [self.corpus[i] for i in test_idx]
        test_sizes = [self.sizes[i] for i in test_idx]
        
        return (train_corpus, train_sizes), (test_corpus, test_sizes)
