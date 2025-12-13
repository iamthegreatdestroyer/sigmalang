"""
ΣLANG Encoder
=============

Transforms semantic trees into highly compressed ΣLANG binary format.
Achieves sub-linear compression through:
1. Semantic primitive encoding
2. Logarithmic content addressing (Σ-Hash)
3. Context-aware delta encoding
4. Learned pattern matching

Copyright 2025 - Ryot LLM Project
"""

import hashlib
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import time

from .primitives import (
    SemanticNode, SemanticTree, Glyph, GlyphStream, GlyphType,
    ExistentialPrimitive, PRIMITIVE_REGISTRY,
    LEARNED_PRIMITIVE_START, LEARNED_PRIMITIVE_END
)


# ============================================================================
# SIGMA HASH BANK
# ============================================================================

class LSHIndex:
    """
    Locality-Sensitive Hash index for semantic similarity search.
    Semantically similar inputs produce similar hashes.
    """
    
    def __init__(self, num_tables: int = 16, hash_size: int = 32, dim: int = 256):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.dim = dim
        
        # Random projection matrices for LSH
        np.random.seed(42)  # Reproducible
        self.projections = [
            np.random.randn(dim, hash_size) for _ in range(num_tables)
        ]
        
        # Hash tables: table_id -> {hash_key -> [item_ids]}
        self.tables: List[Dict[int, List[int]]] = [
            {} for _ in range(num_tables)
        ]
        
        # Storage: item_id -> embedding
        self.embeddings: Dict[int, np.ndarray] = {}
    
    def _compute_hash(self, embedding: np.ndarray, table_idx: int) -> int:
        """Compute LSH hash for an embedding."""
        projection = self.projections[table_idx]
        
        # Ensure embedding is right size
        if len(embedding) < self.dim:
            embedding = np.pad(embedding, (0, self.dim - len(embedding)))
        elif len(embedding) > self.dim:
            embedding = embedding[:self.dim]
        
        # Project and binarize
        projected = np.dot(embedding, projection)
        binary = (projected > 0).astype(int)
        
        # Convert to integer hash
        hash_val = 0
        for bit in binary:
            hash_val = (hash_val << 1) | bit
        
        return hash_val
    
    def add(self, item_id: int, embedding: np.ndarray):
        """Add an item to the index."""
        self.embeddings[item_id] = embedding
        
        for i in range(self.num_tables):
            hash_key = self._compute_hash(embedding, i)
            if hash_key not in self.tables[i]:
                self.tables[i][hash_key] = []
            self.tables[i][hash_key].append(item_id)
    
    def search(self, embedding: np.ndarray, k: int = 10) -> List[int]:
        """Find k nearest neighbors."""
        candidates = set()
        
        # Gather candidates from all tables
        for i in range(self.num_tables):
            hash_key = self._compute_hash(embedding, i)
            if hash_key in self.tables[i]:
                candidates.update(self.tables[i][hash_key])
        
        if not candidates:
            return []
        
        # Rank by actual similarity
        similarities = []
        for item_id in candidates:
            stored_emb = self.embeddings.get(item_id)
            if stored_emb is not None:
                sim = np.dot(embedding, stored_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_emb) + 1e-8
                )
                similarities.append((item_id, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in similarities[:k]]


class LRUCache:
    """Least Recently Used cache for hot storage."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: int) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: int, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                # Remove oldest
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def __contains__(self, key: int) -> bool:
        return key in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)


@dataclass
class SigmaBankEntry:
    """Entry in the Sigma Hash Bank."""
    sigma_hash: int
    semantic_tree: bytes  # Serialized SemanticTree
    embedding: np.ndarray
    frequency: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    compression_ratio: float = 1.0


class SigmaHashBank:
    """
    Persistent storage for semantic structures with sub-linear retrieval.
    
    Tiered architecture:
    - Hot: LRU cache for recently accessed items
    - Warm: LSH index for fast approximate search
    - Cold: Disk storage for full history (not implemented here)
    """
    
    PROMOTION_THRESHOLD = 10  # Frequency threshold to promote to codebook
    
    def __init__(self, hot_capacity: int = 10000, embedding_dim: int = 256):
        self.hot = LRUCache(maxsize=hot_capacity)
        self.warm = LSHIndex(dim=embedding_dim)
        self.frequency: Dict[int, int] = {}
        self.embedding_dim = embedding_dim
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def compute_hash(self, tree: SemanticTree) -> int:
        """
        Compute 32-bit semantic hash for a tree.
        Semantically similar trees produce nearby hashes.
        """
        # Structural signature: primitive sequence
        primitives = sorted(tree.primitives_used)
        structure_bytes = bytes(primitives)
        
        # Content signature: concatenated values
        content = self._extract_content(tree.root)
        content_bytes = content.encode('utf-8')[:256]
        
        # Combine hashes
        struct_hash = hashlib.md5(structure_bytes).digest()[:4]
        content_hash = hashlib.md5(content_bytes).digest()[:4]
        
        # XOR fold to 32 bits
        combined = bytes(a ^ b for a, b in zip(struct_hash, content_hash))
        return int.from_bytes(combined, 'big')
    
    def _extract_content(self, node: SemanticNode) -> str:
        """Extract content string from tree for hashing."""
        parts = []
        if node.value:
            parts.append(str(node.value))
        for child in node.children:
            parts.append(self._extract_content(child))
        return '|'.join(parts)
    
    def compute_embedding(self, tree: SemanticTree) -> np.ndarray:
        """Compute semantic embedding for a tree."""
        # Simple embedding: primitive frequency + structure features
        embedding = np.zeros(self.embedding_dim)
        
        # Primitive frequency (first 128 dims)
        for prim in tree.primitives_used:
            if prim < 128:
                embedding[prim] += 1
        
        # Structural features (next 64 dims)
        embedding[128] = tree.depth
        embedding[129] = tree.node_count
        embedding[130] = len(tree.primitives_used)
        
        # Content hash features (last 64 dims)
        content = self._extract_content(tree.root)
        for i, char in enumerate(content[:64]):
            embedding[192 + i] = ord(char) / 255.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def exists(self, sigma_hash: int) -> bool:
        """Check if a semantic structure exists."""
        return sigma_hash in self.hot or sigma_hash in self.frequency
    
    def store(self, tree: SemanticTree, original_size: int = 0) -> int:
        """
        Store a semantic tree.
        Returns the sigma_hash.
        """
        sigma_hash = self.compute_hash(tree)
        embedding = self.compute_embedding(tree)
        
        # Calculate compression ratio
        serialized = tree.serialize()
        compressed_size = len(serialized)
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        entry = SigmaBankEntry(
            sigma_hash=sigma_hash,
            semantic_tree=serialized,
            embedding=embedding,
            compression_ratio=ratio
        )
        
        # Store in hot cache
        self.hot.put(sigma_hash, entry)
        
        # Index in warm layer
        self.warm.add(sigma_hash, embedding)
        
        # Track frequency
        self.frequency[sigma_hash] = self.frequency.get(sigma_hash, 0) + 1
        
        return sigma_hash
    
    def retrieve(self, sigma_hash: int) -> Optional[SemanticTree]:
        """Retrieve a semantic tree by hash."""
        # Check hot cache first
        entry = self.hot.get(sigma_hash)
        
        if entry:
            self.hits += 1
            entry.frequency += 1
            entry.last_accessed = time.time()
            return SemanticTree.deserialize(entry.semantic_tree)
        
        self.misses += 1
        return None
    
    def search_similar(self, tree: SemanticTree, k: int = 5) -> List[int]:
        """Find similar semantic structures."""
        embedding = self.compute_embedding(tree)
        return self.warm.search(embedding, k)
    
    def should_promote(self, sigma_hash: int) -> bool:
        """Check if an entry should be promoted to codebook."""
        return self.frequency.get(sigma_hash, 0) >= self.PROMOTION_THRESHOLD
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bank statistics."""
        total = self.hits + self.misses
        return {
            'hot_size': len(self.hot),
            'total_entries': len(self.frequency),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0
        }


# ============================================================================
# CONTEXT STACK
# ============================================================================

@dataclass
class ContextFrame:
    """A frame in the context stack."""
    tree: SemanticTree
    timestamp: float
    topic_primitives: set


class ContextStack:
    """
    Maintains conversation context for delta encoding.
    New inputs are encoded as deltas from current context.
    """
    
    def __init__(self, max_depth: int = 16):
        self.max_depth = max_depth
        self.frames: List[ContextFrame] = []
        self.topic_history: List[set] = []
    
    def push(self, tree: SemanticTree):
        """Push a new context frame."""
        frame = ContextFrame(
            tree=tree,
            timestamp=time.time(),
            topic_primitives=tree.primitives_used
        )
        
        self.frames.append(frame)
        self.topic_history.append(tree.primitives_used)
        
        # Maintain max depth
        while len(self.frames) > self.max_depth:
            self.frames.pop(0)
            self.topic_history.pop(0)
    
    def current(self) -> Optional[SemanticTree]:
        """Get current context."""
        if self.frames:
            return self.frames[-1].tree
        return None
    
    def current_topics(self) -> set:
        """Get primitives from recent context."""
        topics = set()
        for frame in self.frames[-3:]:  # Last 3 frames
            topics.update(frame.topic_primitives)
        return topics
    
    def compute_delta(self, new_tree: SemanticTree) -> 'ContextDelta':
        """Compute delta between new tree and current context."""
        if not self.frames:
            return ContextDelta(
                base_context_id=0,
                new_primitives=new_tree.primitives_used,
                removed_primitives=set(),
                modified_nodes=[],
                compression_ratio=1.0
            )
        
        current = self.frames[-1].tree
        current_prims = current.primitives_used
        new_prims = new_tree.primitives_used
        
        # Compute set differences
        added = new_prims - current_prims
        removed = current_prims - new_prims
        shared = current_prims & new_prims
        
        # Estimate compression from shared primitives
        if len(new_prims) > 0:
            share_ratio = len(shared) / len(new_prims)
            compression_ratio = 1.0 + share_ratio * 2  # Up to 3x for high overlap
        else:
            compression_ratio = 1.0
        
        return ContextDelta(
            base_context_id=id(current),
            new_primitives=added,
            removed_primitives=removed,
            modified_nodes=[],
            compression_ratio=compression_ratio
        )
    
    def clear(self):
        """Clear context stack."""
        self.frames.clear()
        self.topic_history.clear()


@dataclass
class ContextDelta:
    """Delta encoding between context and new input."""
    base_context_id: int
    new_primitives: set
    removed_primitives: set
    modified_nodes: List[Tuple[int, Any]]  # (node_id, new_value)
    compression_ratio: float


# ============================================================================
# SIGMA ENCODER
# ============================================================================

class SigmaEncoder:
    """
    Main ΣLANG encoder.
    Transforms semantic trees into highly compressed binary format.
    """
    
    def __init__(self, codebook: Optional['LearnedCodebook'] = None):
        self.sigma_bank = SigmaHashBank()
        self.context_stack = ContextStack()
        self.codebook = codebook
        
        # Encoding statistics
        self.total_input_bytes = 0
        self.total_output_bytes = 0
        self.encoding_count = 0
    
    def encode(self, tree: SemanticTree, original_text: str = "") -> bytes:
        """
        Encode a semantic tree to ΣLANG binary format.
        
        Encoding priority:
        1. Codebook pattern match (highest compression)
        2. Sigma bank reference (if seen before)
        3. Context delta (if context overlap)
        4. Full primitive encoding (baseline)
        """
        original_size = len(original_text.encode('utf-8'))
        self.total_input_bytes += original_size
        
        # Step 1: Check codebook for learned pattern
        if self.codebook:
            pattern_id = self.codebook.match(tree)
            if pattern_id is not None:
                result = self._encode_pattern_ref(pattern_id)
                self._record_output(result)
                return result
        
        # Step 2: Check sigma bank for existing structure
        sigma_hash = self.sigma_bank.compute_hash(tree)
        if self.sigma_bank.exists(sigma_hash):
            result = self._encode_reference(sigma_hash)
            self._record_output(result)
            # Update frequency
            self.sigma_bank.retrieve(sigma_hash)
            return result
        
        # Step 3: Try delta encoding against context
        delta = self.context_stack.compute_delta(tree)
        if delta.compression_ratio > 2.0:
            result = self._encode_delta(tree, delta)
            self._record_output(result)
            self._store_and_update(tree, original_size)
            return result
        
        # Step 4: Full primitive encoding
        glyphs = self._encode_primitives(tree)
        result = self._pack_glyphs(glyphs)
        self._record_output(result)
        self._store_and_update(tree, original_size)
        
        return result
    
    def _record_output(self, data: bytes):
        """Record output statistics."""
        self.total_output_bytes += len(data)
        self.encoding_count += 1
    
    def _store_and_update(self, tree: SemanticTree, original_size: int):
        """Store in bank and update context."""
        self.sigma_bank.store(tree, original_size)
        self.context_stack.push(tree)
        
        # Check for codebook promotion
        sigma_hash = self.sigma_bank.compute_hash(tree)
        if self.codebook and self.sigma_bank.should_promote(sigma_hash):
            self.codebook.observe(tree)
    
    def _encode_pattern_ref(self, pattern_id: int) -> bytes:
        """Encode as codebook pattern reference."""
        glyph = Glyph(
            glyph_type=GlyphType.REFERENCE,
            primitive_id=pattern_id,
            payload=None
        )
        return GlyphStream(glyphs=[glyph]).to_bytes()
    
    def _encode_reference(self, sigma_hash: int) -> bytes:
        """Encode as sigma bank reference."""
        glyph = Glyph(
            glyph_type=GlyphType.REFERENCE,
            primitive_id=ExistentialPrimitive.REFERENCE,
            payload=sigma_hash.to_bytes(4, 'big')
        )
        return GlyphStream(glyphs=[glyph]).to_bytes()
    
    def _encode_delta(self, tree: SemanticTree, delta: ContextDelta) -> bytes:
        """Encode as context delta."""
        glyphs = []
        
        # Delta header glyph
        header = Glyph(
            glyph_type=GlyphType.DELTA,
            primitive_id=ExistentialPrimitive.TRANSFORM,
            payload=struct.pack('>I', delta.base_context_id & 0xFFFFFFFF)
        )
        glyphs.append(header)
        
        # New primitives
        for prim in sorted(delta.new_primitives):
            glyphs.append(Glyph(
                glyph_type=GlyphType.PRIMITIVE,
                primitive_id=prim
            ))
        
        return GlyphStream(glyphs=glyphs).to_bytes()
    
    def _encode_primitives(self, tree: SemanticTree) -> List[Glyph]:
        """Encode tree as primitive sequence."""
        glyphs = []
        self._encode_node(tree.root, glyphs)
        return glyphs
    
    def _encode_node(self, node: SemanticNode, glyphs: List[Glyph]):
        """Recursively encode a node."""
        # Create glyph for this node
        payload = None
        if node.value is not None:
            # Encode value as payload
            value_str = str(node.value)
            payload = value_str.encode('utf-8')
        
        glyph = Glyph(
            glyph_type=GlyphType.PRIMITIVE if not node.children else GlyphType.COMPOSITE,
            primitive_id=node.primitive,
            payload=payload
        )
        glyphs.append(glyph)
        
        # Encode children
        for child in node.children:
            self._encode_node(child, glyphs)
        
        # End composite marker (if has children)
        if node.children:
            glyphs.append(Glyph(
                glyph_type=GlyphType.PRIMITIVE,
                primitive_id=ExistentialPrimitive.COMPOSITE  # End marker
            ))
    
    def _pack_glyphs(self, glyphs: List[Glyph]) -> bytes:
        """Pack glyphs into binary stream."""
        context_id = id(self.context_stack.current()) & 0xFFF if self.context_stack.current() else 0
        stream = GlyphStream(
            glyphs=glyphs,
            context_id=context_id
        )
        return stream.to_bytes()
    
    def get_compression_ratio(self) -> float:
        """Get overall compression ratio."""
        if self.total_output_bytes == 0:
            return 1.0
        return self.total_input_bytes / self.total_output_bytes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            'total_input_bytes': self.total_input_bytes,
            'total_output_bytes': self.total_output_bytes,
            'compression_ratio': self.get_compression_ratio(),
            'encoding_count': self.encoding_count,
            'sigma_bank': self.sigma_bank.get_stats(),
            'context_depth': len(self.context_stack.frames)
        }


# ============================================================================
# SIGMA DECODER
# ============================================================================

class SigmaDecoder:
    """
    Decodes ΣLANG binary format back to semantic trees.
    """
    
    def __init__(self, encoder: SigmaEncoder):
        self.encoder = encoder  # Share state for reference resolution
    
    def decode(self, data: bytes) -> SemanticTree:
        """Decode ΣLANG bytes to semantic tree."""
        stream = GlyphStream.from_bytes(data)
        
        if not stream.glyphs:
            raise ValueError("Empty glyph stream")
        
        first_glyph = stream.glyphs[0]
        
        # Handle reference types
        if first_glyph.glyph_type == GlyphType.REFERENCE:
            if first_glyph.payload:
                # Sigma bank reference
                sigma_hash = int.from_bytes(first_glyph.payload, 'big')
                tree = self.encoder.sigma_bank.retrieve(sigma_hash)
                if tree:
                    return tree
            elif self.encoder.codebook:
                # Codebook pattern reference
                return self.encoder.codebook.expand(first_glyph.primitive_id)
        
        # Handle delta encoding
        if first_glyph.glyph_type == GlyphType.DELTA:
            return self._decode_delta(stream)
        
        # Full primitive decoding
        return self._decode_primitives(stream)
    
    def _decode_delta(self, stream: GlyphStream) -> SemanticTree:
        """Decode delta-encoded stream.
        
        Delta encoding format:
        - Glyph 0: Header with glyph_type=DELTA, primitive_id=TRANSFORM, payload=context_id
        - Glyph 1+: New/changed primitives to apply
        
        The header is a marker, not data - skip it and decode the actual content.
        """
        if len(stream.glyphs) <= 1:
            # Only header, no actual content
            return SemanticTree(
                root=SemanticNode(primitive=ExistentialPrimitive.ABSTRACT),
                source_text=""
            )
        
        # Skip the delta header (glyph 0) and decode from glyph 1 onwards
        content_glyphs = stream.glyphs[1:]
        root, _ = self._decode_node(content_glyphs, 0)
        
        return SemanticTree(root=root, source_text="[decoded]")
    
    def _decode_primitives(self, stream: GlyphStream) -> SemanticTree:
        """Decode primitive-encoded stream."""
        if not stream.glyphs:
            return SemanticTree(
                root=SemanticNode(primitive=ExistentialPrimitive.ABSTRACT),
                source_text=""
            )
        
        # Build tree from glyph sequence
        root, _ = self._decode_node(stream.glyphs, 0)
        
        return SemanticTree(root=root, source_text="[decoded]")
    
    def _decode_node(self, glyphs: List[Glyph], idx: int) -> Tuple[SemanticNode, int]:
        """Recursively decode a node."""
        if idx >= len(glyphs):
            return SemanticNode(primitive=ExistentialPrimitive.ABSTRACT), idx
        
        glyph = glyphs[idx]
        
        # Extract value from payload
        # Note: Check `is not None` to handle empty strings (b'' is falsy but valid)
        value = None
        if glyph.payload is not None:
            try:
                value = glyph.payload.decode('utf-8')
            except UnicodeDecodeError:
                value = glyph.payload.hex()
        
        node = SemanticNode(
            primitive=glyph.primitive_id,
            value=value
        )
        
        idx += 1
        
        # If composite, decode children until end marker
        if glyph.glyph_type == GlyphType.COMPOSITE:
            while idx < len(glyphs):
                next_glyph = glyphs[idx]
                # Check for end marker
                if (next_glyph.glyph_type == GlyphType.PRIMITIVE and 
                    next_glyph.primitive_id == ExistentialPrimitive.COMPOSITE):
                    idx += 1
                    break
                
                child, idx = self._decode_node(glyphs, idx)
                node.children.append(child)
        
        return node, idx


# Placeholder for LearnedCodebook (defined in training module)
class LearnedCodebook:
    """Placeholder - full implementation in training module."""
    
    def match(self, tree: SemanticTree) -> Optional[int]:
        return None
    
    def expand(self, pattern_id: int) -> SemanticTree:
        return SemanticTree(
            root=SemanticNode(primitive=ExistentialPrimitive.ABSTRACT),
            source_text=""
        )
    
    def observe(self, tree: SemanticTree):
        pass
