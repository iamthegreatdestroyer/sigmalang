"""
Semantic Search Engine for sigmalang.

This module provides vector-based semantic search capabilities including
document indexing, similarity search, and hybrid retrieval methods.

Synthesizes patterns from:
- Information Retrieval (TF-IDF, BM25)
- Vector Databases (FAISS-like indexing)
- Hyperdimensional Computing (bundling, binding)
- Approximate Nearest Neighbor (LSH, product quantization)

Classes:
    SearchMode: Enumeration of search modes
    IndexConfig: Configuration for search index
    SearchResult: Single search result with metadata
    SearchResults: Collection of search results
    Document: Document for indexing
    VectorIndex: Core vector storage and retrieval
    InvertedIndex: Token-based inverted index
    HybridSearcher: Combines vector and keyword search
    SemanticSearchEngine: Main unified interface

Example:
    >>> engine = SemanticSearchEngine()
    >>> engine.index_documents([doc1, doc2, doc3])
    >>> results = engine.search("quantum computing applications")
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, 
    Set, Tuple, Union, Generic, TypeVar
)

import numpy as np

from .transformer_embeddings import (
    TransformerEncoder,
    EmbeddingConfig,
    EmbeddingCache,
)


class SearchMode(Enum):
    """Search mode enumeration."""
    VECTOR = "vector"          # Pure vector similarity search
    KEYWORD = "keyword"        # Pure keyword/token search
    HYBRID = "hybrid"          # Combined vector + keyword
    RERANK = "rerank"          # Vector search with reranking


@dataclass
class IndexConfig:
    """Configuration for search index.
    
    Attributes:
        embedding_dim: Dimension of embeddings
        use_normalization: Whether to L2-normalize vectors
        use_quantization: Whether to use product quantization
        num_clusters: Number of clusters for IVF index
        num_probes: Number of probes for IVF search
        use_lsh: Whether to use LSH for approximate search
        lsh_num_tables: Number of LSH hash tables
        lsh_num_bits: Number of bits per hash
        cache_embeddings: Whether to cache embeddings
        max_cache_size: Maximum cache size
    """
    embedding_dim: int = 256
    use_normalization: bool = True
    use_quantization: bool = False
    num_clusters: int = 100
    num_probes: int = 10
    use_lsh: bool = False
    lsh_num_tables: int = 10
    lsh_num_bits: int = 8
    cache_embeddings: bool = True
    max_cache_size: int = 10000


@dataclass
class Document:
    """Document for indexing.
    
    Attributes:
        id: Unique document identifier
        text: Document text content
        title: Optional document title
        metadata: Additional metadata
        embedding: Precomputed embedding (optional)
    """
    id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text[:100].encode()).hexdigest()[:16]
    
    def get_searchable_text(self) -> str:
        """Get text for searching."""
        parts = [self.text]
        if self.title:
            parts.insert(0, self.title)
        return " ".join(parts)


@dataclass
class SearchResult:
    """Single search result.
    
    Attributes:
        document: The matched document
        score: Relevance score
        rank: Result rank (1-based)
        highlights: Text highlights/snippets
        match_type: Type of match (vector/keyword/hybrid)
    """
    document: Document
    score: float
    rank: int = 0
    highlights: List[str] = field(default_factory=list)
    match_type: str = "vector"
    
    def __lt__(self, other: 'SearchResult') -> bool:
        """Compare by score (for heap operations)."""
        return self.score < other.score


@dataclass
class SearchResults:
    """Collection of search results.
    
    Attributes:
        results: List of search results
        query: Original query
        total_matches: Total number of matches
        search_time_ms: Search time in milliseconds
        mode: Search mode used
    """
    results: List[SearchResult]
    query: str
    total_matches: int = 0
    search_time_ms: float = 0.0
    mode: SearchMode = SearchMode.VECTOR
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self) -> Iterator[SearchResult]:
        return iter(self.results)
    
    def __getitem__(self, idx: int) -> SearchResult:
        return self.results[idx]
    
    def get_documents(self) -> List[Document]:
        """Get all documents from results."""
        return [r.document for r in self.results]
    
    def get_top_k(self, k: int) -> List[SearchResult]:
        """Get top k results."""
        return self.results[:k]


class VectorIndex:
    """Core vector storage and similarity search.
    
    Implements efficient vector indexing with optional LSH
    for approximate nearest neighbor search.
    """
    
    def __init__(self, config: Optional[IndexConfig] = None):
        """Initialize vector index.
        
        Args:
            config: Index configuration
        """
        self.config = config or IndexConfig()
        
        # Storage
        self._vectors: Dict[str, np.ndarray] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        
        # Matrix form for batch operations
        self._vector_matrix: Optional[np.ndarray] = None
        self._matrix_dirty: bool = True
        
        # LSH tables (if enabled)
        self._lsh_tables: List[Dict[int, Set[str]]] = []
        self._lsh_projections: List[np.ndarray] = []
        
        self._lock = threading.Lock()
        self._next_idx = 0
        
        if self.config.use_lsh:
            self._initialize_lsh()
    
    def _initialize_lsh(self):
        """Initialize LSH hash tables."""
        for _ in range(self.config.lsh_num_tables):
            # Random projection vectors
            projection = np.random.randn(
                self.config.lsh_num_bits, 
                self.config.embedding_dim
            )
            self._lsh_projections.append(projection)
            self._lsh_tables.append(defaultdict(set))
    
    def _compute_lsh_hash(
        self, 
        vector: np.ndarray, 
        table_idx: int
    ) -> int:
        """Compute LSH hash for a vector.
        
        Args:
            vector: Input vector
            table_idx: Which hash table
            
        Returns:
            Hash value as integer
        """
        projection = self._lsh_projections[table_idx]
        # Sign of projections gives binary hash
        signs = (projection @ vector) > 0
        # Convert to integer hash
        hash_val = sum(b << i for i, b in enumerate(signs))
        return hash_val
    
    def add(self, doc_id: str, vector: np.ndarray):
        """Add vector to index.
        
        Args:
            doc_id: Document identifier
            vector: Embedding vector
        """
        with self._lock:
            # Normalize if configured
            if self.config.use_normalization:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            
            # Store vector
            self._vectors[doc_id] = vector.copy()
            self._id_to_idx[doc_id] = self._next_idx
            self._idx_to_id[self._next_idx] = doc_id
            self._next_idx += 1
            self._matrix_dirty = True
            
            # Add to LSH tables
            if self.config.use_lsh:
                for table_idx in range(self.config.lsh_num_tables):
                    hash_val = self._compute_lsh_hash(vector, table_idx)
                    self._lsh_tables[table_idx][hash_val].add(doc_id)
    
    def remove(self, doc_id: str) -> bool:
        """Remove vector from index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if doc_id not in self._vectors:
                return False
            
            vector = self._vectors[doc_id]
            
            # Remove from LSH tables
            if self.config.use_lsh:
                for table_idx in range(self.config.lsh_num_tables):
                    hash_val = self._compute_lsh_hash(vector, table_idx)
                    self._lsh_tables[table_idx][hash_val].discard(doc_id)
            
            # Remove from storage
            del self._vectors[doc_id]
            idx = self._id_to_idx.pop(doc_id)
            del self._idx_to_id[idx]
            self._matrix_dirty = True
            
            return True
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (doc_id, score) tuples
        """
        if not self._vectors:
            return []
        
        # Normalize query
        if self.config.use_normalization:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        if self.config.use_lsh:
            return self._search_lsh(query_vector, top_k, threshold)
        else:
            return self._search_exact(query_vector, top_k, threshold)
    
    def _search_exact(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float
    ) -> List[Tuple[str, float]]:
        """Exact nearest neighbor search.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Build matrix if needed
        self._ensure_matrix()
        
        if self._vector_matrix is None or len(self._vector_matrix) == 0:
            return []
        
        # Compute all similarities
        similarities = self._vector_matrix @ query_vector
        
        # Get top-k
        if len(similarities) <= top_k:
            top_indices = np.argsort(-similarities)
        else:
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                doc_id = self._idx_to_id.get(idx)
                if doc_id:
                    results.append((doc_id, score))
        
        return results
    
    def _search_lsh(
        self,
        query_vector: np.ndarray,
        top_k: int,
        threshold: float
    ) -> List[Tuple[str, float]]:
        """LSH-based approximate search.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Get candidate set from LSH tables
        candidates: Set[str] = set()
        
        for table_idx in range(self.config.lsh_num_tables):
            hash_val = self._compute_lsh_hash(query_vector, table_idx)
            candidates.update(self._lsh_tables[table_idx].get(hash_val, set()))
        
        if not candidates:
            # Fall back to exact search if no candidates
            return self._search_exact(query_vector, top_k, threshold)
        
        # Score candidates
        results = []
        for doc_id in candidates:
            if doc_id in self._vectors:
                vec = self._vectors[doc_id]
                score = float(np.dot(query_vector, vec))
                if score >= threshold:
                    results.append((doc_id, score))
        
        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _ensure_matrix(self):
        """Ensure vector matrix is up to date."""
        if not self._matrix_dirty:
            return
        
        with self._lock:
            if not self._vectors:
                self._vector_matrix = None
                return
            
            # Build matrix from vectors in index order
            ids = sorted(self._id_to_idx.keys(), key=lambda x: self._id_to_idx[x])
            vectors = [self._vectors[doc_id] for doc_id in ids]
            self._vector_matrix = np.array(vectors)
            self._matrix_dirty = False
    
    def __len__(self) -> int:
        return len(self._vectors)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._vectors


class InvertedIndex:
    """Token-based inverted index for keyword search.
    
    Implements BM25-like scoring for keyword relevance.
    """
    
    # Default stopwords
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'it', 'its', 'as', 'if'
    }
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = False
    ):
        """Initialize inverted index.
        
        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter
            use_stemming: Whether to apply stemming
        """
        self.k1 = k1
        self.b = b
        self.use_stemming = use_stemming
        
        # Inverted index: token -> set of doc_ids
        self._index: Dict[str, Set[str]] = defaultdict(set)
        
        # Document lengths
        self._doc_lengths: Dict[str, int] = {}
        
        # Term frequencies per document
        self._term_freqs: Dict[str, Counter] = {}
        
        # Document frequency per term
        self._doc_freqs: Counter = Counter()
        
        # Total documents
        self._num_docs = 0
        self._avg_doc_length = 0.0
        
        self._lock = threading.Lock()
    
    def add(self, doc_id: str, text: str):
        """Add document to index.
        
        Args:
            doc_id: Document identifier
            text: Document text
        """
        tokens = self._tokenize(text)
        
        with self._lock:
            # Store term frequencies
            term_freq = Counter(tokens)
            self._term_freqs[doc_id] = term_freq
            self._doc_lengths[doc_id] = len(tokens)
            
            # Update inverted index
            for token in set(tokens):
                if doc_id not in self._index[token]:
                    self._index[token].add(doc_id)
                    self._doc_freqs[token] += 1
            
            # Update statistics
            self._num_docs += 1
            self._avg_doc_length = (
                sum(self._doc_lengths.values()) / self._num_docs
            )
    
    def remove(self, doc_id: str) -> bool:
        """Remove document from index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if removed
        """
        with self._lock:
            if doc_id not in self._doc_lengths:
                return False
            
            # Remove from inverted index
            term_freq = self._term_freqs.get(doc_id, {})
            for token in term_freq:
                if doc_id in self._index[token]:
                    self._index[token].discard(doc_id)
                    self._doc_freqs[token] -= 1
                    if self._doc_freqs[token] <= 0:
                        del self._doc_freqs[token]
                        del self._index[token]
            
            # Remove document data
            del self._doc_lengths[doc_id]
            del self._term_freqs[doc_id]
            
            # Update statistics
            self._num_docs -= 1
            if self._num_docs > 0:
                self._avg_doc_length = (
                    sum(self._doc_lengths.values()) / self._num_docs
                )
            else:
                self._avg_doc_length = 0.0
            
            return True
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search using BM25 scoring.
        
        Args:
            query: Query string
            top_k: Number of results
            
        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens or self._num_docs == 0:
            return []
        
        # Get candidate documents
        candidates: Set[str] = set()
        for token in query_tokens:
            candidates.update(self._index.get(token, set()))
        
        if not candidates:
            return []
        
        # Score candidates using BM25
        scores = []
        for doc_id in candidates:
            score = self._compute_bm25(doc_id, query_tokens)
            scores.append((doc_id, score))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _compute_bm25(
        self,
        doc_id: str,
        query_tokens: List[str]
    ) -> float:
        """Compute BM25 score for document.
        
        Args:
            doc_id: Document identifier
            query_tokens: Query tokens
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self._doc_lengths.get(doc_id, 0)
        term_freq = self._term_freqs.get(doc_id, {})
        
        for token in query_tokens:
            if token not in term_freq:
                continue
            
            tf = term_freq[token]
            df = self._doc_freqs.get(token, 0)
            
            # IDF component
            idf = math.log(
                (self._num_docs - df + 0.5) / (df + 0.5) + 1
            )
            
            # TF component with saturation and length normalization
            tf_component = (
                tf * (self.k1 + 1) /
                (tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length))
            )
            
            score += idf * tf_component
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.STOPWORDS]
        
        # Simple stemming (suffix removal) if enabled
        if self.use_stemming:
            tokens = [self._simple_stem(t) for t in tokens]
        
        return tokens
    
    def _simple_stem(self, word: str) -> str:
        """Simple suffix-based stemming.
        
        Args:
            word: Input word
            
        Returns:
            Stemmed word
        """
        suffixes = ['ing', 'ed', 'ly', 'es', 's', 'ment', 'tion', 'ness']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def __len__(self) -> int:
        return self._num_docs


class HybridSearcher:
    """Combines vector and keyword search.
    
    Implements reciprocal rank fusion for result merging.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: InvertedIndex,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60
    ):
        """Initialize hybrid searcher.
        
        Args:
            vector_index: Vector similarity index
            keyword_index: Keyword inverted index
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores
            rrf_k: RRF smoothing parameter
        """
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
    
    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID
    ) -> List[Tuple[str, float]]:
        """Perform hybrid search.
        
        Args:
            query_text: Query text for keyword search
            query_vector: Query embedding for vector search
            top_k: Number of results
            mode: Search mode
            
        Returns:
            List of (doc_id, score) tuples
        """
        if mode == SearchMode.VECTOR:
            return self.vector_index.search(query_vector, top_k)
        elif mode == SearchMode.KEYWORD:
            return self.keyword_index.search(query_text, top_k)
        elif mode == SearchMode.HYBRID:
            return self._hybrid_search(query_text, query_vector, top_k)
        elif mode == SearchMode.RERANK:
            return self._rerank_search(query_text, query_vector, top_k)
        else:
            return self.vector_index.search(query_vector, top_k)
    
    def _hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Combine vector and keyword search with RRF.
        
        Args:
            query_text: Query text
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            Merged results
        """
        # Get results from both indexes
        vector_results = self.vector_index.search(query_vector, top_k * 2)
        keyword_results = self.keyword_index.search(query_text, top_k * 2)
        
        # Compute RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        
        for rank, (doc_id, _) in enumerate(vector_results, 1):
            rrf_scores[doc_id] += self.vector_weight / (self.rrf_k + rank)
        
        for rank, (doc_id, _) in enumerate(keyword_results, 1):
            rrf_scores[doc_id] += self.keyword_weight / (self.rrf_k + rank)
        
        # Sort by RRF score
        results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _rerank_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Vector search with keyword reranking.
        
        Args:
            query_text: Query text
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            Reranked results
        """
        # Initial vector search
        vector_results = self.vector_index.search(query_vector, top_k * 3)
        
        if not vector_results:
            return []
        
        # Get keyword scores for candidates
        candidate_ids = [doc_id for doc_id, _ in vector_results]
        keyword_results = self.keyword_index.search(query_text, len(candidate_ids))
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}
        
        # Combine scores
        combined = []
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1.0
        
        for doc_id, vector_score in vector_results:
            keyword_score = keyword_scores.get(doc_id, 0) / max_keyword if max_keyword > 0 else 0
            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            combined.append((doc_id, combined_score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


class SemanticSearchEngine:
    """Main unified interface for semantic search.
    
    Provides document indexing, multi-modal search, and result ranking.
    
    Example:
        >>> engine = SemanticSearchEngine()
        >>> engine.index_document(Document(id="1", text="Hello world"))
        >>> results = engine.search("greeting")
    """
    
    def __init__(
        self,
        index_config: Optional[IndexConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """Initialize search engine.
        
        Args:
            index_config: Index configuration
            embedding_config: Embedding configuration
            vector_weight: Weight for vector similarity
            keyword_weight: Weight for keyword matching
        """
        self.index_config = index_config or IndexConfig()
        self.embedding_config = embedding_config or EmbeddingConfig(
            fallback_dim=self.index_config.embedding_dim
        )
        
        # Initialize components
        self._vector_index = VectorIndex(self.index_config)
        self._keyword_index = InvertedIndex()
        self._hybrid_searcher = HybridSearcher(
            self._vector_index,
            self._keyword_index,
            vector_weight,
            keyword_weight
        )
        
        # Document storage
        self._documents: Dict[str, Document] = {}
        
        # Encoder
        self._encoder = TransformerEncoder(config=self.embedding_config)
        self._embedding_cache = EmbeddingCache(
            max_size=self.index_config.max_cache_size
        )
        
        self._lock = threading.Lock()
    
    def index_document(self, document: Document) -> bool:
        """Index a single document.
        
        Args:
            document: Document to index
            
        Returns:
            True if indexed successfully
        """
        with self._lock:
            # Get or compute embedding
            if document.embedding is not None:
                embedding = document.embedding
            else:
                embedding = self._get_embedding(document.get_searchable_text())
            
            # Add to indexes
            self._vector_index.add(document.id, embedding)
            self._keyword_index.add(document.id, document.get_searchable_text())
            
            # Store document
            self._documents[document.id] = document
            
            return True
    
    def index_documents(self, documents: List[Document]) -> int:
        """Index multiple documents.
        
        Args:
            documents: Documents to index
            
        Returns:
            Number of documents indexed
        """
        count = 0
        for doc in documents:
            if self.index_document(doc):
                count += 1
        return count
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from index.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if removed
        """
        with self._lock:
            if doc_id not in self._documents:
                return False
            
            self._vector_index.remove(doc_id)
            self._keyword_index.remove(doc_id)
            del self._documents[doc_id]
            
            return True
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResults:
        """Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results
            mode: Search mode
            threshold: Minimum score threshold
            filters: Metadata filters
            
        Returns:
            SearchResults object
        """
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Perform search
        raw_results = self._hybrid_searcher.search(
            query, query_embedding, top_k * 2, mode
        )
        
        # Apply filters and threshold
        results = []
        for doc_id, score in raw_results:
            if score < threshold:
                continue
            
            doc = self._documents.get(doc_id)
            if doc is None:
                continue
            
            # Apply metadata filters
            if filters and not self._matches_filters(doc, filters):
                continue
            
            # Create search result
            highlights = self._generate_highlights(doc.text, query)
            result = SearchResult(
                document=doc,
                score=score,
                rank=len(results) + 1,
                highlights=highlights,
                match_type=mode.value
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=results,
            query=query,
            total_matches=len(raw_results),
            search_time_ms=search_time_ms,
            mode=mode
        )
    
    def search_similar(
        self,
        doc_id: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> SearchResults:
        """Find documents similar to a given document.
        
        Args:
            doc_id: Source document ID
            top_k: Number of results
            exclude_self: Whether to exclude the source document
            
        Returns:
            SearchResults object
        """
        start_time = time.time()
        
        doc = self._documents.get(doc_id)
        if doc is None:
            return SearchResults(
                results=[],
                query=f"similar_to:{doc_id}",
                total_matches=0,
                search_time_ms=0
            )
        
        # Get document embedding
        embedding = self._get_embedding(doc.get_searchable_text())
        
        # Search
        k = top_k + 1 if exclude_self else top_k
        raw_results = self._vector_index.search(embedding, k)
        
        results = []
        for similar_id, score in raw_results:
            if exclude_self and similar_id == doc_id:
                continue
            
            similar_doc = self._documents.get(similar_id)
            if similar_doc:
                result = SearchResult(
                    document=similar_doc,
                    score=score,
                    rank=len(results) + 1,
                    match_type="similar"
                )
                results.append(result)
            
            if len(results) >= top_k:
                break
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=results,
            query=f"similar_to:{doc_id}",
            total_matches=len(results),
            search_time_ms=search_time_ms,
            mode=SearchMode.VECTOR
        )
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document or None
        """
        return self._documents.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """Get all indexed documents.
        
        Returns:
            List of documents
        """
        return list(self._documents.values())
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()[:16]
        
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        
        embedding = self._encoder.encode_text(text)
        
        # Ensure correct dimension
        if len(embedding) != self.index_config.embedding_dim:
            # Pad or truncate
            if len(embedding) < self.index_config.embedding_dim:
                embedding = np.pad(
                    embedding,
                    (0, self.index_config.embedding_dim - len(embedding))
                )
            else:
                embedding = embedding[:self.index_config.embedding_dim]
        
        if self.index_config.cache_embeddings:
            self._embedding_cache.put(cache_key, embedding)
        
        return embedding
    
    def _matches_filters(
        self,
        doc: Document,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if document matches filters.
        
        Args:
            doc: Document to check
            filters: Filter criteria
            
        Returns:
            True if matches
        """
        for key, value in filters.items():
            doc_value = doc.metadata.get(key)
            
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            elif doc_value != value:
                return False
        
        return True
    
    def _generate_highlights(
        self,
        text: str,
        query: str,
        max_highlights: int = 3,
        context_words: int = 10
    ) -> List[str]:
        """Generate text highlights/snippets.
        
        Args:
            text: Document text
            query: Search query
            max_highlights: Maximum highlights
            context_words: Words of context around match
            
        Returns:
            List of highlight strings
        """
        query_terms = set(query.lower().split())
        words = text.split()
        highlights = []
        
        for i, word in enumerate(words):
            if word.lower().strip('.,!?;:') in query_terms:
                # Extract context
                start = max(0, i - context_words)
                end = min(len(words), i + context_words + 1)
                snippet = ' '.join(words[start:end])
                
                if start > 0:
                    snippet = '...' + snippet
                if end < len(words):
                    snippet = snippet + '...'
                
                highlights.append(snippet)
                
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    def __len__(self) -> int:
        return len(self._documents)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._documents


# Convenience functions

def create_search_engine(
    use_lsh: bool = False,
    embedding_dim: int = 256
) -> SemanticSearchEngine:
    """Create semantic search engine with configuration.
    
    Args:
        use_lsh: Whether to use LSH for approximate search
        embedding_dim: Embedding dimension
        
    Returns:
        SemanticSearchEngine instance
    """
    config = IndexConfig(
        embedding_dim=embedding_dim,
        use_lsh=use_lsh
    )
    return SemanticSearchEngine(index_config=config)


def quick_search(
    documents: List[str],
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Quick search over a list of text documents.
    
    Convenience function for simple search tasks.
    
    Args:
        documents: List of document texts
        query: Search query
        top_k: Number of results
        
    Returns:
        List of (document_text, score) tuples
    """
    engine = SemanticSearchEngine()
    
    # Index documents
    for i, text in enumerate(documents):
        doc = Document(id=str(i), text=text)
        engine.index_document(doc)
    
    # Search
    results = engine.search(query, top_k=top_k)
    
    return [(r.document.text, r.score) for r in results]
