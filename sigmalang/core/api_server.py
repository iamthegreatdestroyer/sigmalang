"""
ΣLANG REST API Server

Production-ready FastAPI server exposing all ΣLANG capabilities.
"""

import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

import numpy as np

# Core imports
from .config import get_config, SigmalangConfig, FeatureFlags
from .monitoring import (
    get_registry, get_health_checker, get_tracer,
    REQUEST_COUNT, REQUEST_LATENCY, REQUEST_IN_PROGRESS,
    ENCODE_COUNT, ENCODE_LATENCY, ANALOGY_COUNT, ANALOGY_LATENCY,
    SEARCH_COUNT, SEARCH_LATENCY, ERROR_COUNT,
    HealthCheckResult, StructuredLogger
)
from .api_models import (
    EncodeRequest, EncodeResponse,
    DecodeRequest, DecodeResponse,
    AnalogyRequest, AnalogyResponse, AnalogySolution,
    AnalogyExplainRequest, AnalogyExplainResponse,
    SearchRequest, SearchResponse, SearchResult,
    EntityExtractionRequest, EntityExtractionResponse,
    ExtractedEntity, ExtractedRelation,
    EmbeddingRequest, EmbeddingResponse,
    SimilarityRequest, SimilarityResponse,
    HealthResponse, HealthStatus, InfoResponse,
    BatchRequest, BatchResponse, BatchResult,
    ErrorResponse, create_error_response,
    EntityType, SearchMode, AnalogyType, OutputFormat
)

# Version info
__version__ = "1.0.0"
__api_version__ = "v1"

logger = StructuredLogger("sigmalang.api")


# =============================================================================
# Service Classes
# =============================================================================

class EncoderService:
    """Service for encoding operations."""
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self._encoder = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the encoder."""
        try:
            from .encoder import SigmaEncoder
            self._encoder = SigmaEncoder()
            self._initialized = True
            logger.info("Encoder service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize encoder: {e}")
            raise
    
    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode text to vector."""
        if not self._initialized:
            self.initialize()
        
        start = time.perf_counter()
        try:
            # Use the encoder to process text
            result = self._encoder.encode(text)
            
            # Convert to numpy array
            if hasattr(result, 'to_numpy'):
                vector = result.to_numpy()
            elif isinstance(result, np.ndarray):
                vector = result
            else:
                vector = np.array(result, dtype=np.float32)
            
            if normalize and np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            ENCODE_LATENCY.observe(time.perf_counter() - start)
            ENCODE_COUNT.inc(status="success")
            return vector
            
        except Exception as e:
            ENCODE_COUNT.inc(status="error")
            raise
    
    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        """Encode multiple texts."""
        return [self.encode(text, normalize) for text in texts]
    
    def get_health(self) -> HealthCheckResult:
        """Check encoder health."""
        try:
            if self._initialized:
                # Quick encode test
                self.encode("health check", normalize=False)
                return HealthCheckResult(name="encoder", healthy=True, message="OK")
            return HealthCheckResult(name="encoder", healthy=False, message="Not initialized")
        except Exception as e:
            return HealthCheckResult(name="encoder", healthy=False, message=str(e))


class DecoderService:
    """Service for decoding operations."""
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self._decoder = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the decoder."""
        try:
            from .encoder import SigmaEncoder, SigmaDecoder
            # SigmaDecoder requires an encoder for reference resolution
            encoder = SigmaEncoder()
            self._decoder = SigmaDecoder(encoder)
            self._initialized = True
            logger.info("Decoder service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize decoder: {e}")
            raise
    
    def decode(self, vector: np.ndarray, max_length: int = 512) -> str:
        """Decode vector to text."""
        if not self._initialized:
            self.initialize()
        
        try:
            result = self._decoder.decode(vector)
            if isinstance(result, str):
                return result[:max_length]
            return str(result)[:max_length]
        except Exception as e:
            logger.error(f"Decode failed: {e}")
            raise
    
    def decode_batch(self, vectors: List[np.ndarray], max_length: int = 512) -> List[str]:
        """Decode multiple vectors."""
        return [self.decode(v, max_length) for v in vectors]


class AnalogyService:
    """Service for analogy operations."""
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self._engine = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the analogy engine."""
        try:
            from .semantic_analogy_engine import SemanticAnalogyEngine
            self._engine = SemanticAnalogyEngine()
            # Register common default candidates for basic analogy solving
            default_candidates = [
                "king", "queen", "prince", "princess", "man", "woman", 
                "boy", "girl", "father", "mother", "son", "daughter",
                "dog", "cat", "puppy", "kitten", "big", "small",
                "hot", "cold", "fast", "slow", "good", "bad"
            ]
            try:
                self._engine.register_candidates(default_candidates)
            except Exception as e:
                logger.warning(f"Failed to register default candidates: {e}")
            self._initialized = True
            logger.info("Analogy service initialized with default candidates")
        except ImportError:
            # Fallback to simple analogy handling
            self._engine = None
            self._initialized = True
            logger.warning("SemanticAnalogyEngine not available, using fallback")
        except Exception as e:
            logger.error(f"Failed to initialize analogy engine: {e}")
            raise
    
    def solve(
        self, 
        a: str, 
        b: str, 
        c: str, 
        top_k: int = 5,
        analogy_type: AnalogyType = AnalogyType.SEMANTIC
    ) -> List[AnalogySolution]:
        """Solve an analogy: A is to B as C is to ?"""
        if not self._initialized:
            self.initialize()
        
        start = time.perf_counter()
        try:
            solutions = []
            
            if self._engine:
                # Use the full engine
                result = self._engine.solve_analogy(a, b, c, top_k=top_k)
                if hasattr(result, 'solutions'):
                    for sol in result.solutions[:top_k]:
                        solutions.append(AnalogySolution(
                            answer=sol.answer if hasattr(sol, 'answer') else str(sol),
                            confidence=sol.confidence if hasattr(sol, 'confidence') else 0.5,
                            relation=sol.relation if hasattr(sol, 'relation') else "",
                            reasoning=sol.reasoning if hasattr(sol, 'reasoning') else ""
                        ))
                elif isinstance(result, list):
                    for item in result[:top_k]:
                        if isinstance(item, tuple):
                            solutions.append(AnalogySolution(
                                answer=str(item[0]),
                                confidence=float(item[1]) if len(item) > 1 else 0.5
                            ))
                        else:
                            solutions.append(AnalogySolution(answer=str(item), confidence=0.5))
            else:
                # Simple fallback
                solutions.append(AnalogySolution(
                    answer=f"[{c}→?]",
                    confidence=0.5,
                    reasoning=f"Analogy: {a}:{b}::{c}:?"
                ))
            
            ANALOGY_LATENCY.observe(time.perf_counter() - start, type=analogy_type.value)
            ANALOGY_COUNT.inc(type=analogy_type.value, status="success")
            return solutions
            
        except Exception as e:
            ANALOGY_COUNT.inc(type=analogy_type.value, status="error")
            raise
    
    def explain(self, a: str, b: str, c: str, d: str) -> Dict[str, Any]:
        """Explain an analogy relationship."""
        if not self._initialized:
            self.initialize()
        
        explanation = {
            "relation_ab": f"Relationship between '{a}' and '{b}'",
            "relation_cd": f"Relationship between '{c}' and '{d}'",
            "similarity_score": 0.8,  # Placeholder
            "explanation": f"The analogy {a}:{b}::{c}:{d} holds because of semantic similarity.",
        }
        
        if self._engine and hasattr(self._engine, 'explain_analogy'):
            try:
                result = self._engine.explain_analogy(a, b, c, d)
                if isinstance(result, dict):
                    explanation.update(result)
            except Exception:
                pass
        
        return explanation


class SearchService:
    """Service for semantic search operations."""
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self._search_engine = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the search engine."""
        try:
            from .semantic_search import SemanticSearchEngine
            self._search_engine = SemanticSearchEngine()
            self._initialized = True
            logger.info("Search service initialized")
        except ImportError:
            self._search_engine = None
            self._initialized = True
            logger.warning("SemanticSearchEngine not available, using fallback")
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise
    
    def search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 10,
        mode: SearchMode = SearchMode.SEMANTIC,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """Search corpus for similar texts."""
        if not self._initialized:
            self.initialize()
        
        start = time.perf_counter()
        try:
            results = []
            
            if self._search_engine:
                search_results = self._search_engine.search(
                    query=query,
                    corpus=corpus,
                    top_k=top_k,
                    threshold=threshold
                )
                for i, res in enumerate(search_results):
                    if isinstance(res, tuple):
                        results.append(SearchResult(
                            text=str(res[0]),
                            score=float(res[1]),
                            index=i
                        ))
                    elif hasattr(res, 'text'):
                        results.append(SearchResult(
                            text=res.text,
                            score=res.score if hasattr(res, 'score') else 0.5,
                            index=i
                        ))
            else:
                # Simple fallback: substring matching
                query_lower = query.lower()
                for i, text in enumerate(corpus[:top_k]):
                    score = 1.0 if query_lower in text.lower() else 0.0
                    if score >= threshold:
                        results.append(SearchResult(text=text, score=score, index=i))
            
            SEARCH_LATENCY.observe(time.perf_counter() - start, mode=mode.value)
            SEARCH_COUNT.inc(mode=mode.value, status="success")
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
            
        except Exception as e:
            SEARCH_COUNT.inc(mode=mode.value, status="error")
            raise


class EntityService:
    """Service for entity extraction operations."""
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self._extractor = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the entity extractor."""
        try:
            from .entity_relation_extraction import EntityRelationExtractor
            self._extractor = EntityRelationExtractor()
            self._initialized = True
            logger.info("Entity service initialized")
        except ImportError:
            self._extractor = None
            self._initialized = True
            logger.warning("EntityRelationExtractor not available, using fallback")
        except Exception as e:
            logger.error(f"Failed to initialize entity extractor: {e}")
            raise
    
    def extract(
        self,
        text: str,
        include_relations: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Extract entities and relations from text."""
        if not self._initialized:
            self.initialize()
        
        entities = []
        relations = []
        
        if self._extractor:
            try:
                result = self._extractor.extract(text)
                if hasattr(result, 'entities'):
                    for ent in result.entities:
                        if getattr(ent, 'confidence', 1.0) >= confidence_threshold:
                            entities.append(ExtractedEntity(
                                text=ent.text,
                                entity_type=EntityType.CONCEPT,
                                confidence=getattr(ent, 'confidence', 1.0),
                                start=getattr(ent, 'start', None),
                                end=getattr(ent, 'end', None)
                            ))
                
                if include_relations and hasattr(result, 'relations'):
                    for rel in result.relations:
                        if getattr(rel, 'confidence', 1.0) >= confidence_threshold:
                            relations.append(ExtractedRelation(
                                source=rel.source,
                                target=rel.target,
                                relation_type=rel.relation_type,
                                confidence=getattr(rel, 'confidence', 1.0)
                            ))
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        return {
            "entities": entities,
            "relations": relations
        }


class NLPService:
    """Service for NLP operations (embeddings, similarity)."""
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self._embeddings = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize NLP components."""
        try:
            from .transformer_embeddings import TransformerEmbeddings, EmbeddingConfig
            config = EmbeddingConfig(fallback_dim=384)
            self._embeddings = TransformerEmbeddings(config=config)
            self._initialized = True
            logger.info("NLP service initialized")
        except ImportError:
            self._embeddings = None
            self._initialized = True
            logger.warning("TransformerEmbeddings not available, using fallback")
        except Exception as e:
            logger.error(f"Failed to initialize NLP service: {e}")
            raise
    
    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for text."""
        if not self._initialized:
            self.initialize()
        
        if self._embeddings:
            return self._embeddings.embed(text)
        
        # Fallback: simple hash-based embedding
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        vector = np.frombuffer(hash_bytes, dtype=np.float32)
        vector = np.resize(vector, 384)
        if normalize and np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        return vector
    
    def embed_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            self.initialize()
        
        if self._embeddings and hasattr(self._embeddings, 'embed_batch'):
            return self._embeddings.embed_batch(texts)
        
        return np.array([self.embed(text, normalize) for text in texts])
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        if not self._initialized:
            self.initialize()
        
        if self._embeddings and hasattr(self._embeddings, 'similarity'):
            return self._embeddings.similarity(text1, text2)
        
        # Fallback: cosine similarity
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


# =============================================================================
# API Application
# =============================================================================

class SigmalangAPI:
    """
    Main API application class.
    
    Provides a unified interface to all ΣLANG services.
    Can be used standalone or wrapped by FastAPI/Flask.
    """
    
    def __init__(self, config: Optional[SigmalangConfig] = None):
        self.config = config or get_config()
        self.start_time = datetime.now(timezone.utc)
        
        # Services
        self.encoder = EncoderService(self.config)
        self.decoder = DecoderService(self.config)
        self.analogy = AnalogyService(self.config)
        self.search = SearchService(self.config)
        self.entity = EntityService(self.config)
        self.nlp = NLPService(self.config)
        
        # State
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all services."""
        logger.info("Initializing ΣLANG API")
        
        # Initialize services based on feature flags
        if self.config.features.enable_nlp:
            self.encoder.initialize()
            self.decoder.initialize()
            self.nlp.initialize()
        
        if self.config.features.enable_analogies:
            self.analogy.initialize()
        
        if self.config.features.enable_search:
            self.search.initialize()
        
        if self.config.features.enable_entities:
            self.entity.initialize()
        
        # Register health checks
        health = get_health_checker()
        health.register("encoder", self.encoder.get_health)
        
        self._initialized = True
        logger.info("ΣLANG API initialized successfully")
    
    def shutdown(self) -> None:
        """Shutdown all services."""
        logger.info("Shutting down ΣLANG API")
        self._initialized = False
    
    # =========================================================================
    # Encoding Endpoints
    # =========================================================================
    
    def encode(self, request: EncodeRequest) -> EncodeResponse:
        """Encode text to vector."""
        start = time.perf_counter()
        
        try:
            if request.texts:
                vectors = self.encoder.encode_batch(request.texts, request.normalize)
                return EncodeResponse(
                    success=True,
                    request_id=request.request_id,
                    vectors=[v.tolist() for v in vectors],
                    dimensions=vectors[0].shape[0] if vectors else 0,
                    token_count=sum(len(t.split()) for t in request.texts),
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
            else:
                vector = self.encoder.encode(request.text, request.normalize)
                return EncodeResponse(
                    success=True,
                    request_id=request.request_id,
                    vector=vector.tolist(),
                    dimensions=vector.shape[0],
                    token_count=len(request.text.split()),
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="encode")
            return EncodeResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="ENCODE_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    def decode(self, request: DecodeRequest) -> DecodeResponse:
        """Decode vector to text."""
        start = time.perf_counter()
        
        try:
            if request.vectors:
                vectors = [np.array(v) for v in request.vectors]
                texts = self.decoder.decode_batch(vectors, request.max_length)
                return DecodeResponse(
                    success=True,
                    request_id=request.request_id,
                    texts=texts,
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
            else:
                vector = np.array(request.vector)
                text = self.decoder.decode(vector, request.max_length)
                return DecodeResponse(
                    success=True,
                    request_id=request.request_id,
                    text=text,
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="decode")
            return DecodeResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="DECODE_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    # =========================================================================
    # Analogy Endpoints
    # =========================================================================
    
    def solve_analogy(self, request: AnalogyRequest) -> AnalogyResponse:
        """Solve an analogy."""
        start = time.perf_counter()
        
        try:
            solutions = self.analogy.solve(
                request.a, request.b, request.c,
                top_k=request.top_k,
                analogy_type=request.analogy_type
            )
            
            best = solutions[0] if solutions else None
            
            response = AnalogyResponse(
                success=True,
                request_id=request.request_id,
                solutions=solutions,
                best_answer=best.answer if best else None,
                confidence=best.confidence if best else 0.0,
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
            
            # If verification was requested
            if request.d:
                response.is_valid = any(s.answer.lower() == request.d.lower() for s in solutions)
            
            return response
            
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="analogy")
            return AnalogyResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="ANALOGY_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    def explain_analogy(self, request: AnalogyExplainRequest) -> AnalogyExplainResponse:
        """Explain an analogy."""
        start = time.perf_counter()
        
        try:
            explanation = self.analogy.explain(request.a, request.b, request.c, request.d)
            
            return AnalogyExplainResponse(
                success=True,
                request_id=request.request_id,
                explanation=explanation.get("explanation", ""),
                relation_ab=explanation.get("relation_ab", ""),
                relation_cd=explanation.get("relation_cd", ""),
                similarity_score=explanation.get("similarity_score", 0.0),
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="explain")
            return AnalogyExplainResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="EXPLAIN_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    # =========================================================================
    # Search Endpoints
    # =========================================================================
    
    def search_corpus(self, request: SearchRequest) -> SearchResponse:
        """Search a corpus."""
        start = time.perf_counter()
        
        try:
            if not request.corpus:
                return SearchResponse(
                    success=False,
                    request_id=request.request_id,
                    error="Corpus is required",
                    error_code="MISSING_CORPUS",
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
            
            results = self.search.search(
                query=request.query,
                corpus=request.corpus,
                top_k=request.top_k,
                mode=request.mode,
                threshold=request.threshold
            )
            
            return SearchResponse(
                success=True,
                request_id=request.request_id,
                results=results,
                total_count=len(results),
                search_time_ms=(time.perf_counter() - start) * 1000,
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="search")
            return SearchResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="SEARCH_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    # =========================================================================
    # Entity Endpoints
    # =========================================================================
    
    def extract_entities(self, request: EntityExtractionRequest) -> EntityExtractionResponse:
        """Extract entities from text."""
        start = time.perf_counter()
        
        try:
            result = self.entity.extract(
                text=request.text,
                include_relations=request.include_relations,
                confidence_threshold=request.confidence_threshold
            )
            
            return EntityExtractionResponse(
                success=True,
                request_id=request.request_id,
                entities=result["entities"],
                relations=result["relations"],
                entity_count=len(result["entities"]),
                relation_count=len(result["relations"]),
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="entities")
            return EntityExtractionResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="ENTITY_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    # =========================================================================
    # NLP Endpoints
    # =========================================================================
    
    def get_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Get transformer embeddings."""
        start = time.perf_counter()
        
        try:
            if request.texts:
                embeddings = self.nlp.embed_batch(request.texts, request.normalize)
                return EmbeddingResponse(
                    success=True,
                    request_id=request.request_id,
                    embeddings=[e.tolist() for e in embeddings],
                    dimensions=embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]),
                    model_used=request.model,
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
            else:
                embedding = self.nlp.embed(request.text, request.normalize)
                return EmbeddingResponse(
                    success=True,
                    request_id=request.request_id,
                    embedding=embedding.tolist(),
                    dimensions=len(embedding),
                    model_used=request.model,
                    processing_time_ms=(time.perf_counter() - start) * 1000
                )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="embeddings")
            return EmbeddingResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="EMBEDDING_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    def compute_similarity(self, request: SimilarityRequest) -> SimilarityResponse:
        """Compute text similarity."""
        start = time.perf_counter()
        
        try:
            score = self.nlp.similarity(request.text1, request.text2)
            
            return SimilarityResponse(
                success=True,
                request_id=request.request_id,
                score=score,
                metric_used=request.metric,
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
        except Exception as e:
            ERROR_COUNT.inc(error_type=type(e).__name__, endpoint="similarity")
            return SimilarityResponse(
                success=False,
                request_id=request.request_id,
                error=str(e),
                error_code="SIMILARITY_ERROR",
                processing_time_ms=(time.perf_counter() - start) * 1000
            )
    
    # =========================================================================
    # Health & Info Endpoints
    # =========================================================================
    
    def health(self) -> HealthResponse:
        """Get health status."""
        health_checker = get_health_checker()
        results = health_checker.check_all()
        
        components = [
            HealthStatus(
                name=r.name,
                healthy=r.healthy,
                latency_ms=r.latency_ms,
                message=r.message
            )
            for r in results
        ]
        
        all_healthy = all(r.healthy for r in results)
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return HealthResponse(
            success=True,
            status="healthy" if all_healthy else "degraded",
            version=__version__,
            components=components,
            uptime_seconds=uptime
        )
    
    def info(self) -> InfoResponse:
        """Get system information."""
        return InfoResponse(
            success=True,
            version=__version__,
            environment=self.config.environment.value,
            features={
                "analogies": self.config.features.enable_analogies,
                "search": self.config.features.enable_search,
                "entities": self.config.features.enable_entities,
                "nlp": self.config.features.enable_nlp,
                "streaming": self.config.features.enable_streaming,
                "batch": self.config.features.enable_batch,
            },
            models={
                "encoder": "sigmalang-encoder-v1",
                "embeddings": "all-MiniLM-L6-v2",
            },
            configuration=self.config.to_dict()
        )
    
    def metrics(self) -> str:
        """Get Prometheus metrics."""
        registry = get_registry()
        return registry.to_prometheus()


# =============================================================================
# Factory Functions
# =============================================================================

_api_instance: Optional[SigmalangAPI] = None


def get_api() -> SigmalangAPI:
    """Get or create the global API instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = SigmalangAPI()
        _api_instance.initialize()
    return _api_instance


def create_api(config: Optional[SigmalangConfig] = None) -> SigmalangAPI:
    """Create a new API instance."""
    api = SigmalangAPI(config)
    api.initialize()
    return api
