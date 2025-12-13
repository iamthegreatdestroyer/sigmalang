"""
ΣLANG API Data Models

Pydantic models for API request/response validation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import hashlib


# =============================================================================
# Enums
# =============================================================================

class AnalogyType(str, Enum):
    """Types of analogies."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    PROPORTIONAL = "proportional"
    CAUSAL = "causal"
    FUNCTIONAL = "functional"


class SearchMode(str, Enum):
    """Search modes."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    FUZZY = "fuzzy"


class EntityType(str, Enum):
    """Entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    CONCEPT = "concept"
    TECHNICAL = "technical"
    CUSTOM = "custom"


class OutputFormat(str, Enum):
    """Output format options."""
    JSON = "json"
    COMPACT = "compact"
    BINARY = "binary"
    BASE64 = "base64"


# =============================================================================
# Base Models
# =============================================================================

@dataclass
class BaseRequest:
    """Base class for all API requests."""
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.request_id is None:
            self.request_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique request ID."""
        content = f"{self.timestamp.isoformat()}-{id(self)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


@dataclass
class BaseResponse:
    """Base class for all API responses."""
    success: bool = True
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=_utcnow)
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None


# =============================================================================
# Encoding Models
# =============================================================================

@dataclass
class EncodeRequest(BaseRequest):
    """Request to encode text to ΣLANG vectors."""
    text: str = ""
    texts: Optional[List[str]] = None
    output_format: OutputFormat = OutputFormat.JSON
    include_metadata: bool = False
    normalize: bool = True
    dimensions: Optional[int] = None


@dataclass
class EncodeResponse(BaseResponse):
    """Response containing encoded vectors."""
    vector: Optional[List[float]] = None
    vectors: Optional[List[List[float]]] = None
    dimensions: int = 0
    metadata: Optional[Dict[str, Any]] = None
    token_count: int = 0


@dataclass
class DecodeRequest(BaseRequest):
    """Request to decode vectors back to text."""
    vector: Optional[List[float]] = None
    vectors: Optional[List[List[float]]] = None
    input_format: OutputFormat = OutputFormat.JSON
    max_length: int = 512
    temperature: float = 1.0


@dataclass
class DecodeResponse(BaseResponse):
    """Response containing decoded text."""
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    confidence: float = 0.0
    alternatives: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Analogy Models
# =============================================================================

@dataclass
class AnalogyRequest(BaseRequest):
    """Request to solve an analogy."""
    a: str = ""  # First term
    b: str = ""  # Second term (related to a)
    c: str = ""  # Third term (find d such that a:b::c:d)
    d: Optional[str] = None  # Optional: verify if d is correct
    analogy_type: AnalogyType = AnalogyType.SEMANTIC
    top_k: int = 5
    include_explanation: bool = False
    threshold: float = 0.0


@dataclass
class AnalogySolution:
    """A single analogy solution."""
    answer: str
    confidence: float
    relation: str = ""
    reasoning: str = ""


@dataclass
class AnalogyResponse(BaseResponse):
    """Response containing analogy solutions."""
    solutions: List[AnalogySolution] = field(default_factory=list)
    best_answer: Optional[str] = None
    confidence: float = 0.0
    explanation: Optional[str] = None
    relation_type: Optional[str] = None
    is_valid: Optional[bool] = None  # If d was provided for verification


@dataclass
class AnalogyExplainRequest(BaseRequest):
    """Request to explain an analogy relationship."""
    a: str = ""
    b: str = ""
    c: str = ""
    d: str = ""
    depth: int = 2  # Explanation depth level


@dataclass
class AnalogyExplainResponse(BaseResponse):
    """Response containing analogy explanation."""
    explanation: str = ""
    relation_ab: str = ""
    relation_cd: str = ""
    similarity_score: float = 0.0
    structural_analysis: Optional[Dict[str, Any]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None


# =============================================================================
# Search Models
# =============================================================================

@dataclass
class SearchRequest(BaseRequest):
    """Request for semantic search."""
    query: str = ""
    corpus: Optional[List[str]] = None
    corpus_id: Optional[str] = None
    mode: SearchMode = SearchMode.SEMANTIC
    top_k: int = 10
    threshold: float = 0.0
    include_scores: bool = True
    include_highlights: bool = False
    filters: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    score: float
    index: int = 0
    metadata: Optional[Dict[str, Any]] = None
    highlights: Optional[List[str]] = None


@dataclass
class SearchResponse(BaseResponse):
    """Response containing search results."""
    results: List[SearchResult] = field(default_factory=list)
    total_count: int = 0
    query_vector: Optional[List[float]] = None
    search_time_ms: float = 0.0


# =============================================================================
# Entity Extraction Models
# =============================================================================

@dataclass
class EntityExtractionRequest(BaseRequest):
    """Request to extract entities from text."""
    text: str = ""
    texts: Optional[List[str]] = None
    entity_types: Optional[List[EntityType]] = None
    include_relations: bool = True
    include_positions: bool = True
    confidence_threshold: float = 0.5


@dataclass
class ExtractedEntity:
    """A single extracted entity."""
    text: str
    entity_type: EntityType
    confidence: float
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExtractedRelation:
    """A relation between entities."""
    source: str
    target: str
    relation_type: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EntityExtractionResponse(BaseResponse):
    """Response containing extracted entities and relations."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    entity_count: int = 0
    relation_count: int = 0


# =============================================================================
# NLP Models
# =============================================================================

@dataclass
class EmbeddingRequest(BaseRequest):
    """Request for transformer embeddings."""
    text: str = ""
    texts: Optional[List[str]] = None
    model: str = "default"
    normalize: bool = True
    pooling: str = "mean"


@dataclass
class EmbeddingResponse(BaseResponse):
    """Response containing embeddings."""
    embedding: Optional[List[float]] = None
    embeddings: Optional[List[List[float]]] = None
    dimensions: int = 0
    model_used: str = ""


@dataclass
class SimilarityRequest(BaseRequest):
    """Request to compute text similarity."""
    text1: str = ""
    text2: str = ""
    texts1: Optional[List[str]] = None
    texts2: Optional[List[str]] = None
    metric: str = "cosine"


@dataclass
class SimilarityResponse(BaseResponse):
    """Response containing similarity scores."""
    score: Optional[float] = None
    scores: Optional[List[List[float]]] = None  # Matrix for batch comparison
    metric_used: str = ""


@dataclass
class TranslationRequest(BaseRequest):
    """Request for cross-lingual encoding."""
    text: str = ""
    source_language: str = "auto"
    target_language: str = "en"
    preserve_semantics: bool = True


@dataclass
class TranslationResponse(BaseResponse):
    """Response containing translation."""
    translated_text: str = ""
    source_language_detected: str = ""
    confidence: float = 0.0
    vector: Optional[List[float]] = None


# =============================================================================
# Health & Info Models
# =============================================================================

@dataclass
class HealthStatus:
    """Health check result for a component."""
    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""


@dataclass
class HealthResponse(BaseResponse):
    """Response for health check endpoint."""
    status: str = "healthy"
    version: str = "1.0.0"
    components: List[HealthStatus] = field(default_factory=list)
    uptime_seconds: float = 0.0


@dataclass
class InfoResponse(BaseResponse):
    """Response for system info endpoint."""
    version: str = "1.0.0"
    environment: str = "development"
    features: Dict[str, bool] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Batch Processing Models
# =============================================================================

@dataclass
class BatchRequest(BaseRequest):
    """Request for batch processing."""
    operations: List[Dict[str, Any]] = field(default_factory=list)
    parallel: bool = True
    fail_fast: bool = False
    timeout_per_operation: float = 30.0


@dataclass
class BatchResult:
    """Result for a single batch operation."""
    index: int
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class BatchResponse(BaseResponse):
    """Response for batch processing."""
    results: List[BatchResult] = field(default_factory=list)
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0


# =============================================================================
# Error Models
# =============================================================================

@dataclass
class ErrorDetail:
    """Detailed error information."""
    field: Optional[str] = None
    message: str = ""
    code: str = ""


@dataclass
class ErrorResponse(BaseResponse):
    """Standard error response."""
    success: bool = False
    error: str = ""
    error_code: str = ""
    details: List[ErrorDetail] = field(default_factory=list)
    trace_id: Optional[str] = None


# =============================================================================
# Streaming Models
# =============================================================================

@dataclass
class StreamChunk:
    """A chunk in a streaming response."""
    index: int
    data: Any
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StreamRequest(BaseRequest):
    """Request for streaming operations."""
    operation: str = ""
    data: Any = None
    chunk_size: int = 100
    buffer_size: int = 1000


# =============================================================================
# Utility Functions
# =============================================================================

def create_error_response(
    error: str,
    error_code: str = "UNKNOWN_ERROR",
    request_id: Optional[str] = None,
    details: Optional[List[ErrorDetail]] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        success=False,
        error=error,
        error_code=error_code,
        request_id=request_id,
        details=details or [],
        timestamp=datetime.now(timezone.utc)
    )


def validate_request(request: BaseRequest) -> List[ErrorDetail]:
    """Validate a request and return list of errors."""
    errors = []
    # Subclasses can override to add validation
    return errors
