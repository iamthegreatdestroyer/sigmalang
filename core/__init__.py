"""ΣLANG Core Module - Primitives, Parser, Encoder, and API Services"""

from .primitives import (
    SemanticNode,
    SemanticTree,
    Glyph,
    GlyphStream,
    GlyphType,
    PrimitiveTier,
    ExistentialPrimitive,
    CodePrimitive,
    MathPrimitive,
    LogicPrimitive,
    EntityPrimitive,
    ActionPrimitive,
    CommunicationPrimitive,
    StructurePrimitive,
    LearnedPrimitive,
    PRIMITIVE_REGISTRY,
    LEARNED_PRIMITIVE_START,
    LEARNED_PRIMITIVE_END
)

from .parser import (
    SemanticParser,
    SemanticTreePrinter,
    IntentType,
    ParsedEntity,
    ParsedRelation
)

from .encoder import (
    SigmaEncoder,
    SigmaDecoder,
    SigmaHashBank,
    ContextStack,
    ContextDelta,
    LSHIndex,
    LRUCache
)

# Phase 3: Configuration Management
from .config import (
    Environment,
    ServerConfig,
    RateLimitConfig,
    AuthConfig,
    CacheConfig,
    EncoderConfig,
    MonitoringConfig,
    FeatureFlags,
    SecretsManager,
    SigmalangConfig,
    get_config,
    reset_config,
    configure_from_file,
    configure_from_env
)

# Phase 3: API Data Models
from .api_models import (
    # Enums
    AnalogyType,
    SearchMode,
    EntityType,
    OutputFormat,
    # Base Classes
    BaseRequest,
    BaseResponse,
    # Core Request/Response Models
    EncodeRequest,
    EncodeResponse,
    DecodeRequest,
    DecodeResponse,
    AnalogyRequest,
    AnalogySolution,
    AnalogyResponse,
    AnalogyExplainRequest,
    AnalogyExplainResponse,
    SearchRequest,
    SearchResult,
    SearchResponse,
    # Entity Extraction
    EntityExtractionRequest,
    ExtractedEntity,
    ExtractedRelation,
    EntityExtractionResponse,
    # Embeddings & Similarity
    EmbeddingRequest,
    EmbeddingResponse,
    SimilarityRequest,
    SimilarityResponse,
    # Translation
    TranslationRequest,
    TranslationResponse,
    # Health & Info
    HealthStatus,
    HealthResponse,
    InfoResponse,
    # Batch Processing
    BatchRequest,
    BatchResult,
    BatchResponse,
    # Errors
    ErrorDetail,
    ErrorResponse,
    # Streaming
    StreamChunk,
    StreamRequest,
    # Utilities
    create_error_response,
    validate_request
)

# Phase 3: Monitoring and Observability
from .monitoring import (
    MetricsRegistry,
    HealthChecker,
    StructuredLogger,
    Tracer,
    timed,
    counted,
    traced,
    get_registry,
    get_health_checker,
    get_tracer,
    configure_logging,
    MetricType,
    MetricValue,
    Counter,
    Gauge,
    Histogram,
    HealthCheckResult,
    Span
)

# Phase 3: API Server
from .api_server import (
    EncoderService,
    DecoderService,
    AnalogyService,
    SearchService,
    EntityService,
    NLPService,
    SigmalangAPI,
    get_api,
    create_api
)
