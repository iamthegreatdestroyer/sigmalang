"""ΣLANG Core Module - Primitives, Parser, Encoder, and API Services"""

# Phase 3: API Data Models
from .api_models import (
    AnalogyExplainRequest,
    AnalogyExplainResponse,
    AnalogyRequest,
    AnalogyResponse,
    AnalogySolution,
    # Enums
    AnalogyType,
    # Base Classes
    BaseRequest,
    BaseResponse,
    # Batch Processing
    BatchRequest,
    BatchResponse,
    BatchResult,
    DecodeRequest,
    DecodeResponse,
    # Embeddings & Similarity
    EmbeddingRequest,
    EmbeddingResponse,
    # Core Request/Response Models
    EncodeRequest,
    EncodeResponse,
    # Entity Extraction
    EntityExtractionRequest,
    EntityExtractionResponse,
    EntityType,
    # Errors
    ErrorDetail,
    ErrorResponse,
    ExtractedEntity,
    ExtractedRelation,
    HealthResponse,
    # Health & Info
    HealthStatus,
    InfoResponse,
    OutputFormat,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarityRequest,
    SimilarityResponse,
    # Streaming
    StreamChunk,
    StreamRequest,
    # Translation
    TranslationRequest,
    TranslationResponse,
    # Utilities
    create_error_response,
    validate_request,
)

# Phase 3: API Server
from .api_server import (
    AnalogyService,
    DecoderService,
    EncoderService,
    EntityService,
    NLPService,
    SearchService,
    SigmalangAPI,
    create_api,
    get_api,
)

# Phase 3: Configuration Management
from .config import (
    AuthConfig,
    CacheConfig,
    EncoderConfig,
    Environment,
    FeatureFlags,
    MonitoringConfig,
    RateLimitConfig,
    SecretsManager,
    ServerConfig,
    SigmalangConfig,
    configure_from_env,
    configure_from_file,
    get_config,
    reset_config,
)
from .encoder import ContextDelta, ContextStack, LRUCache, LSHIndex, SigmaDecoder, SigmaEncoder, SigmaHashBank

# Phase 3: Monitoring and Observability
from .monitoring import (
    Counter,
    Gauge,
    HealthChecker,
    HealthCheckResult,
    Histogram,
    MetricsRegistry,
    MetricType,
    MetricValue,
    Span,
    StructuredLogger,
    Tracer,
    configure_logging,
    counted,
    get_health_checker,
    get_registry,
    get_tracer,
    timed,
    traced,
)
from .parser import IntentType, ParsedEntity, ParsedRelation, SemanticParser, SemanticTreePrinter
from .primitives import (
    LEARNED_PRIMITIVE_END,
    LEARNED_PRIMITIVE_START,
    PRIMITIVE_REGISTRY,
    ActionPrimitive,
    CodePrimitive,
    CommunicationPrimitive,
    EntityPrimitive,
    ExistentialPrimitive,
    Glyph,
    GlyphStream,
    GlyphType,
    LearnedPrimitive,
    LogicPrimitive,
    MathPrimitive,
    PrimitiveTier,
    SemanticNode,
    SemanticTree,
    StructurePrimitive,
)
