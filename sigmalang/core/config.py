"""
ΣLANG Configuration Management

Environment-based configuration with validation, secrets management,
and feature flags for production deployment.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum
from pathlib import Path
import hashlib
import base64
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Enum
# =============================================================================

class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, value: str) -> 'Environment':
        """Parse environment from string."""
        value_lower = value.lower()
        for env in cls:
            if env.value == value_lower:
                return env
        return cls.DEVELOPMENT


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: float = 30.0
    keepalive_timeout: int = 65
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Create from environment variables."""
        return cls(
            host=os.getenv("SIGMALANG_HOST", "0.0.0.0"),
            port=int(os.getenv("SIGMALANG_PORT", "8000")),
            workers=int(os.getenv("SIGMALANG_WORKERS", "4")),
            reload=os.getenv("SIGMALANG_RELOAD", "false").lower() == "true",
            debug=os.getenv("SIGMALANG_DEBUG", "false").lower() == "true",
            log_level=os.getenv("SIGMALANG_LOG_LEVEL", "INFO"),
            cors_origins=os.getenv("SIGMALANG_CORS_ORIGINS", "*").split(","),
            max_request_size=int(os.getenv("SIGMALANG_MAX_REQUEST_SIZE", str(10 * 1024 * 1024))),
            request_timeout=float(os.getenv("SIGMALANG_REQUEST_TIMEOUT", "30.0")),
            keepalive_timeout=int(os.getenv("SIGMALANG_KEEPALIVE_TIMEOUT", "65")),
        )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10
    by_ip: bool = True
    by_api_key: bool = True
    whitelist_ips: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> 'RateLimitConfig':
        """Create from environment variables."""
        whitelist = os.getenv("SIGMALANG_RATE_LIMIT_WHITELIST", "")
        return cls(
            enabled=os.getenv("SIGMALANG_RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("SIGMALANG_RATE_LIMIT_RPM", "60")),
            burst_size=int(os.getenv("SIGMALANG_RATE_LIMIT_BURST", "10")),
            by_ip=os.getenv("SIGMALANG_RATE_LIMIT_BY_IP", "true").lower() == "true",
            by_api_key=os.getenv("SIGMALANG_RATE_LIMIT_BY_KEY", "true").lower() == "true",
            whitelist_ips=whitelist.split(",") if whitelist else [],
        )


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enabled: bool = False
    api_key_header: str = "X-API-Key"
    jwt_enabled: bool = False
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    api_keys: Set[str] = field(default_factory=set)
    
    @classmethod
    def from_env(cls) -> 'AuthConfig':
        """Create from environment variables."""
        api_keys_str = os.getenv("SIGMALANG_API_KEYS", "")
        api_keys = set(k.strip() for k in api_keys_str.split(",") if k.strip())
        
        return cls(
            enabled=os.getenv("SIGMALANG_AUTH_ENABLED", "false").lower() == "true",
            api_key_header=os.getenv("SIGMALANG_API_KEY_HEADER", "X-API-Key"),
            jwt_enabled=os.getenv("SIGMALANG_JWT_ENABLED", "false").lower() == "true",
            jwt_secret=os.getenv("SIGMALANG_JWT_SECRET", ""),
            jwt_algorithm=os.getenv("SIGMALANG_JWT_ALGORITHM", "HS256"),
            jwt_expiry_hours=int(os.getenv("SIGMALANG_JWT_EXPIRY_HOURS", "24")),
            api_keys=api_keys,
        )
    
    def validate_api_key(self, key: str) -> bool:
        """Validate an API key."""
        if not self.enabled:
            return True
        return key in self.api_keys


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    backend: str = "memory"  # memory, redis
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600  # 1 hour
    max_size: int = 10000
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create from environment variables."""
        return cls(
            enabled=os.getenv("SIGMALANG_CACHE_ENABLED", "true").lower() == "true",
            backend=os.getenv("SIGMALANG_CACHE_BACKEND", "memory"),
            redis_url=os.getenv("SIGMALANG_REDIS_URL", "redis://localhost:6379/0"),
            default_ttl=int(os.getenv("SIGMALANG_CACHE_TTL", "3600")),
            max_size=int(os.getenv("SIGMALANG_CACHE_MAX_SIZE", "10000")),
        )


@dataclass
class EncoderConfig:
    """Encoder configuration."""
    dimensions: int = 512
    hash_functions: int = 8
    cache_size: int = 1000
    enable_parallel: bool = True
    max_workers: int = 4
    batch_size: int = 100
    
    @classmethod
    def from_env(cls) -> 'EncoderConfig':
        """Create from environment variables."""
        return cls(
            dimensions=int(os.getenv("SIGMALANG_ENCODER_DIMENSIONS", "512")),
            hash_functions=int(os.getenv("SIGMALANG_ENCODER_HASH_FUNCTIONS", "8")),
            cache_size=int(os.getenv("SIGMALANG_ENCODER_CACHE_SIZE", "1000")),
            enable_parallel=os.getenv("SIGMALANG_ENCODER_PARALLEL", "true").lower() == "true",
            max_workers=int(os.getenv("SIGMALANG_ENCODER_MAX_WORKERS", "4")),
            batch_size=int(os.getenv("SIGMALANG_ENCODER_BATCH_SIZE", "100")),
        )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    tracing_enabled: bool = False
    tracing_endpoint: str = "http://localhost:4317"
    tracing_sample_rate: float = 0.1
    log_format: str = "json"  # json, text
    log_requests: bool = True
    log_responses: bool = False
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """Create from environment variables."""
        return cls(
            metrics_enabled=os.getenv("SIGMALANG_METRICS_ENABLED", "true").lower() == "true",
            metrics_port=int(os.getenv("SIGMALANG_METRICS_PORT", "9090")),
            health_check_enabled=os.getenv("SIGMALANG_HEALTH_CHECK_ENABLED", "true").lower() == "true",
            tracing_enabled=os.getenv("SIGMALANG_TRACING_ENABLED", "false").lower() == "true",
            tracing_endpoint=os.getenv("SIGMALANG_TRACING_ENDPOINT", "http://localhost:4317"),
            tracing_sample_rate=float(os.getenv("SIGMALANG_TRACING_SAMPLE_RATE", "0.1")),
            log_format=os.getenv("SIGMALANG_LOG_FORMAT", "json"),
            log_requests=os.getenv("SIGMALANG_LOG_REQUESTS", "true").lower() == "true",
            log_responses=os.getenv("SIGMALANG_LOG_RESPONSES", "false").lower() == "true",
        )


@dataclass
class FeatureFlags:
    """Feature flags for controlling functionality."""
    enable_analogies: bool = True
    enable_search: bool = True
    enable_entities: bool = True
    enable_nlp: bool = True
    enable_multimodal: bool = True
    enable_streaming: bool = True
    enable_batch: bool = True
    enable_admin: bool = False
    experimental_features: Set[str] = field(default_factory=set)
    
    @classmethod
    def from_env(cls) -> 'FeatureFlags':
        """Create from environment variables."""
        experimental = os.getenv("SIGMALANG_EXPERIMENTAL_FEATURES", "")
        return cls(
            enable_analogies=os.getenv("SIGMALANG_FEATURE_ANALOGIES", "true").lower() == "true",
            enable_search=os.getenv("SIGMALANG_FEATURE_SEARCH", "true").lower() == "true",
            enable_entities=os.getenv("SIGMALANG_FEATURE_ENTITIES", "true").lower() == "true",
            enable_nlp=os.getenv("SIGMALANG_FEATURE_NLP", "true").lower() == "true",
            enable_multimodal=os.getenv("SIGMALANG_FEATURE_MULTIMODAL", "true").lower() == "true",
            enable_streaming=os.getenv("SIGMALANG_FEATURE_STREAMING", "true").lower() == "true",
            enable_batch=os.getenv("SIGMALANG_FEATURE_BATCH", "true").lower() == "true",
            enable_admin=os.getenv("SIGMALANG_FEATURE_ADMIN", "false").lower() == "true",
            experimental_features=set(f.strip() for f in experimental.split(",") if f.strip()),
        )
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_map = {
            "analogies": self.enable_analogies,
            "search": self.enable_search,
            "entities": self.enable_entities,
            "nlp": self.enable_nlp,
            "multimodal": self.enable_multimodal,
            "streaming": self.enable_streaming,
            "batch": self.enable_batch,
            "admin": self.enable_admin,
        }
        if feature in feature_map:
            return feature_map[feature]
        return feature in self.experimental_features


# =============================================================================
# Secrets Management
# =============================================================================

class SecretsManager:
    """
    Secure secrets management with encryption support.
    Supports environment variables, files, and external secret stores.
    """
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize secrets manager.
        
        Args:
            encryption_key: Optional key for encrypting/decrypting secrets
        """
        self._secrets: Dict[str, str] = {}
        self._encryption_key = encryption_key or os.getenv("SIGMALANG_SECRETS_KEY")
        self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load secrets from environment variables."""
        prefix = "SIGMALANG_SECRET_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                secret_name = key[len(prefix):].lower()
                self._secrets[secret_name] = value
    
    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret by name."""
        return self._secrets.get(name.lower(), default)
    
    def set(self, name: str, value: str) -> None:
        """Set a secret value."""
        self._secrets[name.lower()] = value
    
    def has(self, name: str) -> bool:
        """Check if a secret exists."""
        return name.lower() in self._secrets
    
    def load_from_file(self, path: Union[str, Path]) -> None:
        """Load secrets from a JSON file."""
        path = Path(path)
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    self._secrets[key.lower()] = value
    
    @staticmethod
    def hash_secret(value: str) -> str:
        """Create a hash of a secret for comparison."""
        return hashlib.sha256(value.encode()).hexdigest()
    
    @staticmethod
    def generate_key() -> str:
        """Generate a random API key."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class SigmalangConfig:
    """
    Main configuration class that aggregates all config sections.
    """
    environment: Environment = Environment.DEVELOPMENT
    server: ServerConfig = field(default_factory=ServerConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    @classmethod
    def from_env(cls) -> 'SigmalangConfig':
        """Create complete configuration from environment variables."""
        env_str = os.getenv("SIGMALANG_ENV", "development")
        environment = Environment.from_string(env_str)
        
        config = cls(
            environment=environment,
            server=ServerConfig.from_env(),
            rate_limit=RateLimitConfig.from_env(),
            auth=AuthConfig.from_env(),
            cache=CacheConfig.from_env(),
            encoder=EncoderConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            features=FeatureFlags.from_env(),
        )
        
        # Apply environment-specific defaults
        config._apply_environment_defaults()
        
        return config
    
    def _apply_environment_defaults(self) -> None:
        """Apply environment-specific default overrides."""
        if self.environment == Environment.DEVELOPMENT:
            self.server.debug = True
            self.server.reload = True
            self.auth.enabled = False
            self.monitoring.log_format = "text"
            
        elif self.environment == Environment.TESTING:
            self.server.debug = True
            self.cache.enabled = False
            self.auth.enabled = False
            self.rate_limit.enabled = False
            
        elif self.environment == Environment.PRODUCTION:
            self.server.debug = False
            self.server.reload = False
            self.monitoring.log_format = "json"
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Production-specific validations
        if self.is_production():
            if self.auth.enabled and not self.auth.api_keys:
                errors.append("Production requires API keys when auth is enabled")
            if self.auth.jwt_enabled and not self.auth.jwt_secret:
                errors.append("JWT secret is required when JWT is enabled")
            if self.server.debug:
                errors.append("Debug mode should be disabled in production")
        
        # General validations
        if self.server.port < 1 or self.server.port > 65535:
            errors.append(f"Invalid port number: {self.server.port}")
        if self.server.workers < 1:
            errors.append(f"Workers must be at least 1: {self.server.workers}")
        if self.rate_limit.requests_per_minute < 1:
            errors.append("Rate limit must be at least 1 request per minute")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets)."""
        return {
            "environment": self.environment.value,
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "debug": self.server.debug,
                "log_level": self.server.log_level,
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "requests_per_minute": self.rate_limit.requests_per_minute,
            },
            "auth": {
                "enabled": self.auth.enabled,
                "jwt_enabled": self.auth.jwt_enabled,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "backend": self.cache.backend,
            },
            "encoder": {
                "dimensions": self.encoder.dimensions,
                "parallel": self.encoder.enable_parallel,
            },
            "monitoring": {
                "metrics_enabled": self.monitoring.metrics_enabled,
                "tracing_enabled": self.monitoring.tracing_enabled,
            },
            "features": {
                "analogies": self.features.enable_analogies,
                "search": self.features.enable_search,
                "entities": self.features.enable_entities,
                "nlp": self.features.enable_nlp,
            },
        }


# =============================================================================
# Global Configuration Instance
# =============================================================================

_global_config: Optional[SigmalangConfig] = None
_global_secrets: Optional[SecretsManager] = None


def get_config() -> SigmalangConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = SigmalangConfig.from_env()
    return _global_config


def set_config(config: SigmalangConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration (for testing)."""
    global _global_config
    _global_config = None


def get_secrets() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _global_secrets
    if _global_secrets is None:
        _global_secrets = SecretsManager()
    return _global_secrets


def set_secrets(secrets: SecretsManager) -> None:
    """Set the global secrets manager instance."""
    global _global_secrets
    _global_secrets = secrets


# =============================================================================
# Configuration Loader Utilities
# =============================================================================

def load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a JSON or TOML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        if path.suffix == '.json':
            return json.load(f)
        elif path.suffix == '.toml':
            try:
                import tomllib
                return tomllib.loads(f.read())
            except ImportError:
                import tomli
                return tomli.loads(f.read())
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# Environment Variable Helpers
# =============================================================================

def env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def env_int(key: str, default: int = 0) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(key: str, default: float = 0.0) -> float:
    """Get float from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_list(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    """Get list from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]


def configure_from_file(path: str) -> Dict[str, Any]:
    """
    Load configuration from a file and apply it to global config.
    
    Args:
        path: Path to configuration file (YAML, JSON, or TOML)
        
    Returns:
        The loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If file format is unsupported
    """
    config = load_config_file(path)
    
    # Merge with existing config
    current = get_config()
    merged = merge_configs(current, config)
    
    # Apply merged config
    for key, value in merged.items():
        set_config(key, value)
    
    return merged


def configure_from_env(prefix: str = "SIGMALANG") -> Dict[str, Any]:
    """
    Load configuration from environment variables with given prefix.
    
    Environment variables should be named like:
    - SIGMALANG_DEBUG -> config["debug"]
    - SIGMALANG_LOG_LEVEL -> config["log_level"]
    - SIGMALANG_API_HOST -> config["api_host"]
    
    Args:
        prefix: Environment variable prefix (default: "SIGMALANG")
        
    Returns:
        Dictionary of configuration values loaded from environment
    """
    config = {}
    prefix_upper = prefix.upper() + "_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix_upper):
            # Convert SIGMALANG_LOG_LEVEL to log_level
            config_key = key[len(prefix_upper):].lower()
            
            # Type coercion based on value content
            if value.lower() in ("true", "false"):
                config[config_key] = value.lower() == "true"
            elif value.isdigit():
                config[config_key] = int(value)
            elif _is_float(value):
                config[config_key] = float(value)
            else:
                config[config_key] = value
    
    # Apply to global config
    for key, value in config.items():
        set_config(key, value)
    
    return config


def _is_float(value: str) -> bool:
    """Check if string represents a float value."""
    try:
        float(value)
        return "." in value or "e" in value.lower()
    except ValueError:
        return False
