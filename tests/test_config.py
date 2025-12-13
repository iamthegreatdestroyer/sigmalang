"""
Tests for Phase 3: Configuration Management

Comprehensive test suite for the configuration system.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config import (
    # Enums
    Environment,
    # Config classes
    ServerConfig,
    RateLimitConfig,
    AuthConfig,
    CacheConfig,
    EncoderConfig,
    MonitoringConfig,
    FeatureFlags,
    # Secrets
    SecretsManager,
    # Main config
    SigmalangConfig,
    # Functions
    get_config,
    set_config,
    reset_config,
    get_secrets,
    set_secrets,
    load_config_file,
    merge_configs,
    env_bool,
    env_int,
    env_float,
    env_list,
)


# =============================================================================
# Environment Tests
# =============================================================================

class TestEnvironment:
    """Tests for Environment enum."""
    
    def test_environment_values(self):
        """Test all environment values exist."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
    
    def test_from_string_valid(self):
        """Test parsing valid environment strings."""
        assert Environment.from_string("development") == Environment.DEVELOPMENT
        assert Environment.from_string("PRODUCTION") == Environment.PRODUCTION
        assert Environment.from_string("Testing") == Environment.TESTING
    
    def test_from_string_invalid(self):
        """Test parsing invalid environment string defaults to development."""
        assert Environment.from_string("invalid") == Environment.DEVELOPMENT
        assert Environment.from_string("") == Environment.DEVELOPMENT


# =============================================================================
# ServerConfig Tests
# =============================================================================

class TestServerConfig:
    """Tests for ServerConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4
        assert config.reload is False
        assert config.debug is False
        assert config.log_level == "INFO"
    
    def test_from_env(self):
        """Test creating config from environment variables."""
        with patch.dict(os.environ, {
            "SIGMALANG_HOST": "127.0.0.1",
            "SIGMALANG_PORT": "9000",
            "SIGMALANG_WORKERS": "8",
            "SIGMALANG_DEBUG": "true",
        }):
            config = ServerConfig.from_env()
            
            assert config.host == "127.0.0.1"
            assert config.port == 9000
            assert config.workers == 8
            assert config.debug is True


# =============================================================================
# RateLimitConfig Tests
# =============================================================================

class TestRateLimitConfig:
    """Tests for RateLimitConfig."""
    
    def test_default_values(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        
        assert config.enabled is True
        assert config.requests_per_minute == 60
        assert config.burst_size == 10
    
    def test_from_env(self):
        """Test creating from environment."""
        with patch.dict(os.environ, {
            "SIGMALANG_RATE_LIMIT_ENABLED": "false",
            "SIGMALANG_RATE_LIMIT_RPM": "120",
        }):
            config = RateLimitConfig.from_env()
            
            assert config.enabled is False
            assert config.requests_per_minute == 120


# =============================================================================
# AuthConfig Tests
# =============================================================================

class TestAuthConfig:
    """Tests for AuthConfig."""
    
    def test_default_values(self):
        """Test default auth configuration."""
        config = AuthConfig()
        
        assert config.enabled is False
        assert config.jwt_enabled is False
        assert len(config.api_keys) == 0
    
    def test_validate_api_key_disabled(self):
        """Test API key validation when auth disabled."""
        config = AuthConfig(enabled=False)
        
        assert config.validate_api_key("any_key") is True
    
    def test_validate_api_key_enabled(self):
        """Test API key validation when auth enabled."""
        config = AuthConfig(
            enabled=True,
            api_keys={"valid_key_1", "valid_key_2"}
        )
        
        assert config.validate_api_key("valid_key_1") is True
        assert config.validate_api_key("invalid_key") is False
    
    def test_from_env_with_api_keys(self):
        """Test loading API keys from environment."""
        with patch.dict(os.environ, {
            "SIGMALANG_AUTH_ENABLED": "true",
            "SIGMALANG_API_KEYS": "key1,key2,key3",
        }):
            config = AuthConfig.from_env()
            
            assert config.enabled is True
            assert "key1" in config.api_keys
            assert "key2" in config.api_keys
            assert len(config.api_keys) == 3


# =============================================================================
# CacheConfig Tests
# =============================================================================

class TestCacheConfig:
    """Tests for CacheConfig."""
    
    def test_default_values(self):
        """Test default cache configuration."""
        config = CacheConfig()
        
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.default_ttl == 3600
    
    def test_from_env(self):
        """Test creating from environment."""
        with patch.dict(os.environ, {
            "SIGMALANG_CACHE_BACKEND": "redis",
            "SIGMALANG_REDIS_URL": "redis://myhost:6379/1",
        }):
            config = CacheConfig.from_env()
            
            assert config.backend == "redis"
            assert config.redis_url == "redis://myhost:6379/1"


# =============================================================================
# EncoderConfig Tests
# =============================================================================

class TestEncoderConfig:
    """Tests for EncoderConfig."""
    
    def test_default_values(self):
        """Test default encoder configuration."""
        config = EncoderConfig()
        
        assert config.dimensions == 512
        assert config.hash_functions == 8
        assert config.enable_parallel is True


# =============================================================================
# MonitoringConfig Tests
# =============================================================================

class TestMonitoringConfig:
    """Tests for MonitoringConfig."""
    
    def test_default_values(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()
        
        assert config.metrics_enabled is True
        assert config.health_check_enabled is True
        assert config.tracing_enabled is False
        assert config.log_format == "json"


# =============================================================================
# FeatureFlags Tests
# =============================================================================

class TestFeatureFlags:
    """Tests for FeatureFlags."""
    
    def test_default_values(self):
        """Test default feature flags."""
        flags = FeatureFlags()
        
        assert flags.enable_analogies is True
        assert flags.enable_search is True
        assert flags.enable_entities is True
        assert flags.enable_admin is False
    
    def test_is_enabled(self):
        """Test feature enabled check."""
        flags = FeatureFlags(
            enable_analogies=True,
            enable_admin=False,
            experimental_features={"new_feature"}
        )
        
        assert flags.is_enabled("analogies") is True
        assert flags.is_enabled("admin") is False
        assert flags.is_enabled("new_feature") is True
        assert flags.is_enabled("unknown") is False


# =============================================================================
# SecretsManager Tests
# =============================================================================

class TestSecretsManager:
    """Tests for SecretsManager."""
    
    def test_get_set_secret(self):
        """Test getting and setting secrets."""
        secrets = SecretsManager()
        
        secrets.set("my_secret", "secret_value")
        
        assert secrets.get("my_secret") == "secret_value"
        assert secrets.get("MY_SECRET") == "secret_value"  # Case insensitive
    
    def test_get_default(self):
        """Test getting non-existent secret with default."""
        secrets = SecretsManager()
        
        assert secrets.get("nonexistent") is None
        assert secrets.get("nonexistent", "default") == "default"
    
    def test_has_secret(self):
        """Test checking if secret exists."""
        secrets = SecretsManager()
        secrets.set("exists", "value")
        
        assert secrets.has("exists") is True
        assert secrets.has("not_exists") is False
    
    def test_load_from_file(self):
        """Test loading secrets from file."""
        secrets = SecretsManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"api_key": "12345", "db_password": "secret"}, f)
            path = f.name
        
        try:
            secrets.load_from_file(path)
            
            assert secrets.get("api_key") == "12345"
            assert secrets.get("db_password") == "secret"
        finally:
            Path(path).unlink()
    
    def test_hash_secret(self):
        """Test secret hashing."""
        hash1 = SecretsManager.hash_secret("password123")
        hash2 = SecretsManager.hash_secret("password123")
        hash3 = SecretsManager.hash_secret("different")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_generate_key(self):
        """Test key generation."""
        key1 = SecretsManager.generate_key()
        key2 = SecretsManager.generate_key()
        
        assert len(key1) > 20
        assert key1 != key2
    
    def test_load_from_env(self):
        """Test loading secrets from environment."""
        with patch.dict(os.environ, {
            "SIGMALANG_SECRET_API_KEY": "env_api_key",
            "SIGMALANG_SECRET_DB_PASSWORD": "env_db_pass",
        }):
            secrets = SecretsManager()
            
            assert secrets.get("api_key") == "env_api_key"
            assert secrets.get("db_password") == "env_db_pass"


# =============================================================================
# SigmalangConfig Tests
# =============================================================================

class TestSigmalangConfig:
    """Tests for main SigmalangConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        reset_config()
        config = SigmalangConfig()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.server.port == 8000
        assert config.cache.enabled is True
    
    def test_from_env(self):
        """Test creating from environment."""
        with patch.dict(os.environ, {
            "SIGMALANG_ENV": "production",
            "SIGMALANG_PORT": "9000",
        }):
            config = SigmalangConfig.from_env()
            
            assert config.environment == Environment.PRODUCTION
            assert config.server.port == 9000
    
    def test_environment_defaults_development(self):
        """Test development environment defaults."""
        with patch.dict(os.environ, {"SIGMALANG_ENV": "development"}):
            config = SigmalangConfig.from_env()
            
            assert config.server.debug is True
            assert config.server.reload is True
            assert config.auth.enabled is False
    
    def test_environment_defaults_production(self):
        """Test production environment defaults."""
        with patch.dict(os.environ, {"SIGMALANG_ENV": "production"}):
            config = SigmalangConfig.from_env()
            
            assert config.server.debug is False
            assert config.server.reload is False
            assert config.monitoring.log_format == "json"
    
    def test_is_production(self):
        """Test is_production check."""
        config = SigmalangConfig(environment=Environment.PRODUCTION)
        assert config.is_production() is True
        
        config = SigmalangConfig(environment=Environment.DEVELOPMENT)
        assert config.is_production() is False
    
    def test_validate_valid_config(self):
        """Test validation with valid config."""
        config = SigmalangConfig()
        errors = config.validate()
        
        assert len(errors) == 0
    
    def test_validate_invalid_port(self):
        """Test validation with invalid port."""
        config = SigmalangConfig()
        config.server.port = 99999
        
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("port" in e.lower() for e in errors)
    
    def test_validate_production_without_jwt_secret(self):
        """Test validation fails in production without JWT secret."""
        config = SigmalangConfig(environment=Environment.PRODUCTION)
        config.auth.enabled = True
        config.auth.jwt_enabled = True
        config.auth.jwt_secret = ""
        
        errors = config.validate()
        
        assert any("jwt" in e.lower() for e in errors)
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = SigmalangConfig()
        config_dict = config.to_dict()
        
        assert "environment" in config_dict
        assert "server" in config_dict
        assert "features" in config_dict
        assert config_dict["environment"] == "development"


# =============================================================================
# Global Config Tests
# =============================================================================

class TestGlobalConfig:
    """Tests for global configuration functions."""
    
    def test_get_config_creates_instance(self):
        """Test get_config creates instance if not exists."""
        reset_config()
        config = get_config()
        
        assert config is not None
        assert isinstance(config, SigmalangConfig)
    
    def test_get_config_returns_same_instance(self):
        """Test get_config returns same instance."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_set_config(self):
        """Test setting global config."""
        reset_config()
        custom_config = SigmalangConfig(environment=Environment.STAGING)
        set_config(custom_config)
        
        assert get_config().environment == Environment.STAGING
    
    def test_reset_config(self):
        """Test resetting global config."""
        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        assert config1 is not config2


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_env_bool(self):
        """Test env_bool function."""
        with patch.dict(os.environ, {"TEST_TRUE": "true", "TEST_FALSE": "false"}):
            assert env_bool("TEST_TRUE") is True
            assert env_bool("TEST_FALSE") is False
            assert env_bool("TEST_MISSING") is False
            assert env_bool("TEST_MISSING", True) is True
    
    def test_env_int(self):
        """Test env_int function."""
        with patch.dict(os.environ, {"TEST_INT": "42", "TEST_INVALID": "abc"}):
            assert env_int("TEST_INT") == 42
            assert env_int("TEST_INVALID") == 0
            assert env_int("TEST_MISSING", 10) == 10
    
    def test_env_float(self):
        """Test env_float function."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert abs(env_float("TEST_FLOAT") - 3.14) < 0.01
            assert env_float("TEST_MISSING", 1.0) == 1.0
    
    def test_env_list(self):
        """Test env_list function."""
        with patch.dict(os.environ, {"TEST_LIST": "a,b,c"}):
            result = env_list("TEST_LIST")
            
            assert result == ["a", "b", "c"]
            assert env_list("TEST_MISSING") == []
            assert env_list("TEST_MISSING", ["default"]) == ["default"]
    
    def test_load_config_file_json(self):
        """Test loading JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value", "nested": {"a": 1}}, f)
            path = f.name
        
        try:
            config = load_config_file(path)
            
            assert config["key"] == "value"
            assert config["nested"]["a"] == 1
        finally:
            Path(path).unlink()
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config_file("/nonexistent/config.json")
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 5}, "e": 6}
        
        result = merge_configs(base, override)
        
        assert result["a"] == 1
        assert result["b"]["c"] == 5
        assert result["b"]["d"] == 3
        assert result["e"] == 6


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
