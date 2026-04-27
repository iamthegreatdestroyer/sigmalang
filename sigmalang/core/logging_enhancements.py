"""
Structured Logging Enhancements - Sprint 4

Enhancements to the existing StructuredLogger:
- Correlation IDs for request tracing
- Sensitive data redaction
- Request context propagation
- Enhanced JSON formatting
"""

import re
import threading
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, List, Optional, Set

# Context variables for async-safe request tracking
REQUEST_ID: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
USER_ID: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
TRACE_ID: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


# =============================================================================
# Sensitive Data Redaction
# =============================================================================

class SensitiveDataRedactor:
    """Redacts sensitive information from log messages."""

    # Patterns for sensitive data
    PATTERNS = {
        'api_key': re.compile(r'\b(api[_-]?key|apikey)[\s:=]+["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.IGNORECASE),
        'password': re.compile(r'\b(password|passwd|pwd)[\s:=]+["\']?([^\s"\']{6,})["\']?', re.IGNORECASE),
        'token': re.compile(r'\b(token|bearer)[\s:=]+["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.IGNORECASE),
        'secret': re.compile(r'\b(secret|client_secret)[\s:=]+["\']?([a-zA-Z0-9_\-]{16,})["\']?', re.IGNORECASE),
        'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
        'ssn': re.compile(r'\b\d{3}[\s-]?\d{2}[\s-]?\d{4}\b'),
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    }

    # Fields to always redact
    SENSITIVE_FIELDS = {
        'password', 'passwd', 'pwd', 'secret', 'api_key', 'apikey',
        'token', 'access_token', 'refresh_token', 'bearer', 'authorization',
        'credit_card', 'cvv', 'ssn', 'private_key'
    }

    def __init__(self, redaction_char: str = '*'):
        self.redaction_char = redaction_char
        self.enabled = True

    def redact_string(self, text: str) -> str:
        """Redact sensitive data from a string."""
        if not self.enabled or not isinstance(text, str):
            return text

        # Apply pattern-based redaction
        for pattern_name, pattern in self.PATTERNS.items():
            if pattern_name in {'email', 'ip_address'}:
                # Partial redaction for emails and IPs
                text = pattern.sub(lambda m: self._partial_redact(m.group(0)), text)
            else:
                # Full redaction for secrets
                text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)

        return text

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from a dictionary."""
        if not self.enabled or not isinstance(data, dict):
            return data

        redacted = {}
        for key, value in data.items():
            # Check if key is sensitive
            if key.lower() in self.SENSITIVE_FIELDS:
                redacted[key] = '[REDACTED]'
            elif isinstance(value, str):
                redacted[key] = self.redact_string(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_dict(item) if isinstance(item, dict)
                    else self.redact_string(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                redacted[key] = value

        return redacted

    def _partial_redact(self, text: str) -> str:
        """Partially redact a string (show first and last few chars)."""
        if len(text) <= 6:
            return self.redaction_char * len(text)

        if '@' in text:  # Email
            parts = text.split('@')
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                redacted_local = local[0] + self.redaction_char * (len(local) - 2) + local[-1] if len(local) > 2 else self.redaction_char * len(local)
                return f"{redacted_local}@{domain}"

        # Default partial redaction
        return text[:2] + self.redaction_char * (len(text) - 4) + text[-2:]


# Global redactor instance
_redactor = SensitiveDataRedactor()


def get_redactor() -> SensitiveDataRedactor:
    """Get the global sensitive data redactor."""
    return _redactor


# =============================================================================
# Correlation ID Management
# =============================================================================

class CorrelationIDManager:
    """Manages correlation IDs for request tracing."""

    @staticmethod
    def generate_request_id() -> str:
        """Generate a new request ID."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID."""
        return str(uuid.uuid4())

    @staticmethod
    def set_request_id(request_id: Optional[str] = None) -> str:
        """Set the request ID for the current context."""
        if request_id is None:
            request_id = CorrelationIDManager.generate_request_id()
        REQUEST_ID.set(request_id)
        return request_id

    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get the request ID for the current context."""
        return REQUEST_ID.get()

    @staticmethod
    def set_trace_id(trace_id: Optional[str] = None) -> str:
        """Set the trace ID for the current context."""
        if trace_id is None:
            trace_id = CorrelationIDManager.generate_trace_id()
        TRACE_ID.set(trace_id)
        return trace_id

    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get the trace ID for the current context."""
        return TRACE_ID.get()

    @staticmethod
    def set_user_id(user_id: Optional[str]) -> None:
        """Set the user ID for the current context."""
        USER_ID.set(user_id)

    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get the user ID for the current context."""
        return USER_ID.get()

    @staticmethod
    def clear_context() -> None:
        """Clear all context variables."""
        REQUEST_ID.set(None)
        TRACE_ID.set(None)
        USER_ID.set(None)

    @staticmethod
    def get_context() -> Dict[str, Optional[str]]:
        """Get all context variables."""
        return {
            'request_id': REQUEST_ID.get(),
            'trace_id': TRACE_ID.get(),
            'user_id': USER_ID.get()
        }


# =============================================================================
# Enhanced Structured Logger
# =============================================================================

class EnhancedStructuredLogger:
    """
    Enhanced StructuredLogger with Sprint 4 improvements:
    - Automatic correlation ID injection
    - Sensitive data redaction
    - Request context propagation
    """

    def __init__(self, base_logger):
        """
        Wrap an existing StructuredLogger.

        Args:
            base_logger: Instance of StructuredLogger from monitoring.py
        """
        self.base_logger = base_logger
        self.redactor = get_redactor()
        self.correlation_manager = CorrelationIDManager()

    def _enrich_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich log context with correlation IDs."""
        # Get correlation IDs
        request_id = self.correlation_manager.get_request_id()
        trace_id = self.correlation_manager.get_trace_id()
        user_id = self.correlation_manager.get_user_id()

        # Add to context if available
        if request_id:
            kwargs['request_id'] = request_id
        if trace_id:
            kwargs['trace_id'] = trace_id
        if user_id:
            kwargs['user_id'] = user_id

        return kwargs

    def _redact_sensitive_data(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive data from log context."""
        return self.redactor.redact_dict(kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with enhancements."""
        kwargs = self._enrich_context(kwargs)
        kwargs = self._redact_sensitive_data(kwargs)
        message = self.redactor.redact_string(message)
        self.base_logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message with enhancements."""
        kwargs = self._enrich_context(kwargs)
        kwargs = self._redact_sensitive_data(kwargs)
        message = self.redactor.redact_string(message)
        self.base_logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with enhancements."""
        kwargs = self._enrich_context(kwargs)
        kwargs = self._redact_sensitive_data(kwargs)
        message = self.redactor.redact_string(message)
        self.base_logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with enhancements."""
        kwargs = self._enrich_context(kwargs)
        kwargs = self._redact_sensitive_data(kwargs)
        message = self.redactor.redact_string(message)
        self.base_logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with enhancements."""
        kwargs = self._enrich_context(kwargs)
        kwargs = self._redact_sensitive_data(kwargs)
        message = self.redactor.redact_string(message)
        self.base_logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with enhancements."""
        kwargs = self._enrich_context(kwargs)
        kwargs = self._redact_sensitive_data(kwargs)
        message = self.redactor.redact_string(message)
        self.base_logger.exception(message, **kwargs)

    def set_context(self, **kwargs) -> None:
        """Set persistent context."""
        self.base_logger.set_context(**kwargs)

    def with_context(self, **kwargs):
        """Temporary context manager."""
        return self.base_logger.with_context(**kwargs)


# =============================================================================
# Decorators for Request Tracing
# =============================================================================

def with_correlation_id(func):
    """
    Decorator to automatically assign a correlation ID to a request/function.

    Usage:
        @with_correlation_id
        def handle_request():
            logger.info("Processing request")  # Will include request_id
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate and set request ID
        CorrelationIDManager.set_request_id()

        try:
            return func(*args, **kwargs)
        finally:
            # Clear context after request
            CorrelationIDManager.clear_context()

    return wrapper


def with_trace(trace_name: Optional[str] = None):
    """
    Decorator to create a traced span for a function.

    Usage:
        @with_trace("encode_operation")
        def encode_text(text):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # Set trace ID if not already set
            if not CorrelationIDManager.get_trace_id():
                CorrelationIDManager.set_trace_id()

            # Execute function
            return func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Factory Functions
# =============================================================================

def create_enhanced_logger(name: str) -> EnhancedStructuredLogger:
    """
    Create an enhanced structured logger.

    Usage:
        from sigmalang.core.logging_enhancements import create_enhanced_logger
        logger = create_enhanced_logger(__name__)
        logger.info("Processing request", user_id="12345")
    """
    from .monitoring import StructuredLogger
    base_logger = StructuredLogger(name)
    return EnhancedStructuredLogger(base_logger)
