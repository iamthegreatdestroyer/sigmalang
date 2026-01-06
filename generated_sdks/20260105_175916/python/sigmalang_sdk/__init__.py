"""
SigmaLang Python SDK
====================

Enterprise-grade SDK for SigmaLang semantic compression.

Usage:
    from sigmalang_sdk import SigmaLang

    client = SigmaLang(api_key="your-key")
    result = client.compress("Hello world")
"""

__version__ = "1.0.0"
__author__ = "SigmaLang Team"

from .client import SigmaLang

__all__ = ["SigmaLang"]
