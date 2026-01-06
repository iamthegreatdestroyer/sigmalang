"""
SigmaLang Python Client
=======================
"""

import requests
from typing import Dict, Any, Optional
import json

class SigmaLang:
    """SigmaLang API Client"""

    def __init__(self, api_key: str, base_url: str = "https://api.sigmalang.com"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def compress(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compress text using SigmaLang semantic compression"""
        payload = {"text": text}
        if options:
            payload.update(options)

        response = self.session.post(f"{self.base_url}/compress", json=payload)
        response.raise_for_status()
        return response.json()

    def decompress(self, compressed: str) -> str:
        """Decompress SigmaLang compressed data"""
        payload = {"compressed": compressed}
        response = self.session.post(f"{self.base_url}/decompress", json=payload)
        response.raise_for_status()
        return response.json()["text"]

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text semantic structure"""
        payload = {"text": text}
        response = self.session.post(f"{self.base_url}/analyze", json=payload)
        response.raise_for_status()
        return response.json()
