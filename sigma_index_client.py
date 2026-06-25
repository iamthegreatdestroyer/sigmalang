"""
sigma-index Python client.

Provides vector + text search to any Python project in the Sigma Ecosystem.
Connects to the sigma-index Go server on port 8200.

Used by: sigmalang, In-My-Head, sigma-harvest, FactMarrow, Steve-AI

Usage:
    from sigma_index_client import SigmaIndex

    idx = SigmaIndex()
    idx.add("doc1", vector=[0.1, 0.2, ...], text="some text")
    results = idx.search(text="query", k=5)
    results = idx.search(vector=[0.1, ...], text="query", k=10)  # hybrid
"""

import os
from typing import Optional

import httpx


class SigmaIndex:
    def __init__(self, base_url: str = "", namespace: str = "default"):
        self.base_url = base_url or os.getenv("SIGMA_INDEX_URL", "http://localhost:8200")
        self.namespace = namespace
        self._client = httpx.Client(timeout=10.0)

    def add(self, doc_id: str, vector: list[float] = None, text: str = ""):
        self._client.post(f"{self.base_url}/add", json={
            "namespace": self.namespace,
            "id": doc_id,
            "vector": vector or [],
            "text": text,
        })

    def search(
        self,
        vector: list[float] = None,
        text: str = "",
        k: int = 10,
    ) -> list[dict]:
        resp = self._client.post(f"{self.base_url}/search", json={
            "namespace": self.namespace,
            "vector": vector or [],
            "text": text,
            "k": k,
        })
        data = resp.json()
        return data.get("results", [])

    def delete(self, doc_id: str):
        self._client.post(f"{self.base_url}/delete", json={
            "namespace": self.namespace,
            "id": doc_id,
        })

    def healthy(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False


class SigmaIndexAsync:
    """Async version for use in FastAPI/async contexts."""

    def __init__(self, base_url: str = "", namespace: str = "default"):
        self.base_url = base_url or os.getenv("SIGMA_INDEX_URL", "http://localhost:8200")
        self.namespace = namespace

    async def add(self, doc_id: str, vector: list[float] = None, text: str = ""):
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{self.base_url}/add", json={
                "namespace": self.namespace,
                "id": doc_id,
                "vector": vector or [],
                "text": text,
            })

    async def search(
        self,
        vector: list[float] = None,
        text: str = "",
        k: int = 10,
    ) -> list[dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self.base_url}/search", json={
                "namespace": self.namespace,
                "vector": vector or [],
                "text": text,
                "k": k,
            })
            return resp.json().get("results", [])

    async def delete(self, doc_id: str):
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{self.base_url}/delete", json={
                "namespace": self.namespace,
                "id": doc_id,
            })
