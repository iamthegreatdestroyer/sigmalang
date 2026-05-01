"""
Compression profiles for the Token Recycler.

Each agent tier has a default CompressionProfile.  Per-agent overrides are
resolved by get_profile_for_agent(), which falls back to the tier default
when no exact match exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from .exceptions import ProfileNotFoundError

__all__ = [
    "CompressionProfile",
    "tier_1_profile",
    "tier_2_profile",
    "tier_3_profile",
    "tier_4_profile",
    "tier_5_profile",
    "tier_6_profile",
    "tier_7_profile",
    "tier_8_profile",
    "get_profile_for_agent",
    "AGENT_TIER_MAP",
]


@dataclass
class CompressionProfile:
    """Per-agent compression configuration.

    Attributes:
        agent_id: Agent identifier (empty string = tier default).
        tier: Agent tier (1–8).
        compression_ratio: Target token reduction fraction (0.40–0.70).
        critical_tokens: Tokens that must never be compressed or replaced.
        semantic_threshold: Cosine similarity floor for drift detection (0.85).
        refresh_interval: Seconds before a context is considered stale (1800).
        last_refresh: Unix timestamp of last refresh (0 = never).
    """

    agent_id: str = ""
    tier: int = 2
    compression_ratio: float = 0.70
    critical_tokens: List[str] = field(default_factory=list)
    semantic_threshold: float = 0.85
    refresh_interval: float = 1800.0
    last_refresh: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Tier factory functions
# ---------------------------------------------------------------------------


def tier_1_profile(agent_id: str = "") -> CompressionProfile:
    """Foundational agents: APEX, CIPHER, ARCHITECT, AXIOM, VELOCITY — 60% compression."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=1,
        compression_ratio=0.60,
        critical_tokens=[
            "AES-256-GCM",
            "ECDH-P384",
            "Argon2id",
            "SHA-256",
            "TLS 1.3",
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n^2)",
            "NIST",
            "OWASP",
            "Bloom Filter",
            "LSH",
            "HNSW",
        ],
        semantic_threshold=0.90,
        refresh_interval=1800.0,
    )


def tier_2_profile(agent_id: str = "") -> CompressionProfile:
    """Domain specialists: SYNAPSE, FLUX, CORE, STREAM — 70% compression."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=2,
        compression_ratio=0.70,
        critical_tokens=[
            "REST",
            "GraphQL",
            "gRPC",
            "WebSocket",
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "JWT",
            "OAuth2",
            "OpenID",
            "Docker",
            "Kubernetes",
            "Helm",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


def tier_3_profile(agent_id: str = "") -> CompressionProfile:
    """Innovators: NEXUS, GENESIS — 50% compression (needs more context)."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=3,
        compression_ratio=0.50,
        critical_tokens=[
            "breakthrough",
            "synthesis",
            "cross-domain",
            "meta-analysis",
            "pattern recognition",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


def tier_4_profile(agent_id: str = "") -> CompressionProfile:
    """Meta agents: OMNISCIENT, ORACLE — 50% compression."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=4,
        compression_ratio=0.50,
        critical_tokens=[
            "fitness",
            "evolution",
            "ReMem-Elite",
            "meta-learning",
            "phase",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


def tier_5_profile(agent_id: str = "") -> CompressionProfile:
    """Cloud agents — 65% compression."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=5,
        compression_ratio=0.65,
        critical_tokens=[
            "AWS",
            "GCP",
            "Azure",
            "S3",
            "Lambda",
            "IAM",
            "VPC",
            "CloudFormation",
            "Terraform",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


def tier_6_profile(agent_id: str = "") -> CompressionProfile:
    """Edge agents — 65% compression."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=6,
        compression_ratio=0.65,
        critical_tokens=[
            "MQTT",
            "edge",
            "IoT",
            "firmware",
            "latency",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


def tier_7_profile(agent_id: str = "") -> CompressionProfile:
    """Healthcare agents — 65% compression with compliance tokens."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=7,
        compression_ratio=0.65,
        critical_tokens=[
            "HIPAA",
            "PHI",
            "HL7",
            "FHIR",
            "ICD-10",
            "CPT",
            "EHR",
            "EMR",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


def tier_8_profile(agent_id: str = "") -> CompressionProfile:
    """Finance agents — 65% compression with regulatory tokens."""
    return CompressionProfile(
        agent_id=agent_id,
        tier=8,
        compression_ratio=0.65,
        critical_tokens=[
            "PCI-DSS",
            "SOX",
            "GDPR",
            "Basel III",
            "KYC",
            "AML",
            "FIX",
            "SWIFT",
        ],
        semantic_threshold=0.85,
        refresh_interval=1800.0,
    )


# ---------------------------------------------------------------------------
# Agent → tier mapping
# Transcribed from TOKEN_RECYCLER.agent.md section "Supports"
# ---------------------------------------------------------------------------

AGENT_TIER_MAP: Dict[str, int] = {
    # Tier 1 — Foundational
    "APEX": 1,
    "CIPHER": 1,
    "ARCHITECT": 1,
    "AXIOM": 1,
    "VELOCITY": 1,
    # Tier 2 — Domain specialists (representative members from spec)
    "SYNAPSE": 2,
    "FLUX": 2,
    "CORE": 2,
    "STREAM": 2,
    "TOKEN_RECYCLER": 2,
    # Tier 3 — Innovators
    "NEXUS": 3,
    "GENESIS": 3,
    # Tier 4 — Meta
    "OMNISCIENT": 4,
    "ORACLE": 4,
    # Tier 5 — Cloud
    "CLOUD": 5,
    # Tier 6 — Edge
    "EDGE": 6,
    # Tier 7 — Healthcare
    "HEALTHCARE": 7,
    # Tier 8 — Finance
    "FINANCE": 8,
}

_TIER_FACTORIES = {
    1: tier_1_profile,
    2: tier_2_profile,
    3: tier_3_profile,
    4: tier_4_profile,
    5: tier_5_profile,
    6: tier_6_profile,
    7: tier_7_profile,
    8: tier_8_profile,
}


def get_profile_for_agent(agent_id: str) -> CompressionProfile:
    """Return a CompressionProfile for the given agent.

    Looks up the agent's tier in AGENT_TIER_MAP, then calls the
    corresponding tier factory.  Raises ProfileNotFoundError only when
    the agent_id is non-empty and cannot be resolved to any tier.

    Args:
        agent_id: Case-sensitive agent identifier (e.g. 'CIPHER').

    Returns:
        A fresh CompressionProfile with agent_id set.

    Raises:
        ProfileNotFoundError: When agent_id is unknown.
    """
    if not agent_id:
        return tier_2_profile("")

    tier = AGENT_TIER_MAP.get(agent_id.upper())
    if tier is None:
        raise ProfileNotFoundError(agent_id)

    factory = _TIER_FACTORIES[tier]
    profile = factory(agent_id)
    return profile
