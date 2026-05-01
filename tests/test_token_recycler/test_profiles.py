"""Tests for sigmalang.core.token_recycler.profiles."""

from __future__ import annotations

import pytest

from sigmalang.core.token_recycler.exceptions import ProfileNotFoundError
from sigmalang.core.token_recycler.profiles import (
    AGENT_TIER_MAP,
    CompressionProfile,
    get_profile_for_agent,
    tier_1_profile,
    tier_2_profile,
    tier_3_profile,
    tier_4_profile,
    tier_5_profile,
    tier_6_profile,
    tier_7_profile,
    tier_8_profile,
)


def test_module_imports() -> None:
    """All profile exports are importable."""
    assert CompressionProfile
    assert get_profile_for_agent
    assert AGENT_TIER_MAP


# ---------------------------------------------------------------------------
# CompressionProfile dataclass
# ---------------------------------------------------------------------------


def test_compression_profile_defaults() -> None:
    p = CompressionProfile()
    assert p.agent_id == ""
    assert p.tier == 2
    assert p.compression_ratio == pytest.approx(0.70)
    assert p.semantic_threshold == pytest.approx(0.85)
    assert p.refresh_interval == pytest.approx(1800.0)
    assert isinstance(p.critical_tokens, list)


def test_compression_profile_custom() -> None:
    p = CompressionProfile(
        agent_id="TEST",
        tier=1,
        compression_ratio=0.60,
        critical_tokens=["AES-256-GCM"],
        semantic_threshold=0.90,
    )
    assert p.agent_id == "TEST"
    assert p.tier == 1
    assert "AES-256-GCM" in p.critical_tokens


def test_critical_tokens_are_independent_instances() -> None:
    """Two profiles must not share the same list object."""
    p1 = CompressionProfile()
    p2 = CompressionProfile()
    p1.critical_tokens.append("SHARED_TOKEN")
    assert "SHARED_TOKEN" not in p2.critical_tokens


# ---------------------------------------------------------------------------
# Tier factory functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_tier,expected_ratio",
    [
        (tier_1_profile, 1, 0.60),
        (tier_2_profile, 2, 0.70),
        (tier_3_profile, 3, 0.50),
        (tier_4_profile, 4, 0.50),
        (tier_5_profile, 5, 0.65),
        (tier_6_profile, 6, 0.65),
        (tier_7_profile, 7, 0.65),
        (tier_8_profile, 8, 0.65),
    ],
)
def test_tier_factory_returns_correct_tier_and_ratio(factory, expected_tier, expected_ratio) -> None:
    p = factory()
    assert p.tier == expected_tier
    assert p.compression_ratio == pytest.approx(expected_ratio)


@pytest.mark.parametrize(
    "factory",
    [
        tier_1_profile,
        tier_2_profile,
        tier_3_profile,
        tier_4_profile,
        tier_5_profile,
        tier_6_profile,
        tier_7_profile,
        tier_8_profile,
    ],
)
def test_tier_factory_returns_compression_profile(factory) -> None:
    assert isinstance(factory(), CompressionProfile)


@pytest.mark.parametrize(
    "factory",
    [
        tier_1_profile,
        tier_2_profile,
        tier_3_profile,
        tier_4_profile,
        tier_5_profile,
        tier_6_profile,
        tier_7_profile,
        tier_8_profile,
    ],
)
def test_tier_factory_accepts_agent_id(factory) -> None:
    p = factory("MY_AGENT")
    assert p.agent_id == "MY_AGENT"


@pytest.mark.parametrize(
    "factory",
    [
        tier_1_profile,
        tier_2_profile,
        tier_3_profile,
        tier_4_profile,
        tier_5_profile,
        tier_6_profile,
        tier_7_profile,
        tier_8_profile,
    ],
)
def test_tier_factory_has_non_empty_critical_tokens(factory) -> None:
    p = factory()
    assert len(p.critical_tokens) > 0, f"{factory.__name__} must define critical tokens"


def test_tier_1_has_crypto_critical_tokens() -> None:
    p = tier_1_profile()
    assert any("AES" in t or "TLS" in t or "SHA" in t for t in p.critical_tokens)


def test_tier_7_has_hipaa_critical_token() -> None:
    p = tier_7_profile()
    assert "HIPAA" in p.critical_tokens


def test_tier_8_has_pci_dss_critical_token() -> None:
    p = tier_8_profile()
    assert "PCI-DSS" in p.critical_tokens


def test_tier_1_semantic_threshold_higher_than_default() -> None:
    """Tier 1 (security) should demand higher fidelity than default 0.85."""
    p1 = tier_1_profile()
    p2 = tier_2_profile()
    assert p1.semantic_threshold >= p2.semantic_threshold


def test_tier_3_lower_compression_ratio_than_tier_2() -> None:
    """Innovators compress less aggressively than specialists."""
    assert tier_3_profile().compression_ratio < tier_2_profile().compression_ratio


# ---------------------------------------------------------------------------
# AGENT_TIER_MAP coverage
# ---------------------------------------------------------------------------


def test_agent_tier_map_has_known_agents() -> None:
    known = {
        "APEX",
        "CIPHER",
        "ARCHITECT",
        "AXIOM",
        "VELOCITY",
        "SYNAPSE",
        "FLUX",
        "CORE",
        "STREAM",
        "NEXUS",
        "GENESIS",
        "OMNISCIENT",
        "ORACLE",
        "TOKEN_RECYCLER",
    }
    for agent in known:
        assert agent in AGENT_TIER_MAP, f"{agent} missing from AGENT_TIER_MAP"


def test_agent_tier_map_tiers_in_range() -> None:
    for agent, tier in AGENT_TIER_MAP.items():
        assert 1 <= tier <= 8, f"{agent} has tier {tier} outside 1–8"


# ---------------------------------------------------------------------------
# get_profile_for_agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_id,expected_tier",
    [
        ("APEX", 1),
        ("CIPHER", 1),
        ("ARCHITECT", 1),
        ("SYNAPSE", 2),
        ("TOKEN_RECYCLER", 2),
        ("NEXUS", 3),
        ("OMNISCIENT", 4),
        ("CLOUD", 5),
        ("EDGE", 6),
        ("HEALTHCARE", 7),
        ("FINANCE", 8),
    ],
)
def test_get_profile_for_known_agents(agent_id, expected_tier) -> None:
    p = get_profile_for_agent(agent_id)
    assert p.tier == expected_tier
    assert p.agent_id == agent_id


def test_get_profile_preserves_agent_id() -> None:
    p = get_profile_for_agent("CIPHER")
    assert p.agent_id == "CIPHER"


def test_get_profile_unknown_agent_raises() -> None:
    with pytest.raises(ProfileNotFoundError) as exc_info:
        get_profile_for_agent("TOTALLY_UNKNOWN_AGENT_XYZ")
    assert "TOTALLY_UNKNOWN_AGENT_XYZ" in str(exc_info.value)


def test_get_profile_empty_string_returns_default() -> None:
    """Empty agent_id should return a default profile without raising."""
    p = get_profile_for_agent("")
    assert isinstance(p, CompressionProfile)


def test_get_profile_case_insensitive() -> None:
    """Agent lookup should be case-insensitive."""
    p_upper = get_profile_for_agent("CIPHER")
    p_lower = get_profile_for_agent("cipher")
    assert p_upper.tier == p_lower.tier


def test_get_profile_returns_independent_instances() -> None:
    """Two calls must return separate profile objects."""
    p1 = get_profile_for_agent("CIPHER")
    p2 = get_profile_for_agent("CIPHER")
    p1.critical_tokens.append("MUTATED")
    assert "MUTATED" not in p2.critical_tokens


def test_all_tier_1_agents_get_tier_1_profile() -> None:
    tier_1_agents = [a for a, t in AGENT_TIER_MAP.items() if t == 1]
    assert len(tier_1_agents) > 0
    for agent in tier_1_agents:
        p = get_profile_for_agent(agent)
        assert p.tier == 1


def test_compression_ratio_within_valid_range() -> None:
    for agent in AGENT_TIER_MAP:
        p = get_profile_for_agent(agent)
        assert 0.0 < p.compression_ratio <= 1.0, f"{agent}: compression_ratio={p.compression_ratio} out of range"


def test_semantic_threshold_within_valid_range() -> None:
    for agent in AGENT_TIER_MAP:
        p = get_profile_for_agent(agent)
        assert 0.0 < p.semantic_threshold <= 1.0, f"{agent}: semantic_threshold={p.semantic_threshold} out of range"


def test_refresh_interval_positive() -> None:
    for agent in AGENT_TIER_MAP:
        p = get_profile_for_agent(agent)
        assert p.refresh_interval > 0
