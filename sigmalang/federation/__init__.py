"""
Federated Codebook Learning - Phase 7 Track 2

Distributed Tier-2 primitive discovery across multiple SigmaLang nodes
with differential privacy guarantees.
"""

from sigmalang.federation.privacy import DifferentialPrivacy, PrivacyConfig
from sigmalang.federation.client import FederationClient
from sigmalang.federation.aggregation_server import AggregationServer
from sigmalang.federation.consensus import ConsensusProtocol

__all__ = [
    'DifferentialPrivacy', 'PrivacyConfig',
    'FederationClient',
    'AggregationServer',
    'ConsensusProtocol',
]
