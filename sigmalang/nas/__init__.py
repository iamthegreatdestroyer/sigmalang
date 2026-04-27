"""
Neural Architecture Search for SigmaLang - Phase 7 Track 3

Auto-discover optimal encoder/decoder architectures that
maximize compression ratio while minimizing latency.
"""

from sigmalang.nas.evaluator import ArchitectureEvaluator
from sigmalang.nas.evolutionary_search import EvolutionarySearch
from sigmalang.nas.registry import ArchitectureRegistry
from sigmalang.nas.search_space import ArchitectureConfig, SearchSpace

__all__ = [
    'SearchSpace', 'ArchitectureConfig',
    'ArchitectureEvaluator',
    'EvolutionarySearch',
    'ArchitectureRegistry',
]
