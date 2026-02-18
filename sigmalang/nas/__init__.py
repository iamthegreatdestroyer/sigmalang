"""
Neural Architecture Search for SigmaLang - Phase 7 Track 3

Auto-discover optimal encoder/decoder architectures that
maximize compression ratio while minimizing latency.
"""

from sigmalang.nas.search_space import SearchSpace, ArchitectureConfig
from sigmalang.nas.evaluator import ArchitectureEvaluator
from sigmalang.nas.evolutionary_search import EvolutionarySearch
from sigmalang.nas.registry import ArchitectureRegistry

__all__ = [
    'SearchSpace', 'ArchitectureConfig',
    'ArchitectureEvaluator',
    'EvolutionarySearch',
    'ArchitectureRegistry',
]
