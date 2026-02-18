"""
NAS Search Space - Phase 7 Track 3

Defines the architecture search space for SigmaLang encoder/decoder:
layer count, dimensions, activation functions, skip connections,
pooling strategies, and codebook parameters.

The search space is encoded as a flat vector of categorical and continuous
choices, making it compatible with evolutionary and RL-based search.

Research Basis:
    - AMC (2018): RL-based compression architecture search
    - "Search for Efficient LLMs" (Sep 2024): training-free NAS
    - NAS-Bench: standardized search space encoding

Usage:
    space = SearchSpace()
    config = space.sample_random()
    config_dict = config.to_dict()
    restored = ArchitectureConfig.from_dict(config_dict)
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Architecture Configuration
# =============================================================================

# Discrete choices for each dimension
LAYER_COUNTS = [1, 2, 3, 4, 6, 8]
HIDDEN_DIMS = [32, 64, 128, 256, 512]
ACTIVATIONS = ['relu', 'gelu', 'tanh', 'silu', 'identity']
POOLING_TYPES = ['mean', 'max', 'attention', 'cls_token']
NORMALIZATION = ['none', 'layer_norm', 'batch_norm', 'rms_norm']
SKIP_CONNECTIONS = ['none', 'residual', 'dense']
CODEBOOK_SIZES = [64, 128, 256, 512, 1024]
EMBEDDING_DIMS = [32, 64, 128, 256]
QUANTIZATION_BITS = [4, 8, 16, 32]
DROPOUT_RATES = [0.0, 0.05, 0.1, 0.2, 0.3]


@dataclass
class ArchitectureConfig:
    """
    Complete architecture specification for a SigmaLang encoder/decoder pair.

    This is the genome of the NAS search — each field is a gene.
    """
    # Encoder
    encoder_layers: int = 3
    encoder_hidden_dim: int = 128
    encoder_activation: str = 'gelu'
    encoder_pooling: str = 'attention'
    encoder_norm: str = 'layer_norm'
    encoder_skip: str = 'residual'
    encoder_dropout: float = 0.1

    # Decoder
    decoder_layers: int = 2
    decoder_hidden_dim: int = 128
    decoder_activation: str = 'gelu'
    decoder_norm: str = 'layer_norm'
    decoder_skip: str = 'none'
    decoder_dropout: float = 0.1

    # Codebook
    codebook_size: int = 256
    embedding_dim: int = 64
    quantization_bits: int = 8

    # Meta
    generation: int = 0
    parent_id: Optional[str] = None
    architecture_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'encoder_layers': self.encoder_layers,
            'encoder_hidden_dim': self.encoder_hidden_dim,
            'encoder_activation': self.encoder_activation,
            'encoder_pooling': self.encoder_pooling,
            'encoder_norm': self.encoder_norm,
            'encoder_skip': self.encoder_skip,
            'encoder_dropout': self.encoder_dropout,
            'decoder_layers': self.decoder_layers,
            'decoder_hidden_dim': self.decoder_hidden_dim,
            'decoder_activation': self.decoder_activation,
            'decoder_norm': self.decoder_norm,
            'decoder_skip': self.decoder_skip,
            'decoder_dropout': self.decoder_dropout,
            'codebook_size': self.codebook_size,
            'embedding_dim': self.embedding_dim,
            'quantization_bits': self.quantization_bits,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'architecture_id': self.architecture_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ArchitectureConfig':
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def estimated_params(self) -> int:
        """Estimate parameter count for this architecture."""
        enc_params = (
            self.encoder_layers *
            self.encoder_hidden_dim *
            self.encoder_hidden_dim
        )
        dec_params = (
            self.decoder_layers *
            self.decoder_hidden_dim *
            self.decoder_hidden_dim
        )
        cb_params = self.codebook_size * self.embedding_dim
        return enc_params + dec_params + cb_params

    def estimated_memory_bytes(self) -> int:
        """Estimate memory footprint in bytes."""
        params = self.estimated_params()
        bytes_per_param = self.quantization_bits / 8
        return int(params * bytes_per_param)

    def complexity_score(self) -> float:
        """
        Compute a normalized complexity score (0-1).

        Higher = more complex architecture.
        """
        layer_score = (self.encoder_layers + self.decoder_layers) / (max(LAYER_COUNTS) * 2)
        dim_score = self.encoder_hidden_dim / max(HIDDEN_DIMS)
        cb_score = self.codebook_size / max(CODEBOOK_SIZES)
        return (layer_score + dim_score + cb_score) / 3.0


# =============================================================================
# Search Space
# =============================================================================

class SearchSpace:
    """
    Defines and navigates the architecture search space.

    Provides random sampling, mutation, crossover, and encoding/decoding
    between architecture configs and flat numeric vectors.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

        # Define the search dimensions
        self.dimensions = {
            'encoder_layers': LAYER_COUNTS,
            'encoder_hidden_dim': HIDDEN_DIMS,
            'encoder_activation': ACTIVATIONS,
            'encoder_pooling': POOLING_TYPES,
            'encoder_norm': NORMALIZATION,
            'encoder_skip': SKIP_CONNECTIONS,
            'encoder_dropout': DROPOUT_RATES,
            'decoder_layers': LAYER_COUNTS,
            'decoder_hidden_dim': HIDDEN_DIMS,
            'decoder_activation': ACTIVATIONS,
            'decoder_norm': NORMALIZATION,
            'decoder_skip': SKIP_CONNECTIONS,
            'decoder_dropout': DROPOUT_RATES,
            'codebook_size': CODEBOOK_SIZES,
            'embedding_dim': EMBEDDING_DIMS,
            'quantization_bits': QUANTIZATION_BITS,
        }

        self.total_configs = 1
        for choices in self.dimensions.values():
            self.total_configs *= len(choices)

    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions)

    def sample_random(self) -> ArchitectureConfig:
        """Sample a random architecture from the search space."""
        config = {}
        for name, choices in self.dimensions.items():
            config[name] = choices[self._rng.randint(len(choices))]

        return ArchitectureConfig(**config)

    def mutate(
        self,
        config: ArchitectureConfig,
        mutation_rate: float = 0.2
    ) -> ArchitectureConfig:
        """
        Mutate an architecture by randomly changing some genes.

        Args:
            config: parent architecture
            mutation_rate: probability of mutating each gene

        Returns:
            New mutated architecture
        """
        d = config.to_dict()
        for name, choices in self.dimensions.items():
            if self._rng.random() < mutation_rate:
                d[name] = choices[self._rng.randint(len(choices))]

        child = ArchitectureConfig.from_dict(d)
        child.parent_id = config.architecture_id
        child.generation = config.generation + 1
        return child

    def crossover(
        self,
        parent_a: ArchitectureConfig,
        parent_b: ArchitectureConfig
    ) -> ArchitectureConfig:
        """
        Uniform crossover between two parents.

        Each gene is randomly inherited from one parent.
        """
        da = parent_a.to_dict()
        db = parent_b.to_dict()
        child_dict = {}

        for name in self.dimensions:
            if self._rng.random() < 0.5:
                child_dict[name] = da[name]
            else:
                child_dict[name] = db[name]

        child = ArchitectureConfig.from_dict(child_dict)
        child.parent_id = parent_a.architecture_id
        child.generation = max(parent_a.generation, parent_b.generation) + 1
        return child

    def encode_to_vector(self, config: ArchitectureConfig) -> np.ndarray:
        """
        Encode architecture config to a flat numeric vector.

        Each categorical choice is encoded as its index within the choice list.
        """
        d = config.to_dict()
        vec = []
        for name, choices in self.dimensions.items():
            val = d.get(name, choices[0])
            try:
                idx = choices.index(val)
            except ValueError:
                idx = 0
            vec.append(idx / max(len(choices) - 1, 1))  # Normalize to [0, 1]
        return np.array(vec, dtype=np.float32)

    def decode_from_vector(self, vec: np.ndarray) -> ArchitectureConfig:
        """Decode a flat numeric vector back to architecture config."""
        d = {}
        for i, (name, choices) in enumerate(self.dimensions.items()):
            idx = int(round(vec[i] * (len(choices) - 1)))
            idx = max(0, min(idx, len(choices) - 1))
            d[name] = choices[idx]
        return ArchitectureConfig.from_dict(d)

    def get_search_space_summary(self) -> Dict[str, Any]:
        """Get summary of the search space."""
        return {
            'dimensions': self.num_dimensions,
            'total_configurations': self.total_configs,
            'choices_per_dimension': {
                name: len(choices) for name, choices in self.dimensions.items()
            },
        }
