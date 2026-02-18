"""
Evolutionary Architecture Search - Phase 7 Track 3

Population-based evolutionary search with tournament selection,
mutation, crossover, and elitism. Discovers Pareto-optimal
encoder/decoder architectures.

Algorithm:
    1. Initialize random population
    2. Evaluate fitness (Pareto score from evaluator)
    3. Select parents via tournament
    4. Create children via crossover + mutation
    5. Replace worst with children (elitism preserves top-k)
    6. Repeat until convergence or budget exhausted

Research Basis:
    - NSGA-II: Multi-objective evolutionary optimization
    - Regularized Evolution (Real et al., 2019): aging + tournament for NAS

Usage:
    search = EvolutionarySearch(
        population_size=50,
        generations=100,
    )
    best = search.run()
    print(f"Best architecture: {best.to_dict()}")
"""

import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np

from sigmalang.nas.search_space import SearchSpace, ArchitectureConfig
from sigmalang.nas.evaluator import ArchitectureEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for evolutionary search."""
    population_size: int = 50
    generations: int = 100
    tournament_size: int = 5
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    elitism_count: int = 5        # Top-k preserved across generations
    seed: int = 42
    early_stop_patience: int = 20  # Stop if no improvement for N generations


@dataclass
class Individual:
    """An individual in the population."""
    config: ArchitectureConfig
    fitness: float = 0.0
    evaluation: Optional[EvaluationResult] = None
    age: int = 0


@dataclass
class SearchHistory:
    """History of the evolutionary search."""
    generations: List[Dict[str, Any]] = field(default_factory=list)
    best_fitness_per_gen: List[float] = field(default_factory=list)
    avg_fitness_per_gen: List[float] = field(default_factory=list)
    total_evaluations: int = 0
    total_time_s: float = 0.0
    converged: bool = False
    best_config: Optional[Dict[str, Any]] = None


class EvolutionarySearch:
    """
    Evolutionary search for optimal SigmaLang architectures.

    Uses tournament selection + uniform crossover + point mutation
    with elitism to explore the architecture search space.
    """

    def __init__(
        self,
        search_config: Optional[SearchConfig] = None,
        evaluator: Optional[ArchitectureEvaluator] = None,
        callback: Optional[Callable[[int, 'Individual'], None]] = None,
    ):
        self.config = search_config or SearchConfig()
        self.space = SearchSpace(seed=self.config.seed)
        self.evaluator = evaluator or ArchitectureEvaluator()
        self.callback = callback

        self._rng = np.random.RandomState(self.config.seed)
        self.population: List[Individual] = []
        self.history = SearchHistory()
        self._best_ever: Optional[Individual] = None

    def _make_id(self, config: ArchitectureConfig) -> str:
        """Generate a unique ID for an architecture."""
        d = config.to_dict()
        key = str(sorted(d.items()))
        return hashlib.md5(key.encode()).hexdigest()[:10]

    def _init_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.config.population_size):
            config = self.space.sample_random()
            config.architecture_id = self._make_id(config)
            individual = Individual(config=config)
            self.population.append(individual)

    def _evaluate_population(self) -> None:
        """Evaluate all unevaluated individuals."""
        for ind in self.population:
            if ind.evaluation is None:
                ind.evaluation = self.evaluator.evaluate(ind.config)
                ind.fitness = ind.evaluation.pareto_score
                self.history.total_evaluations += 1

    def _tournament_select(self) -> Individual:
        """Select parent via tournament selection."""
        candidates = self._rng.choice(
            len(self.population),
            size=min(self.config.tournament_size, len(self.population)),
            replace=False
        )
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[best]

    def _create_offspring(self) -> ArchitectureConfig:
        """Create one offspring via selection, crossover, and mutation."""
        parent_a = self._tournament_select()

        if self._rng.random() < self.config.crossover_rate:
            parent_b = self._tournament_select()
            child = self.space.crossover(parent_a.config, parent_b.config)
        else:
            child = ArchitectureConfig.from_dict(parent_a.config.to_dict())
            child.parent_id = parent_a.config.architecture_id

        # Mutation
        child = self.space.mutate(child, self.config.mutation_rate)
        child.architecture_id = self._make_id(child)

        return child

    def _replace_worst(self, children: List[Individual]) -> None:
        """Replace worst individuals with children, preserving elites."""
        # Sort by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Protect elites
        n_replace = min(len(children), len(self.population) - self.config.elitism_count)
        if n_replace <= 0:
            return

        # Replace worst n_replace
        for i in range(n_replace):
            self.population[-(i + 1)] = children[i]

    def run(self) -> ArchitectureConfig:
        """
        Run the evolutionary search.

        Returns the best architecture found.
        """
        start_time = time.time()

        logger.info(
            f"Starting NAS: pop={self.config.population_size}, "
            f"gens={self.config.generations}"
        )

        # Initialize
        self._init_population()
        self._evaluate_population()

        best_fitness = -1.0
        patience_counter = 0

        for gen in range(self.config.generations):
            # Create offspring
            n_children = self.config.population_size - self.config.elitism_count
            children = []
            for _ in range(n_children):
                child_config = self._create_offspring()
                child_eval = self.evaluator.evaluate(child_config)
                child = Individual(
                    config=child_config,
                    fitness=child_eval.pareto_score,
                    evaluation=child_eval,
                )
                children.append(child)
                self.history.total_evaluations += 1

            # Age existing population
            for ind in self.population:
                ind.age += 1

            # Replace worst
            self._replace_worst(children)

            # Track stats
            fitnesses = [ind.fitness for ind in self.population]
            gen_best = max(fitnesses)
            gen_avg = sum(fitnesses) / len(fitnesses)

            self.history.best_fitness_per_gen.append(gen_best)
            self.history.avg_fitness_per_gen.append(gen_avg)

            # Update best ever
            best_ind = max(self.population, key=lambda ind: ind.fitness)
            if best_ind.fitness > best_fitness:
                best_fitness = best_ind.fitness
                self._best_ever = best_ind
                patience_counter = 0
            else:
                patience_counter += 1

            # Log progress
            if gen % 10 == 0 or gen == self.config.generations - 1:
                logger.info(
                    f"Gen {gen:3d}: best={gen_best:.4f}, avg={gen_avg:.4f}, "
                    f"evals={self.history.total_evaluations}"
                )

            # Callback
            if self.callback:
                self.callback(gen, best_ind)

            # Record generation
            self.history.generations.append({
                'generation': gen,
                'best_fitness': round(gen_best, 4),
                'avg_fitness': round(gen_avg, 4),
                'best_id': best_ind.config.architecture_id,
            })

            # Early stopping
            if patience_counter >= self.config.early_stop_patience:
                logger.info(f"Early stopping at generation {gen} (no improvement for {patience_counter} gens)")
                self.history.converged = True
                break

        self.history.total_time_s = time.time() - start_time
        self.history.best_config = self._best_ever.config.to_dict() if self._best_ever else None

        logger.info(
            f"Search complete: {self.history.total_evaluations} evaluations "
            f"in {self.history.total_time_s:.1f}s, "
            f"best fitness={best_fitness:.4f}"
        )

        return self._best_ever.config if self._best_ever else self.space.sample_random()

    def get_pareto_front(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-N architectures from the Pareto front.

        Returns sorted by fitness descending.
        """
        ranked = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        results = []
        for ind in ranked[:top_n]:
            entry = ind.config.to_dict()
            entry['fitness'] = round(ind.fitness, 4)
            if ind.evaluation:
                entry['evaluation'] = ind.evaluation.to_dict()
            results.append(entry)
        return results

    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of the search process."""
        return {
            'generations_completed': len(self.history.generations),
            'total_evaluations': self.history.total_evaluations,
            'total_time_s': round(self.history.total_time_s, 1),
            'converged': self.history.converged,
            'best_fitness': round(max(self.history.best_fitness_per_gen) if self.history.best_fitness_per_gen else 0, 4),
            'best_config': self.history.best_config,
            'search_space': self.space.get_search_space_summary(),
        }
