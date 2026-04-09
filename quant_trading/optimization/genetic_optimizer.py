"""遗传算法参数优化器 / Genetic Algorithm Parameter Optimizer.

流程:
1. 初始化种群 (随机参数组合)
2. 评估适应度 (Sharpe/回测结果)
3. 选择 (精英保留 + 锦标赛)
4. 交叉 (均匀/单点)
5. 变异 (高斯/均匀)
6. 重复直到收敛

Genetic Algorithm for parameter optimization.
Workflow: init population -> evaluate fitness -> select -> crossover -> mutate -> repeat.
"""

from __future__ import annotations

__all__ = ["GeneticOptimizer"]

import random
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class GeneticOptimizerError(Exception):
    """Base exception for genetic optimizer errors."""
    pass


class FitnessNotDefinedError(GeneticOptimizerError):
    """Raised when fitness() is called but not overridden in subclass."""
    pass


# ---------------------------------------------------------------------------
# GeneticOptimizer
# ---------------------------------------------------------------------------
class GeneticOptimizer:
    """遗传算法参数优化器 / Genetic Algorithm Parameter Optimizer.

    Attributes:
        param_space: Dictionary defining parameter search space.
                    Format: {param_name: (min_val, max_val, type)}
                    type: 'int', 'float', or 'choice' (list of options)
        population_size: Number of individuals in population.
        n_generations: Maximum number of generations to evolve.
        crossover_rate: Probability of crossover (0.0 - 1.0).
        mutation_rate: Probability of mutation per gene (0.0 - 1.0).
        elite_ratio: Ratio of top individuals to preserve unchanged (0.0 - 1.0).
        seed: Random seed for reproducibility.

    Example:
        >>> class MyFitness(GeneticOptimizer):
        ...     def fitness(self, params: dict) -> float:
        ...         # Your backtest logic here
        ...         return sharpe_ratio
        ...
        >>> optimizer = MyFitness(
        ...     param_space={
        ...         "fast_period": (5, 50, "int"),
        ...         "slow_period": (20, 200, "int"),
        ...         "threshold": (0.001, 0.1, "float"),
        ...     },
        ...     population_size=50,
        ...     n_generations=100,
        ... )
        >>> result = optimizer.optimize()
        >>> print(result["best_params"])
    """

    def __init__(
        self,
        param_space: dict,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        seed: int = 42,
    ):
        """Initialize the genetic optimizer.

        Args:
            param_space: Search space definition.
                        Example: {"period": (10, 100, "int"), "alpha": (0.0, 1.0, "float")}
            population_size: Population size per generation.
            n_generations: Number of generations to run.
            crossover_rate: Crossover probability [0, 1].
            mutation_rate: Per-gene mutation probability [0, 1].
            elite_ratio: Top fraction preserved unchanged [0, 1].
            seed: Random seed for reproducibility.
        """
        self.param_space = param_space
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.seed = seed

        self._random = random.Random(seed)
        self._np_random = np.random.RandomState(seed)

        self.n_elite = max(1, int(population_size * elite_ratio))

        # History tracking
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Fitness (to be overridden by subclass)
    # ------------------------------------------------------------------
    def fitness(self, params: dict) -> float:
        """评估适应度 — 由子类实现 / Evaluate fitness for given parameters.

        This method must be overridden in a subclass. Return a scalar
        fitness value — higher is better.

        Args:
            params: Dictionary of parameter values, e.g.
                    {"fast_period": 12, "slow_period": 26, "threshold": 0.05}

        Returns:
            Fitness score (higher is better). For backtest results,
            typically return Sharpe ratio, total return, or similar.

        Raises:
            FitnessNotDefinedError: If not overridden in subclass.
        """
        raise FitnessNotDefinedError(
            "fitness() must be overridden in a subclass. "
            "Implement your backtest or metric evaluation here."
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_population(self) -> list[dict]:
        """Initialize random population based on param_space."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for name, (lo, hi, ptype) in self.param_space.items():
                if ptype == "int":
                    individual[name] = self._random.randint(int(lo), int(hi))
                elif ptype == "float":
                    individual[name] = self._random.uniform(float(lo), float(hi))
                elif ptype == "choice":
                    individual[name] = self._random.choice(list(lo))  # lo is the list of choices
                else:
                    # Default to float
                    individual[name] = self._random.uniform(float(lo), float(hi))
            population.append(individual)
        return population

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select(
        self,
        population: list[dict],
        fitnesses: list[float],
    ) -> list[dict]:
        """选择算子 / Selection operator.

        Uses elitism (preserve top individuals) + tournament selection.

        Args:
            population: Current population list.
            fitnesses: Fitness scores for each individual.

        Returns:
            Selected population list.
        """
        # Sort by fitness (descending — higher is better)
        sorted_pairs = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)

        # Elite preservation
        elite = [ind for _, ind in sorted_pairs[: self.n_elite]]

        # Tournament selection for the rest
        remaining_size = self.population_size - self.n_elite
        selected = list(elite)

        for _ in range(remaining_size):
            # Binary tournament
            i1 = self._random.randint(0, len(population) - 1)
            i2 = self._random.randint(0, len(population) - 1)
            winner = population[i1] if fitnesses[i1] >= fitnesses[i2] else population[i2]
            selected.append(winner)

        return selected

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------
    def crossover(
        self,
        parent1: dict,
        parent2: dict,
    ) -> tuple[dict, dict]:
        """交叉算子 / Crossover operator.

        Uses uniform crossover — each gene has equal chance from either parent.

        Args:
            parent1: First parent individual.
            parent2: Second parent individual.

        Returns:
            Tuple of (child1, child2) after crossover.
        """
        if self._random.random() > self.crossover_rate:
            # No crossover — return copies
            return dict(parent1), dict(parent2)

        child1, child2 = {}, {}
        for key in parent1:
            if self._random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def mutate(self, individual: dict) -> dict:
        """变异算子 / Mutation operator.

        Uses Gaussian mutation for float parameters and uniform mutation
        for int/choice parameters.

        Args:
            individual: Individual to mutate.

        Returns:
            Mutated individual (may be same as input if no mutation).
        """
        mutated = dict(individual)
        for name, (lo, hi, ptype) in self.param_space.items():
            if self._random.random() < self.mutation_rate:
                if ptype == "int":
                    # Gaussian mutation, clamped to bounds
                    current = mutated[name]
                    sigma = (int(hi) - int(lo)) * 0.1
                    new_val = int(round(current + self._np_random.normal(0, sigma)))
                    mutated[name] = max(int(lo), min(int(hi), new_val))
                elif ptype == "float":
                    # Gaussian mutation
                    current = mutated[name]
                    sigma = (float(hi) - float(lo)) * 0.1
                    new_val = current + self._np_random.normal(0, sigma)
                    mutated[name] = max(float(lo), min(float(hi), new_val))
                elif ptype == "choice":
                    # Random choice mutation
                    choices = list(lo)  # lo is the list of choices
                    mutated[name] = self._random.choice(choices)

        return mutated

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------
    def _reproduce(self, population: list[dict], fitnesses: list[float]) -> list[dict]:
        """Create next generation through selection, crossover, and mutation."""
        selected = self.select(population, fitnesses)

        next_gen = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            c1, c2 = self.crossover(p1, p2)
            next_gen.append(self.mutate(c1))
            next_gen.append(self.mutate(c2))

        # Ensure population size is maintained
        while len(next_gen) < self.population_size:
            idx = self._random.randint(0, len(selected) - 1)
            next_gen.append(self.mutate(dict(selected[idx])))

        return next_gen[: self.population_size]

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    def optimize(
        self,
        callback: Optional[Callable[[int, dict, float], None]] = None,
    ) -> dict:
        """执行遗传算法优化 / Run genetic algorithm optimization.

        Args:
            callback: Optional callback(gen, best_params, best_fitness)
                     called after each generation.

        Returns:
            Dictionary with:
            - best_params: Best parameter combination found.
            - best_fitness: Fitness score of best params.
            - history: List of dicts with per-generation stats.

        Example:
            >>> result = optimizer.optimize()
            >>> print(result["best_params"])
            >>> print(result["best_fitness"])
        """
        # Initialize
        population = self._initialize_population()
        best_params = None
        best_fitness = -np.inf
        self.history = []

        for gen in range(self.n_generations):
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                try:
                    f = self.fitness(ind)
                except Exception as e:
                    # Log the exception for debugging but penalize failed evaluations
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Fitness evaluation failed for {ind}: {e}"
                    )
                    f = -np.inf  # Penalize failed evaluations
                fitnesses.append(f)

            # Track best
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_params = dict(population[gen_best_idx])

            # Record history
            self.history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": float(np.mean(fitnesses)),
                "std_fitness": float(np.std(fitnesses)),
                "best_params": dict(best_params),
            })

            # Callback
            if callback is not None:
                callback(gen, best_params, best_fitness)

            # Progress output
            if gen % max(1, self.n_generations // 10) == 0:
                print(
                    f"Gen {gen:4d} | "
                    f"Best fitness: {best_fitness:12.4f} | "
                    f"Avg fitness: {np.mean(fitnesses):12.4f} | "
                    f"Best params: {best_params}"
                )

            # Check early convergence (optional)
            if gen > 10:
                recent = [h["best_fitness"] for h in self.history[-5:]]
                if max(recent) == min(recent) == best_fitness:
                    print(f"Converged at generation {gen}. Early stopping.")
                    break

            # Reproduce next generation
            if gen < self.n_generations - 1:
                population = self._reproduce(population, fitnesses)

        return {
            "best_params": best_params,
            "best_fitness": best_fitness,
            "history": self.history,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.seed = seed
        self._random = random.Random(seed)
        self._np_random = np.random.RandomState(seed)
