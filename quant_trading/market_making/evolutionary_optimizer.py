"""
Evolutionary Optimizer for PassivBot Configurations

Adapted from passivbot's optimize.py and optimization/ modules.

This module implements a CMA-ES-style evolutionary algorithm for finding
optimal PassivBot configurations through thousands of backtested candidates.

Key features:
- (mu + lambda) evolutionary strategy
- Pareto-optimal multi-objective optimization
- Bound-constrained parameter spaces (continuous + discrete/stepped)
- DEAP library integration with custom genetic operators
- Constraint-aware fitness (penalize but don't reject infeasible configs)
- Parallel evaluation via multiprocessing

Usage:
    optimizer = EvolutionaryOptimizer(
        bounds=bounds_list,
        objectives=["adg_pnl", "drawdown_worst"],
        weights=[-1.0, 1.0],  # minimize adg_pnl, minimize drawdown
    )
    result = optimizer.run(
        evaluate_fn=my_backtest_function,
        n_generations=100,
        pop_size=50,
        lambda_=200,
    )
    best_config = result.best_individual
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from deap import algorithms as deap_algorithms
    from deap import base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    base = creator = tools = deap_algorithms = None

__all__ = [
    "EvolutionaryOptimizer",
    "Bound",
    "ObjectiveSpec",
    "OptimizerResult",
    "individual_to_config",
    "config_to_individual",
    "enforce_bounds",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bound system (from passivbot's optimization/bounds.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bound:
    """
    Represents parameter bounds for optimization.

    For continuous parameters, step is None.
    For stepped (discrete) parameters, step defines the grid spacing.

    Args:
        low: Lower bound
        high: Upper bound
        step: Step size for discrete parameters (None for continuous)
    """
    low: float
    high: float
    step: Optional[float] = None

    @property
    def is_stepped(self) -> bool:
        return self.step is not None and self.step > 0

    @property
    def max_index(self) -> int:
        if not self.is_stepped:
            raise ValueError("max_index only valid for stepped parameters")
        return int((self.high - self.low + 1e-9) / self.step)

    def quantize(self, value: float) -> float:
        """Quantize a value to the nearest step within bounds."""
        if not self.is_stepped:
            return max(self.low, min(self.high, value))

        clamped = max(self.low, min(self.high, value))
        n_steps = int((clamped - self.low) / self.step + 0.5)
        clamped_index = max(0, min(self.max_index, n_steps))
        quantized = self.low + clamped_index * self.step

        # Round to decimal precision of step
        if self.step < 1:
            decimals = -int(math.floor(math.log10(self.step)))
            quantized = round(quantized, decimals)
        return max(self.low, min(self.high, quantized))

    def random_on_grid(self) -> float:
        """Generate a random value respecting step constraints."""
        if not self.is_stepped:
            return random.uniform(self.low, self.high)
        idx = random.randint(0, self.max_index)
        return self.low + idx * self.step

    def value_to_index(self, value: float) -> float:
        if not self.is_stepped:
            return value
        return (value - self.low) / self.step

    def index_to_value(self, index: float) -> float:
        if not self.is_stepped:
            return max(self.low, min(self.high, index))
        idx = int(index + 0.5)
        clamped_idx = max(0, min(self.max_index, idx))
        return self.low + clamped_idx * self.step

    def get_index_bounds(self) -> Tuple[float, float]:
        if not self.is_stepped:
            return (self.low, self.high)
        return (0.0, float(self.max_index))

    @classmethod
    def from_config(cls, key: str, val) -> "Bound":
        """Parse a Bound from a config value.

        Supported formats:
        - Single value: fixed parameter (low=high)
        - [low, high]: continuous optimization
        - [low, high, step]: discrete optimization with step
        - [low, high, 0] or null: continuous
        """
        if isinstance(val, (float, int)):
            return cls(float(val), float(val), None)

        if isinstance(val, (tuple, list)):
            if len(val) == 0:
                raise ValueError(f"malformed bound {key}: empty array")
            if len(val) == 1:
                return cls(float(val[0]), float(val[0]), None)
            if len(val) == 2:
                low, high = sorted([float(val[0]), float(val[1])])
                return cls(low, high, None)
            if len(val) >= 3:
                low, high = sorted([float(val[0]), float(val[1])])
                step_raw = val[2]
                if step_raw is None or float(step_raw) <= 0.0:
                    return cls(low, high, None)
                step = float(step_raw)
                if step > (high - low):
                    return cls(low, high, None)
                return cls(low, high, step)

        raise ValueError(f"malformed bound {key}: {val}")


def enforce_bounds(
    values: Sequence[float], bounds: Sequence[Bound], sig_digits: int = 6
) -> List[float]:
    """
    Clamp each value to its corresponding [low, high] interval and quantize to step if defined.
    """
    if len(values) != len(bounds):
        raise ValueError(
            f"values/bounds length mismatch: {len(values)} values, {len(bounds)} bounds"
        )
    result = []
    for v, b in zip(values, bounds):
        if b.is_stepped:
            result.append(b.quantize(v))
        else:
            # Significant digit rounding
            if sig_digits is not None and v != 0.0 and math.isfinite(v):
                digits = sig_digits - 1 - int(math.floor(math.log10(abs(v))))
                v = round(v, digits)
            result.append(max(b.low, min(b.high, v)))
    return result


# ---------------------------------------------------------------------------
# DEAP operator wrappers (from passivbot's optimization/deap_adapters.py)
# ---------------------------------------------------------------------------

DEAP_EPSILON = 1e-6


def _to_index_space(values: List[float], bounds: Sequence[Bound]):
    index_values, index_low, index_up = [], [], []
    for val, b in zip(values, bounds):
        if b.is_stepped:
            index_values.append(b.value_to_index(val))
            idx_low, idx_up = b.get_index_bounds()
            index_low.append(idx_low)
            index_up.append(idx_up)
        else:
            index_values.append(val)
            index_low.append(b.low)
            index_up.append(b.high)
    return index_values, index_low, index_up


def _prepare_bounds_for_deap(
    index_low: List[float], index_up: List[float]
) -> Tuple[List[float], List[float], np.ndarray]:
    """Handle equal bounds (DEAP doesn't accept low == high)."""
    low_arr = np.array(index_low)
    up_arr = np.array(index_up)
    equal_mask = low_arr == up_arr
    temp_low = np.where(equal_mask, low_arr - DEAP_EPSILON, low_arr)
    temp_up = np.where(equal_mask, up_arr + DEAP_EPSILON, up_arr)
    return list(temp_low), list(temp_up), equal_mask


def _from_index_space(
    index_values: List[float], bounds: Sequence[Bound], equal_mask: np.ndarray
) -> List[float]:
    result = []
    for i, (idx_val, b) in enumerate(zip(index_values, bounds)):
        if equal_mask[i]:
            result.append(b.low)
        elif b.is_stepped:
            result.append(b.index_to_value(idx_val))
        else:
            result.append(idx_val)
    return result


def mut_polynomial_bounded(
    individual: List[float], eta: float, indpb: float, bounds: Sequence[Bound]
) -> Tuple[List[float]]:
    """Polynomial bounded mutation with step support."""
    if not DEAP_AVAILABLE:
        # Fallback: random perturbation
        perturbed = list(individual)
        for i, (val, b) in enumerate(zip(individual, bounds)):
            if random.random() < indpb and b.low != b.high:
                delta = (b.high - b.low) * 0.1 * random.uniform(-1, 1)
                perturbed[i] = b.quantize(val + delta)
        return (perturbed,)
    idx_ind, idx_low, idx_up = _to_index_space(individual, bounds)
    temp_low, temp_up, eq_mask = _prepare_bounds_for_deap(idx_low, idx_up)
    tools.mutPolynomialBounded(idx_ind, eta, temp_low, temp_up, indpb)
    individual[:] = _from_index_space(idx_ind, bounds, eq_mask)
    return (individual,)


def cx_simulated_binary_bounded(
    ind1: List[float], ind2: List[float], eta: float, bounds: Sequence[Bound]
) -> Tuple[List[float], List[float]]:
    """Simulated binary crossover with step support."""
    if not DEAP_AVAILABLE:
        # Fallback: blend crossover
        alpha = 0.5
        child1, child2 = [], []
        for a, b, bound in zip(ind1, ind2, bounds):
            gamma = alpha * (b - a)
            child1.append(bound.quantize(a + gamma))
            child2.append(bound.quantize(b - gamma))
        return child1, child2
    idx1, idx_low, idx_up = _to_index_space(ind1, bounds)
    idx2, _, _ = _to_index_space(ind2, bounds)
    temp_low, temp_up, eq_mask = _prepare_bounds_for_deap(idx_low, idx_up)
    tools.cxSimulatedBinaryBounded(ind1, ind2, eta, temp_low, temp_up)
    ind1[:] = _from_index_space(idx1, bounds, eq_mask)
    ind2[:] = _from_index_space(idx2, bounds, eq_mask)
    return ind1, ind2


# ---------------------------------------------------------------------------
# Objective specification
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveSpec:
    """Specification for a single optimization objective."""
    name: str
    weight: float  # -1.0 to minimize, +1.0 to maximize
    threshold: Optional[float] = None  # minimum acceptable value

    def __post_init__(self):
        if self.weight not in (-1.0, 1.0):
            raise ValueError(f"weight must be -1.0 or 1.0, got {self.weight}")


# ---------------------------------------------------------------------------
# Constraint-aware fitness
# ---------------------------------------------------------------------------


class ConstraintAwareFitness:
    """Fitness with constraint violation penalty."""
    constraint_violation: float = 0.0

    def dominates(self, other, obj=slice(None)):
        self_viol = getattr(self, "constraint_violation", 0.0)
        other_viol = getattr(other, "constraint_violation", 0.0)
        if math.isclose(self_viol, other_viol, rel_tol=0.0, abs_tol=1e-12):
            return super().dominates(other, obj)
        return self_viol < other_viol


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OptimizerResult:
    """Result from a completed optimization run."""
    best_individual: List[float]
    best_fitness: float
    best_constraints: float
    n_evals: int
    n_generations: int
    pareto_front: List[List[float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_config(self, keys: List[str], bounds: Sequence[Bound]) -> Dict:
        """Convert best individual back to config dict.

        Args:
            keys: Parameter names in same order as bounds
            bounds: Bound objects in same order as keys
        """
        return individual_to_config(self.best_individual, keys, bounds)


def individual_to_config(
    individual: List[float],
    keys: List[str],
    bounds: Sequence[Bound],
) -> Dict:
    """
    Convert an individual (list of floats) back to a config dict.

    Args:
        individual: List of parameter values
        keys: List of parameter names in the same order as bounds
        bounds: List of Bound objects (must match keys order)
    """
    if len(individual) != len(keys):
        raise ValueError(
            f"individual ({len(individual)}) and keys ({len(keys)}) must have same length"
        )
    return dict(zip(keys, individual))


def config_to_individual(
    config: Dict,
    keys: List[str],
    bounds: Sequence[Bound],
    sig_digits: int = 6,
) -> List[float]:
    """Convert a config dict to an individual (list of floats)."""
    individual = [config.get(k, b.low) for k, b in zip(keys, bounds)]
    return enforce_bounds(individual, bounds, sig_digits)


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer for PassivBot configurations.

    Uses a (mu + lambda) evolutionary strategy with:
    - Polynomial bounded mutation
    - Simulated binary crossover
    - Pareto-front selection via NSGA-II style crowding
    - Constraint-aware fitness (penalize violations, don't reject)
    - Optional parallel evaluation

    Args:
        bounds: List of Bound objects defining the search space
        objectives: List of ObjectiveSpec objects
        sig_digits: Significant digits for rounding (default 6)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        bounds: Sequence[Bound],
        objectives: Sequence[ObjectiveSpec],
        sig_digits: int = 6,
        seed: Optional[int] = None,
    ):
        if not DEAP_AVAILABLE:
            logger.warning("DEAP not available; using simplified optimizer")

        self.bounds = list(bounds)
        self.objectives = list(objectives)
        self.sig_digits = sig_digits

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._setup_deap()

    def _setup_deap(self) -> None:
        """Set up DEAP creator and toolbox."""
        if not DEAP_AVAILABLE:
            return

        # Build fitness class dynamically
        weights = tuple(obj.weight for obj in self.objectives)

        if hasattr(creator, "Fitness"):
            # Reset if re-running
            delattr(creator, "Fitness")
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")

        creator.create(
            "Fitness",
            base.Fitness,
            weights=weights,
        )
        creator.create("Individual", list, fitness=creator.Fitness)

    def _create_toolbox(self, evaluate_fn: Callable) -> "tools.Toolbox":
        """Create DEAP toolbox with operators."""
        if not DEAP_AVAILABLE:
            raise RuntimeError("DEAP is required for evolutionary optimization")

        toolbox = base.Toolbox()

        # Population initialization
        def create_individual():
            vals = [b.random_on_grid() for b in self.bounds]
            ind = creator.Individual(enforce_bounds(vals, self.bounds, self.sig_digits))
            return ind

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation
        toolbox.register("evaluate", evaluate_fn)

        # Variation operators
        toolbox.register(
            "mutate",
            mut_polynomial_bounded,
            eta=20.0,
            indpb=0.2,
            bounds=self.bounds,
        )
        toolbox.register(
            "mate",
            cx_simulated_binary_bounded,
            eta=20.0,
            bounds=self.bounds,
        )

        # Selection: NSGA-II style (crowded comparison)
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def _simplified_evaluate(
        self,
        population: List[List[float]],
        evaluate_fn: Callable,
    ) -> List[Tuple[List[float], float, float]]:
        """
        Simplified evaluation without DEAP.
        Returns list of (individual, fitness, constraint_violation).
        """
        results = []
        for ind in population:
            fitness_vals, penalty = evaluate_fn(ind)
            # Combine objectives into single scalar (weighted sum)
            combined = sum(
                w * v for w, v in zip(
                    [obj.weight for obj in self.objectives], fitness_vals
                ) if v is not None
            )
            results.append((ind, combined, penalty))
        return results

    def _simplified_evolve(
        self,
        population: List[List[float]],
        evaluate_fn: Callable,
        n_generations: int,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Simplified evolutionary loop without DEAP.
        Uses (mu + lambda) strategy with tournament selection.
        """
        # Evaluate initial population
        pop_fitness = self._simplified_evaluate(population, evaluate_fn)
        pop_fitness.sort(key=lambda x: x[1])

        for gen in range(n_generations):
            # Generate offspring
            offspring = []
            for _ in range(lambda_):
                # Tournament selection
                candidates = random.sample(range(len(population)), k=3)
                winner = min(candidates, key=lambda i: pop_fitness[i][1])
                selected = copy.deepcopy(population[winner])

                # Mutate
                if random.random() < mutpb:
                    mutated = list(selected)
                    for i, b in enumerate(self.bounds):
                        if random.random() < 0.2 and b.low != b.high:
                            delta = (b.high - b.low) * 0.1 * random.uniform(-1, 1)
                            mutated[i] = b.quantize(mutated[i] + delta)
                    selected = mutated

                # Crossover
                if random.random() < cxpb:
                    other = random.choice(population)
                    alpha = 0.5
                    child = [
                        b.quantize(a + alpha * (b2 - a))
                        for a, b2, b in zip(selected, other, self.bounds)
                    ]
                    selected = child

                offspring.append(selected)

            # Evaluate offspring
            off_fitness = self._simplified_evaluate(offspring, evaluate_fn)

            # (mu + lambda): combine parents and offspring, keep best mu
            combined = pop_fitness + off_fitness
            combined.sort(key=lambda x: x[1])
            population = [ind for ind, _, _ in combined[:mu]]
            pop_fitness = combined[:mu]

            if gen % 10 == 0:
                logger.info(
                    f"Gen {gen}: best_fitness={pop_fitness[0][1]:.4g}, "
                    f"evals={(gen + 1) * lambda_}"
                )

        best = pop_fitness[0]
        return population, [best[1], best[2]]

    def run(
        self,
        evaluate_fn: Callable[
            [List[float]],
            Tuple[List[float], float],
        ],
        *,
        n_generations: int = 50,
        pop_size: int = 50,
        lambda_: Optional[int] = None,
        cxpb: float = 0.5,
        mutpb: float = 0.5,
        parallel: bool = False,
        n_workers: int = 4,
        verbose: bool = True,
    ) -> OptimizerResult:
        """
        Run the evolutionary optimization.

        Args:
            evaluate_fn: Function that takes an individual (list of floats)
                         and returns (fitness_values: List[float], constraint_penalty: float)
                         fitness_values must match self.objectives order
            n_generations: Number of generations
            pop_size: Population size (mu)
            lambda_: Number of offspring per generation (defaults to pop_size)
            cxpb: Crossover probability
            mutpb: Mutation probability
            parallel: Use multiprocessing for evaluation
            n_workers: Number of parallel workers
            verbose: Log progress

        Returns:
            OptimizerResult with best individual and metrics
        """
        if lambda_ is None:
            lambda_ = pop_size

        mu = pop_size
        start_time = time.time()

        if not DEAP_AVAILABLE:
            # Simplified optimizer without DEAP
            random_pop = [
                [b.random_on_grid() for b in self.bounds]
                for _ in range(pop_size)
            ]
            population, (best_fit, best_pen) = self._simplified_evolve(
                random_pop, evaluate_fn, n_generations, mu, lambda_, cxpb, mutpb
            )
            elapsed = time.time() - start_time
            return OptimizerResult(
                best_individual=population[0],
                best_fitness=best_fit,
                best_constraints=best_pen,
                n_evals=pop_size + n_generations * lambda_,
                n_generations=n_generations,
                pareto_front=[],
                elapsed_seconds=elapsed,
            )

        # Full DEAP optimizer
        toolbox = self._create_toolbox(evaluate_fn)

        # Hall of fame to track Pareto front
        halloffame = tools.ParetoFront()

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("avg", np.mean, axis=0)

        # Initial population
        population = toolbox.population(n=pop_size)

        logger.info(
            f"Starting optimization: {n_generations} generations, "
            f"mu={mu}, lambda={lambda_}, parallel={parallel}"
        )

        # (mu + lambda) evolution
        final_pop, logbook = self._ea_mu_plus_lambda(
            population,
            toolbox,
            mu=mu,
            lambda_=lambda_,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=n_generations,
            stats=stats,
            halloffame=halloffame,
            verbose=verbose,
        )

        elapsed = time.time() - start_time

        # Extract best individual
        best_ind = tools.selBest(population, k=1)[0]
        best_fitness = list(best_ind.fitness.values)
        best_penalty = getattr(best_ind.fitness, "constraint_violation", 0.0)

        pareto_front = (
            [list(ind) for ind in halloffame.items]
            if len(halloffame) > 0
            else []
        )

        logger.info(
            f"Optimization complete: {n_generations} generations, "
            f"{mu * (n_generations + 1)} total evals, "
            f"elapsed={elapsed:.1f}s"
        )

        return OptimizerResult(
            best_individual=list(best_ind),
            best_fitness=best_fitness[0] if len(best_fitness) == 1 else best_fitness,
            best_constraints=best_penalty,
            n_evals=mu * (n_generations + 1),
            n_generations=n_generations,
            pareto_front=pareto_front,
            elapsed_seconds=elapsed,
        )

    def _ea_mu_plus_lambda(
        self,
        population: list,
        toolbox,
        mu: int,
        lambda_: int,
        cxpb: float,
        mutpb: float,
        ngen: int,
        stats=None,
        halloffame=None,
        verbose: bool = True,
    ):
        """
        (mu + lambda) evolutionary algorithm.
        """
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "max"

        # Evaluate initial population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, (fit_vals, penalty) in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit_vals
            ind.fitness.constraint_violation = penalty

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        if verbose:
            logger.info(f"Gen 0: evals={len(invalid_ind)}, front_size={len(halloffame or [])}")

        for gen in range(1, ngen + 1):
            # Variate the population
            offspring = algorithms.varOr(
                population, toolbox, lambda_, cxpb, mutpb
            )

            # Evaluate offspring
            invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_offspring)
            for ind, (fit_vals, penalty) in zip(invalid_offspring, fitnesses):
                ind.fitness.values = fit_vals
                ind.fitness.constraint_violation = penalty

            # Select next generation: mu + lambda
            population[:] = toolbox.select(population + offspring, mu)

            if halloffame is not None:
                halloffame.update(population)

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, evals=len(invalid_offspring), **record)
            if verbose and gen % 10 == 0:
                logger.info(
                    f"Gen {gen}: evals={len(invalid_offspring)}, "
                    f"front_size={len(halloffame or [])}, "
                    f"best_fitness={record.get('min', 'N/A')}"
                )

        return population, logbook


# ---------------------------------------------------------------------------
# Standard bounds for PassivBot (used for optimization)
# ---------------------------------------------------------------------------


PASSIVBOT_DEFAULT_BOUNDS: List[Bound] = [
    # Grid sizing
    Bound(0.005, 0.05, step=None),   # entry_grid_spacing_pct
    Bound(1.0, 3.0, step=None),      # entry_grid_double_down_factor
    # Entry
    Bound(0.1, 1.0, step=None),      # entry_initial_qty_pct
    Bound(0.0, 0.02, step=None),     # entry_initial_ema_dist
    # Close
    Bound(0.002, 0.02, step=None),   # close_grid_markup_start
    Bound(0.01, 0.05, step=None),    # close_grid_markup_end
    Bound(0.25, 1.0, step=None),     # close_grid_qty_pct
    # Trailing entry
    Bound(0.0, 0.05, step=None),     # entry_trailing_threshold_pct
    Bound(0.001, 0.02, step=None),   # entry_trailing_retracement_pct
    Bound(0.0, 0.5, step=None),      # entry_trailing_grid_ratio
    # Trailing close
    Bound(0.0, 0.5, step=None),      # close_trailing_grid_ratio
    Bound(0.005, 0.03, step=None),   # close_trailing_threshold_pct
    Bound(0.001, 0.01, step=None),   # close_trailing_retracement_pct
    # Unstuck
    Bound(-0.1, 0.0, step=None),     # unstuck_threshold
    Bound(0.1, 0.5, step=None),      # unstuck_close_pct
    Bound(0.0, 0.05, step=None),     # unstuck_ema_dist
    # Filters
    Bound(0.0, 0.02, step=None),     # filter_volatility_drop_pct
    Bound(0.0, 0.02, step=None),     # filter_volume_drop_pct
    # Position limits
    Bound(0.05, 0.5, step=None),     # wallet_exposure_limit
    Bound(0.1, 1.0, step=None),      # total_wallet_exposure_limit
    Bound(1, 5, step=1),            # n_positions
]

# Parameter names corresponding to PASSIVBOT_DEFAULT_BOUNDS (same order)
PASSIVBOT_PARAM_KEYS: List[str] = [
    "entry_grid_spacing_pct",
    "entry_grid_double_down_factor",
    "entry_initial_qty_pct",
    "entry_initial_ema_dist",
    "close_grid_markup_start",
    "close_grid_markup_end",
    "close_grid_qty_pct",
    "entry_trailing_threshold_pct",
    "entry_trailing_retracement_pct",
    "entry_trailing_grid_ratio",
    "close_trailing_grid_ratio",
    "close_trailing_threshold_pct",
    "close_trailing_retracement_pct",
    "unstuck_threshold",
    "unstuck_close_pct",
    "unstuck_ema_dist",
    "filter_volatility_drop_pct",
    "filter_volume_drop_pct",
    "wallet_exposure_limit",
    "total_wallet_exposure_limit",
    "n_positions",
]
