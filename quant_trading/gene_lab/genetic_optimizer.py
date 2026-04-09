"""Genetic Optimizer — absorbed from GeneTrader, adapted for quant_trading.

Combines GeneTrader's genetic algorithm (selection, crossover, mutation, elitism,
diversity maintenance) with quant_trading's WalkforwardAnalyzer for anti-overfitting
validation.

Usage
-----
```python
from quant_trading.gene_lab import (
    GeneticOptimizer,
    WalkForwardValidator,
    Individual,
    Population,
)

# Define parameter space
parameters = [
    {"name": "rsi_period",     "type": "Int",       "start": 5,   "end": 30},
    {"name": "rsi_entry",      "type": "Int",       "start": 20,  "end": 40},
    {"name": "rsi_exit",       "type": "Int",       "start": 60,  "end": 80},
    {"name": "ma_type",        "type": "Categorical", "options": ["SMA", "EMA", "WMA"]},
    {"name": "use_volume",     "type": "Boolean"},
]

all_pairs = ["ETH/USDT", "BTC/USDT"]

opt = GeneticOptimizer(
    parameters=parameters,
    all_pairs=all_pairs,
    population_size=30,
    generations=10,
    mutation_prob=0.2,
    crossover_prob=0.7,
    tournament_size=3,
)
best = opt.optimize()
```
"""
import gc
import random
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from quant_trading.gene_lab.individual import Individual
from quant_trading.gene_lab.population import Population
from quant_trading.gene_lab.operators import (
    crossover,
    mutate,
    select_tournament,
    select_with_diversity,
    maintain_diversity,
    calculate_population_diversity,
)
from quant_trading.gene_lab.evaluation import fitness_function, parse_backtest_results
from quant_trading.backtester.walkforward import WalkforwardAnalyzer

logger = logging.getLogger("GeneLab.GeneticOptimizer")


class GeneticOptimizer:
    """Genetic algorithm optimizer with anti-overfitting walk-forward validation.

    Features
    --------
    - Tournament selection with optional diversity-aware variant
    - Single-point crossover with optional trading-pair exchange
    - Typed gene mutation (Int/Decimal/Categorical/Boolean)
    - Elitism: best individual always survives
    - Diversity maintenance: boosts mutation when population converges
    - Walk-forward analysis integration for out-of-sample testing
    """

    def __init__(
        self,
        parameters: List[Dict[str, Any]],
        all_pairs: List[str],
        population_size: int = 30,
        generations: int = 20,
        mutation_prob: float = 0.2,
        crossover_prob: float = 0.7,
        tournament_size: int = 3,
        elitism_count: int = 1,
        enable_diversity_selection: bool = False,
        diversity_weight: float = 0.3,
        diversity_threshold: float = 0.1,
        num_pairs: Optional[int] = None,
        backtest_weeks: int = 30,
        min_trades: int = 15,
    ):
        """Initialize the genetic optimizer.

        Args:
            parameters: Parameter definitions for strategy genes.
                        Each dict must contain: type (Int/Decimal/Categorical/Boolean),
                        start, end. Int/Decimal may include decimal_places, name.
            all_pairs: Available trading pairs.
            population_size: Number of individuals per generation.
            generations: Number of evolution rounds.
            mutation_prob: Per-gene mutation probability.
            crossover_prob: Probability of crossover per pair.
            tournament_size: Tournament size for selection.
            elitism_count: Number of best individuals to preserve unchanged.
            enable_diversity_selection: Use diversity-aware tournament selection.
            diversity_weight: Weight given to diversity in selection (0-1).
            diversity_threshold: Min population diversity before boost mutation.
            num_pairs: Number of pairs per individual (None = all pairs).
            backtest_weeks: Weeks of data for fitness evaluation.
            min_trades: Minimum trades required for valid fitness.
        """
        self.parameters = parameters
        self.all_pairs = all_pairs
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.enable_diversity_selection = enable_diversity_selection
        self.diversity_weight = diversity_weight
        self.diversity_threshold = diversity_threshold
        self.num_pairs = num_pairs
        self.backtest_weeks = backtest_weeks
        self.min_trades = min_trades
        self.best_individual: Optional[Individual] = None

    def _create_population(
        self, initial_individuals: List[Individual] = None
    ) -> Population:
        """Create a new random population, optionally seeded with individuals."""
        pop = Population.create_random(
            size=self.population_size,
            parameters=self.parameters,
            trading_pairs=self.all_pairs,
            num_pairs=self.num_pairs,
        )
        if initial_individuals:
            pop.individuals.extend(initial_individuals)
        return pop

    def _evaluate_individual(
        self,
        individual: Individual,
        df_train,
        price_col: str = "close",
    ) -> float:
        """Evaluate a single individual's fitness.

        Uses WalkforwardAnalyzer to compute metrics, then applies the
        anti-overfitting fitness_function.

        Args:
            individual: The individual to evaluate.
            df_train: Training DataFrame (OHLCV).
            price_col: Price column name.

        Returns:
            Fitness score (negative = disqualified).
        """
        try:
            wf = WalkforwardAnalyzer(
                train_window=max(20, len(df_train) // 3),
                test_window=max(5, len(df_train) // 6),
                step=max(1, len(df_train) // 10),
                metric="sharpe",
            )
            result = wf.analyze(
                df_train,
                strategy_func=lambda df, **params: self._signals_from_params(
                    df, individual, params, price_col
                ),
            )

            if result.test_metrics.empty:
                return -1.0

            # Map WalkforwardAnalyzer output to evaluation.py format
            parsed = {
                "total_profit_percent": float(
                    result.test_metrics["total_return"].mean()
                ),
                "win_rate": 0.5,  # Not computed by WalkforwardAnalyzer
                "max_drawdown": float(
                    result.test_metrics["max_drawdown"].mean()
                ),
                "sharpe_ratio": float(result.test_metrics["sharpe"].mean()),
                "sortino_ratio": float(result.test_metrics["sortino"].mean()),
                "profit_factor": 1.0,  # Not computed by WalkforwardAnalyzer
                "daily_avg_trades": 2.0,  # Placeholder
                "avg_trade_duration": 360,  # Placeholder: 6 hours
                "total_trades": int(
                    result.test_metrics["total_return"].count()
                    * self.backtest_weeks
                    / 30
                ),
            }

            fitness = fitness_function(
                parsed,
                generation=0,
                strategy_name=f"GA_{id(individual)}",
                timeframe="1d",
                num_parameters=len(self.parameters),
                backtest_weeks=self.backtest_weeks,
            )
            return fitness

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return -1.0

    def _signals_from_params(
        self,
        df,
        individual: Individual,
        params: dict,
        price_col: str,
    ):
        """Generate trading signals from gene values + walkforward params.

        This is a simple RSI-based signal generator driven by the
        individual's gene values. Override for custom signal generation.
        """
        import numpy as np

        close = df[price_col].values
        n = len(close)

        # Decode genes: individual.genes → strategy params
        rsi_period = int(individual.genes[0]) if len(individual.genes) > 0 else 14
        rsi_entry = float(individual.genes[1]) if len(individual.genes) > 1 else 30.0
        use_volume = bool(individual.genes[3]) if len(individual.genes) > 3 else True

        # Compute RSI
        deltas = np.diff(close, prepend=close[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gains = self._rolling_mean(gains, rsi_period)
        avg_losses = self._rolling_mean(losses, rsi_period)
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - 100 / (1 + rs)

        signals = np.zeros(n)
        for i in range(rsi_period, n):
            if rsi[i] < rsi_entry:
                signals[i] = 1  # Long
            elif rsi[i] > (100 - rsi_entry):
                signals[i] = -1  # Short/close

        return signals

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean with same semantics as MyTT _rolling_mean."""
        n = len(arr)
        result = np.empty(n)
        result[:] = np.nan
        if window < 1:
            return result
        for i in range(window - 1, n):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result

    def optimize(
        self,
        df_train,
        initial_individuals: List[Individual] = None,
        price_col: str = "close",
    ) -> Individual:
        """Run genetic algorithm optimization.

        Args:
            df_train: Training DataFrame (OHLCV).
            initial_individuals: Optional seeds for the initial population.
            price_col: Price column for signal generation.

        Returns:
            The best Individual found.
        """
        # Trim population size for seed
        population = self._create_population(initial_individuals)

        for gen in range(self.generations):
            # Evaluate fitness for all individuals
            for ind in population.individuals:
                if ind.fitness is None:
                    ind.fitness = self._evaluate_individual(
                        ind, df_train, price_col
                    )

            # Filter valid individuals
            valid = [
                ind
                for ind in population.individuals
                if ind.fitness is not None and ind.fitness > 0
            ]
            if not valid:
                valid = [
                    ind
                    for ind in population.individuals
                    if ind.fitness is not None
                ]
                if not valid:
                    logger.warning(f"Gen {gen+1}: No valid individuals, skipping.")
                    continue

            best_gen = max(valid, key=lambda i: i.fitness)
            logger.info(
                f"Gen {gen+1}/{self.generations}: "
                f"Best fitness={best_gen.fitness:.4f}, "
                f"Valid={len(valid)}/{len(population)}"
            )

            if (
                self.best_individual is None
                or best_gen.fitness > self.best_individual.fitness
            ):
                self.best_individual = best_gen.copy()

            # Diversity logging
            if self.enable_diversity_selection:
                div = calculate_population_diversity(population.individuals)
                logger.info(f"  Population diversity: {div:.4f}")

            # --- SELECTION ---
            offspring: List[Individual] = []

            # Elitism: carry best forward unchanged
            for _ in range(self.elitism_count):
                elite = max(valid, key=lambda i: i.fitness).copy()
                offspring.append(elite)

            # Fill remaining slots via tournament selection
            while len(offspring) < self.population_size:
                if self.enable_diversity_selection and len(offspring) > 1:
                    ref = offspring[-1]
                    selected = select_with_diversity(
                        valid,
                        self.tournament_size,
                        diversity_weight=self.diversity_weight,
                        reference_individual=ref,
                    )
                else:
                    selected = select_tournament(valid, self.tournament_size)
                offspring.append(selected.copy())

            # --- CROSSOVER ---
            for i in range(self.elitism_count, len(offspring) - 1, 2):
                if random.random() < self.crossover_prob:
                    offspring[i], offspring[i + 1] = crossover(
                        offspring[i], offspring[i + 1], with_pair=True
                    )
                    offspring[i].after_genetic_operation(self.parameters)
                    offspring[i + 1].after_genetic_operation(self.parameters)

            # --- MUTATION (skip elite) ---
            for ind in offspring[self.elitism_count:]:
                mutate(ind, self.mutation_prob)
                ind.after_genetic_operation(self.parameters)

            # --- DIVERSITY MAINTENANCE ---
            if self.enable_diversity_selection:
                mutations = maintain_diversity(
                    offspring[self.elitism_count :],
                    min_diversity=self.diversity_threshold,
                    mutation_boost=self.mutation_prob * 2,
                )
                if mutations > 0:
                    logger.info(f"  Diversity mutations applied: {mutations}")

            # Replace population (trim to population_size)
            population.individuals = offspring[: self.population_size]
            gc.collect()

        logger.info(f"Optimization complete. Best fitness: {self.best_individual.fitness:.4f}")
        return self.best_individual

    def optimize_with_walk_forward(
        self,
        df,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 21,
        price_col: str = "close",
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run optimization with walk-forward cross-validation.

        Args:
            df: Full OHLCV DataFrame.
            train_window: Training window in bars.
            test_window: Test window in bars.
            step: Bars to step forward each fold.
            price_col: Price column name.

        Returns:
            Tuple of (fold_results, composite_fitness)
        """
        wf = WalkforwardAnalyzer(
            train_window=train_window,
            test_window=test_window,
            step=step,
            metric="sharpe",
        )

        n = len(df)
        fold_results = []

        for fold_start in range(0, n - train_window - test_window, step):
            train_end = fold_start + train_window
            test_end = train_end + test_window

            if test_end > n:
                break

            train_df = df.iloc[fold_start:train_end]
            test_df = df.iloc[train_end:test_end]

            logger.info(
                f"Walk-forward fold: train[{fold_start}:{train_end}], "
                f"test[{train_end}:{test_end}]"
            )

            # Optimize on training fold
            best = self.optimize(train_df, price_col=price_col)

            # Evaluate on test fold
            test_fitness = self._evaluate_individual(best, test_df, price_col)

            fold_results.append(
                {
                    "fold": len(fold_results),
                    "train_fitness": best.fitness,
                    "test_fitness": test_fitness,
                    "best_individual": best,
                }
            )

            logger.info(
                f"  Train fitness={best.fitness:.4f}, "
                f"Test fitness={test_fitness:.4f}"
            )

        # Compute composite
        if fold_results:
            test_scores = [r["test_fitness"] for r in fold_results]
            composite = (
                0.4 * sum(test_scores) / len(test_scores)
                + 0.2 * min(test_scores)
                + 0.2 * (1.0 / (1.0 + (max(test_scores) - min(test_scores))))
                + 0.2 * 1.0
            )
        else:
            composite = float("-inf")

        logger.info(f"Walk-forward composite fitness: {composite:.4f}")
        return fold_results, composite

    def get_best_individual(self) -> Optional[Individual]:
        """Return the best individual found so far."""
        return self.best_individual
