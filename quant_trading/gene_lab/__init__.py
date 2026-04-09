"""Gene Lab — Strategy Gene Bank & Genetic Algorithm Optimizer.

Absorbed from GeneTrader (D:/Hive/Data/trading_repos/GeneTrader):

Modules
-------
- GeneBank:           JSON-backed gene vault (ENTRY/EXIT logic snippets)
- GeneExtractor:      AST parser for extracting genes from strategy files
- StrategyBreeder:    Code-generation-based strategy synthesizer
- Individual:         Typed gene individual (Int/Decimal/Categorical/Boolean)
- Population:         GA population manager
- operators:          GA operators — crossover, mutate, select, diversity
- GeneticOptimizer:   Full GA with elitism, diversity maintenance, WFV
- evaluation:         Freqtrade backtest parser + anti-overfitting fitness
- walk_forward:       WalkForwardValidator (rolling/expanding/anchored)

The GeneLab layer bridges gene extraction from existing strategies
with genetic optimization to breed and evolve new trading strategies.
"""
from quant_trading.gene_lab.gene_bank import GeneBank
from quant_trading.gene_lab.gene_extractor import GeneExtractor
from quant_trading.gene_lab.strategy_breeder import StrategyBreeder
from quant_trading.gene_lab.individual import Individual
from quant_trading.gene_lab.population import Population
from quant_trading.gene_lab.operators import (
    crossover,
    mutate,
    select_tournament,
    select_with_diversity,
    maintain_diversity,
    calculate_genetic_distance,
    calculate_population_diversity,
)
from quant_trading.gene_lab.genetic_optimizer import GeneticOptimizer
from quant_trading.gene_lab.evaluation import (
    fitness_function,
    parse_backtest_results,
)
from quant_trading.gene_lab.walk_forward import (
    WalkForwardValidator,
    ValidationPeriod,
    create_validator_from_settings,
)

__all__ = [
    # Original GeneLab components
    "GeneBank",
    "GeneExtractor",
    "StrategyBreeder",
    # GeneTrader GA core
    "Individual",
    "Population",
    "crossover",
    "mutate",
    "select_tournament",
    "select_with_diversity",
    "maintain_diversity",
    "calculate_genetic_distance",
    "calculate_population_diversity",
    # Optimizer
    "GeneticOptimizer",
    # Evaluation
    "fitness_function",
    "parse_backtest_results",
    # Walk-forward
    "WalkForwardValidator",
    "ValidationPeriod",
    "create_validator_from_settings",
]
