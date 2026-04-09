"""Individual — absorbed from GeneTrader.

Represents a single candidate solution in the genetic algorithm:
- Typed genes (Int / Decimal / Categorical / Boolean)
- Trading pairs allocation
- Fitness score
"""
from typing import List, Dict, Any, Optional
import random
import copy


class Individual:
    """A single candidate solution in the genetic algorithm population."""

    def __init__(
        self,
        genes: List[Any],
        trading_pairs: List[str],
        param_types: List[Dict[str, Any]],
    ):
        self.genes = genes
        self.trading_pairs = trading_pairs
        self.param_types = param_types
        self.fitness: Optional[float] = None

    @classmethod
    def create_random(
        cls,
        parameters: List[Dict[str, Any]],
        all_pairs: List[str],
        num_pairs: Optional[int],
    ) -> "Individual":
        """Create a random individual.

        Args:
            parameters: Parameter definitions (each must have 'type', 'start', 'end')
            all_pairs: Available trading pairs
            num_pairs: Number of pairs per individual (None = all pairs)
        """
        genes = []
        for param in parameters:
            param_type = param["type"]
            if param_type == "Int":
                if param.get("name") == "max_open_trades":
                    min_value = max(1, int(param["start"]))
                    value = random.randint(min_value, int(param["end"]))
                else:
                    value = random.randint(int(param["start"]), int(param["end"]))
            elif param_type == "Decimal":
                value = random.uniform(param["start"] + 1e-10, param["end"] - 1e-10)
                value = round(value, param["decimal_places"])
            elif param_type == "Categorical":
                value = random.choice(param["options"])
            elif param_type == "Boolean":
                value = random.choice([True, False])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            genes.append(value)

        if num_pairs is not None:
            trading_pairs = random.sample(all_pairs, min(num_pairs, len(all_pairs)))
        else:
            trading_pairs = all_pairs.copy()
        return cls(genes, trading_pairs, parameters)

    def constrain_genes(self, parameters: List[Dict[str, Any]]) -> None:
        """Clamp gene values to their valid ranges."""
        for i, param in enumerate(parameters):
            if i >= len(self.genes):
                break
            param_type = param["type"]
            if param_type == "Int":
                if param.get("name") == "max_open_trades":
                    min_value = max(1, int(param["start"]))
                    self.genes[i] = int(
                        max(min_value, min(param["end"], self.genes[i]))
                    )
                else:
                    self.genes[i] = int(
                        max(param["start"], min(param["end"], self.genes[i]))
                    )
            elif param_type == "Decimal":
                self.genes[i] = round(
                    max(param["start"], min(param["end"], self.genes[i])),
                    param["decimal_places"],
                )

    def after_genetic_operation(self, parameters: List[Dict[str, Any]]) -> None:
        """Apply constraints after crossover or mutation."""
        self.constrain_genes(parameters)

    def copy(self) -> "Individual":
        """Create a deep copy of this individual."""
        return copy.deepcopy(self)

    def mutate_trading_pairs(
        self, all_pairs: List[str], mutation_rate: float
    ) -> None:
        """Mutate trading pairs using set-based efficient logic."""
        if not self.trading_pairs:
            return

        current_pairs = set(self.trading_pairs)
        all_pairs_set = set(all_pairs)
        available_pairs = list(all_pairs_set - current_pairs)

        for i in range(len(self.trading_pairs)):
            if random.random() < mutation_rate and available_pairs:
                old_pair = self.trading_pairs[i]
                new_pair = random.choice(available_pairs)
                current_pairs.discard(old_pair)
                current_pairs.add(new_pair)
                available_pairs.remove(new_pair)
                available_pairs.append(old_pair)
                self.trading_pairs[i] = new_pair

        self.trading_pairs = list(current_pairs)
