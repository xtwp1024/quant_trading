"""Population — absorbed from GeneTrader.

Manages a collection of Individual objects for the genetic algorithm.
"""
from typing import List, Dict, Optional

from quant_trading.gene_lab.individual import Individual


class Population:
    """A collection of individuals in the genetic algorithm."""

    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @classmethod
    def create_random(
        cls,
        size: int,
        parameters: Dict,
        trading_pairs: List[str],
        num_pairs: Optional[int],
    ) -> "Population":
        """Create a population of random individuals."""
        return cls(
            [
                Individual.create_random(parameters, trading_pairs, num_pairs)
                for _ in range(size)
            ]
        )

    def get_best(self) -> Individual:
        """Return the individual with the highest fitness."""
        if not self.individuals:
            raise ValueError("Cannot get best from empty population")
        return max(
            self.individuals,
            key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"),
        )

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)
