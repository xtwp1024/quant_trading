"""
Exponential decay for memory retention.

Applies two simultaneous decay mechanisms:
    1. Recency score: exponential decay based on time delta
    2. Importance score: multiplicative decay per time step

The compound score (importance + recency) drives memory prioritization
and determines which entries survive cleanup or jump to other layers.
"""

import numpy as np
from typing import Tuple


class ExponentialDecay:
    """
    Applies exponential decay to recency and importance scores.

    Formulas:
        new_recency_score = exp(-(delta + 1) / recency_factor)
        new_importance_score = importance_score * importance_factor

    Args:
        recency_factor: Controls how quickly recency decays.
                        Larger values → slower recency decay.
        importance_factor: Multiplicative decay per step (0 < x <= 1).
                            Larger values → slower importance decay.
    """

    def __init__(
        self,
        recency_factor: float = 10.0,
        importance_factor: float = 0.988,
    ):
        self.recency_factor = recency_factor
        self.importance_factor = importance_factor

    def __call__(
        self, importance_score: float, delta: float
    ) -> Tuple[float, float, float]:
        """
        Apply one step of exponential decay.

        Args:
            importance_score: Current importance score of the memory entry.
            delta: Number of time steps since last access.

        Returns:
            Tuple of (new_recency_score, new_importance_score, new_delta).
            new_delta is incremented by 1 (representing the passage of one time step).
        """
        delta += 1
        new_recency_score = np.exp(-(delta / self.recency_factor))
        new_importance_score = importance_score * self.importance_factor

        return new_recency_score, new_importance_score, delta

    def decay_batch(
        self,
        entries: list,
        score_key: str = "importance_score",
        delta_key: str = "delta",
    ) -> list:
        """
        Apply decay to a list of memory entries in-place.

        Args:
            entries: List of memory entry dicts.
            score_key: Key for the importance score in each entry.
            delta_key: Key for the delta counter in each entry.

        Returns:
            The same list mutated in-place.
        """
        for entry in entries:
            recency, importance, delta = self(
                importance_score=entry[score_key],
                delta=entry[delta_key],
            )
            entry["recency_score"] = recency
            entry[score_key] = importance
            entry[delta_key] = delta
        return entries
