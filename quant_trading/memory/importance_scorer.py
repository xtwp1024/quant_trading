"""
Importance scoring for memory entries.

Importance reflects how critical or valuable a piece of information is,
independent of when it occurred. It is:
    1. Initialized stochastically when a memory is created (layer-dependent distribution)
    2. Updated upward when the memory is accessed with positive feedback

The layer-specific initialization distributions encode the cognitive role:
    - Short-term: skews toward lower scores (transient information)
    - Mid-term:  skews toward medium scores (working knowledge)
    - Long-term: skews toward higher scores (enduring insights)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Callable


class ImportanceScorer(ABC):
    """Abstract base for importance score initialization and update."""

    @abstractmethod
    def initialize(self) -> float:
        """Return an initial importance score for a new memory entry."""
        pass

    def update(self, importance_score: float, access_counter: int) -> float:
        """
        Update importance score based on access history.

        The original FinMem formula: importance += access_counter * 5

        Args:
            importance_score: Current importance score.
            access_counter: Number of times this memory was accessed.

        Returns:
            Updated importance score.
        """
        return importance_score + access_counter * 5


class I_SampleInitialization_Short(ImportanceScorer):
    """
    Short-term memory importance initialization.

    Distribution: P(50)=0.50, P(70)=0.45, P(90)=0.05
    Most new short-term entries start with low importance (transient).
    """

    def initialize(self) -> float:
        probabilities = [0.5, 0.45, 0.05]
        scores = [50.0, 70.0, 90.0]
        return np.random.choice(scores, p=probabilities)


class I_SampleInitialization_Mid(ImportanceScorer):
    """
    Mid-term memory importance initialization.

    Distribution: P(40)=0.05, P(60)=0.80, P(80)=0.15
    Most new mid-term entries start with medium importance.
    """

    def initialize(self) -> float:
        probabilities = [0.05, 0.8, 0.15]
        scores = [40.0, 60.0, 80.0]
        return np.random.choice(scores, p=probabilities)


class I_SampleInitialization_Long(ImportanceScorer):
    """
    Long-term memory importance initialization.

    Distribution: P(40)=0.05, P(60)=0.15, P(80)=0.80
    Most new long-term entries start with high importance (verified knowledge).
    """

    def initialize(self) -> float:
        probabilities = [0.05, 0.15, 0.8]
        scores = [40.0, 60.0, 80.0]
        return np.random.choice(scores, p=probabilities)


class ImportanceScorerByLayer:
    """
    Factory for layer-appropriate ImportanceScorer instances.

    Maps layer name -> appropriate initialization strategy.
    """

    _LAYER_MAP: Dict[str, type] = {
        "short": I_SampleInitialization_Short,
        "mid": I_SampleInitialization_Mid,
        "long": I_SampleInitialization_Long,
        "perceptual": I_SampleInitialization_Short,
        "reflection": I_SampleInitialization_Long,
    }

    def __init__(self, layer: str):
        if layer not in self._LAYER_MAP:
            raise ValueError(
                f"Invalid layer '{layer}'. Must be one of: {list(self._LAYER_MAP.keys())}"
            )
        self._scorer: ImportanceScorer = self._LAYER_MAP[layer]()

    def initialize(self) -> float:
        return self._scorer.initialize()

    def update(self, importance_score: float, access_counter: int) -> float:
        return self._scorer.update(importance_score, access_counter)


def get_importance_scorer(layer: str) -> ImportanceScorerByLayer:
    """Convenience factory function."""
    return ImportanceScorerByLayer(layer)
