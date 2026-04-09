"""
Recency scoring for memory entries.

Recency reflects how recently a memory was created or accessed.
It decays exponentially with time (delta), representing the natural
forgetting curve. Freshly created or accessed memories score close to 1.0,
while older memories decay toward 0.

FinMem uses R_ConstantInitialization: new memories always start at 1.0.
The delta counter tracks elapsed time steps since last access, and
the ExponentialDecay class computes the actual recency score.
"""


class RecencyScorer:
    """
    Recency score management.

    Provides:
        - Initialization: new memories always start at 1.0 (maximum recency)
        - Reset: restore recency to 1.0 when a memory is "refreshed" (e.g., jumped up)

    The actual decay computation is delegated to ExponentialDecay.
    """

    def __init__(self):
        self._initial_value: float = 1.0

    def initialize(self) -> float:
        """
        Return the initial recency score for a new memory entry.

        FinMem convention: always 1.0 for new entries.
        """
        return self._initial_value

    def reset(self) -> float:
        """
        Return the recency score after a memory is refreshed.

        Same as initialize() — memory regains full recency upon re-access
        or upon jumping to a higher-priority layer.
        """
        return self._initial_value

    @property
    def initial_value(self) -> float:
        return self._initial_value
