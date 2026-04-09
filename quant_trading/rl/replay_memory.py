"""
ReplayMemory - Circular replay buffer for DRQN agent.

Stores transitions (state, action, next_state, reward) in a fixed-capacity
circular buffer. Supports random sampling for batch training.

Based on the original DRQN_Stock_Trading ReplayMemory implementation.
"""

from collections import namedtuple
import random

Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward"),
)


class ReplayMemory:
    """
    Circular replay buffer for experience replay.

    Stores up to `capacity` transitions. When full, overwrites oldest
    entries. Supports random sampling for batch training.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        """
        Append a transition to the buffer.

        If buffer is not full, appends None as a placeholder first.
        Stores the transition at the current position, then advances
        the position circularly.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Transition namedtuples.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.memory)
