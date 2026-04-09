"""
Ornstein-Uhlenbeck Noise for DDPG Exploration.

Adapted from PyTorch-DDPG-Stock-Trading (Flood Sung).
Standard OU formula: dX = theta * (mu - X) * dt + sigma * dW

Used for exploration in continuous action spaces (portfolio weights).
"""

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process for continuous action exploration.

    The OU process provides temporally correlated exploration noise,
    which is important for continuous control tasks like portfolio allocation.
    It smoothly varies over time while still providing exploration.
    """

    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize OU noise.

        Args:
            action_dimension: Dimension of the action space
            mu: Mean of the OU process (drift target)
            theta: Rate of mean reversion
            sigma: Volatility (standard deviation of Wiener process)
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.seeds = 0
        self.reset()

    def reset(self):
        """Reset the OU process to mean."""
        self.state = np.ones(self.action_dimension) * self.mu
        self.seeds += 1
        np.random.seed(self.seeds)

    def noise(self):
        """Generate next noise sample using OU process.

        Returns:
            Current state after applying OU update
        """
        x = self.state
        # OU formula: dx = theta * (mu - x) * dt + sigma * dW
        # Here dt = 1, dW ~ N(0, 1)
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def std_noise(self, mu, sigma):
        """Generate standard Gaussian noise with given parameters.

        Args:
            mu: Mean of the Gaussian
            sigma: Standard deviation of the Gaussian

        Returns:
            Gaussian noise sample
        """
        self.seeds += 1
        np.random.seed(self.seeds)
        return np.random.normal(mu, sigma, len(self.state))


if __name__ == '__main__':
    # Test OU noise
    ou = OUNoise(1, sigma=0.1 / 50)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.hist(np.array(states).ravel())
    plt.title("OU Noise Distribution")
    plt.show()
