"""
DDPG Market Environment for Stock Trading.

Adapted from PyTorch-DDPG-Stock-Trading (JohsuaWu1997).
Simple stock market simulation for DDPG training with 3-timestep state window.

Key features:
- 3-timestep state window (prices, volumes)
- Continuous portfolio allocation (Box action space)
- Transaction costs (buy/sell rates)
- Portfolio tracking
"""

import numpy as np
import torch
from gymnasium import spaces


class DDPGMarketEnv:
    """Stock market environment for DDPG training.

    State: 3 timesteps x stock features (price, volume, etc.)
    Action: Portfolio weights (normalized, sum to 1)
    """

    def __init__(self, data, seed=0, asset=1000000.0, unit=100):
        """Initialize market environment.

        Args:
            data: List of [buy_data, sell_data, amount_data], each shape (timesteps, stocks+1)
                  First column is index/timestamp, remaining columns are stock features
            seed: Random seed for reproducibility
            asset: Initial portfolio value
            unit: Minimum trading unit (lot size)
        """
        self.asset = asset
        self.unit = unit
        self.rate = 5e-4  # Buy transaction rate
        self.short_rate = 1e-3  # Sell transaction rate (higher due to short selling cost)
        self.rd_seed = seed

        # Process data: assume first column is index, rest are stock features
        buy_data = np.array(data[0])
        sell_data = np.array(data[1])
        amount_data = np.array(data[2])

        # Shape: (timesteps, stocks, 3) for 3 timesteps of [price_buy, price_sell, amount]
        self.data = torch.tensor(
            np.stack([buy_data[:, 1:].T, sell_data[:, 1:].T, amount_data[:, 1:].T], axis=2),
            dtype=torch.float32
        )  # (stocks, features, timesteps) -> will be transposed

        # Actually: data[i] is (timesteps, stocks+1), we want (timesteps, stocks, 3)
        # Build proper state: (timesteps, stocks, 3) where 3 = [sell_price, buy_price, amount]
        self.data = np.zeros((buy_data.shape[0], buy_data.shape[1] - 1, 3), dtype=np.float32)
        self.data[:, :, 0] = buy_data[:, 1:]   # sell price
        self.data[:, :, 1] = sell_data[:, 1:]  # buy price
        self.data[:, :, 2] = amount_data[:, 1:]  # amount
        self.data = torch.tensor(self.data, dtype=torch.float32)

        # self.data is now (timesteps, stocks, 3)
        self.stock_number = self.data.shape[1]
        self.sample_size = self.data.shape[0]

        # Gymnasium-compatible spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_number,))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.stock_number,)
        )

        self._seed(seed)

    def _seed(self, seed):
        """Set random seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        """Reset environment to initial state.

        Returns:
            Initial state (3, stock_number) tensor
        """
        self.n_step = 0
        self.state = self.data[self.n_step, :, :].T  # Transpose to (3, stock_number)
        self.position = torch.zeros(self.stock_number)
        self.cash = torch.tensor(self.asset)
        self.portfolio = torch.tensor(self.asset)
        self.rewards = torch.zeros(self.sample_size)
        self.cost = torch.zeros(self.sample_size)
        self.success = []
        self.available_cash = torch.zeros(self.sample_size).fill_(self.asset)
        self.book = []
        return self.state

    def step(self, position: torch.Tensor):
        """Execute one step in the environment.

        Args:
            position: Target portfolio position (units per stock)

        Returns:
            next_state, reward, done, info
        """
        self.n_step += 1
        self.state = self.data[self.n_step, :, :].T  # Transpose to (3, stock_number)

        amount = position - self.position
        price = self.state[1, :].view(-1)  # buy price
        price[amount < 0] = self.state[0, :][amount < 0]  # sell price for short positions

        # Transaction costs
        transaction_buy = torch.sum((amount * price)[amount > 0] * self.rate)
        transaction_sell = -torch.sum((amount * price)[amount < 0] * (self.short_rate + self.rate))

        cost_buy = torch.sum((amount * price)[amount > 0])
        cost_sell = torch.sum((amount * price)[amount < 0])

        # Check if we have enough cash
        if self.cash < transaction_buy + cost_buy:
            self.success.append(False)
            self.cost[self.n_step] = transaction_sell
            self.position[amount < 0] = position[amount < 0]
            self.cash -= cost_sell + transaction_sell
        else:
            self.success.append(True)
            self.cost[self.n_step] = transaction_sell + transaction_buy
            self.position = position
            self.cash -= cost_sell + transaction_sell + cost_buy + transaction_buy

        # Update portfolio value
        portfolio = self.cash + torch.sum(self.state[0, :] * self.position)
        reward = portfolio - self.portfolio

        self.portfolio = portfolio
        self.rewards[self.n_step] = portfolio
        self.available_cash[self.n_step] = self.cash
        self.book.append(amount.numpy().ravel().tolist())

        done = self.n_step >= self.sample_size - 1
        return self.state, reward, done, {}

    def plot(self, path=None):
        """Plot portfolio performance vs benchmark.

        Args:
            path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards.numpy().ravel(), label='Portfolio')
        if path:
            plt.savefig(path + '.png')
        plt.close()

    def render(self, mode='human', path=None):
        """Render environment results.

        Args:
            mode: Rendering mode
            path: Optional path to save CSV results
        """
        if path:
            import pandas as pd
            result = np.array([
                self.rewards.numpy().ravel(),
                self.cost.numpy().ravel(),
                self.available_cash.numpy().ravel(),
                self.success]).T
            pd.DataFrame(result, columns=['portfolio', 'transaction', 'cash', 'success']).to_csv(
                path + '-result.csv')
            pd.DataFrame(self.book).to_csv(path + '-book.csv')

    def close(self):
        """Clean up environment resources."""
        pass


def create_sample_market_data(timesteps=100, stocks=5, seed=42):
    """Create synthetic market data for testing DDPG.

    Args:
        timesteps: Number of timesteps to generate
        stocks: Number of stocks in the market
        seed: Random seed

    Returns:
        List of [buy_data, sell_data, amount_data] numpy arrays
    """
    np.random.seed(seed)

    # Generate random walk prices
    base_prices = 10 + np.random.randn(timesteps, stocks) * 0.5
    prices = np.cumsum(base_prices, axis=0)

    # Add timestamp column
    timestamps = np.arange(timesteps).reshape(-1, 1)
    buy_data = np.concatenate([timestamps, prices * 1.001], axis=1)  # Slightly higher buy
    sell_data = np.concatenate([timestamps, prices * 0.999], axis=1)  # Slightly lower sell
    amount_data = np.concatenate([timestamps, np.abs(np.random.randn(timesteps, stocks)) * 1000 + 100], axis=1)

    return [buy_data, sell_data, amount_data]


if __name__ == '__main__':
    # Test the environment
    data = create_sample_market_data(timesteps=50, stocks=3)
    env = DDPGMarketEnv(data, seed=0)

    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    total_reward = 0
    done = False
    while not done:
        # Random action
        action = torch.rand(env.stock_number) * env.unit * 10
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Final portfolio value: {env.portfolio}")
    print(f"Total reward: {total_reward}")
