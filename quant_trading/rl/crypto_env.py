"""
CryptoTradingEnv — Gymnasium-compatible BTC/USDT trading environment.

Adapted from RL-Crypto-Trading-Bot (Vnadh/RL-Crypto-Trading-Bot).
https://github.com/Vnadh/RL-Crypto-Trading-Bot

Action space: Discrete(3) — 0=hold, 1=buy (10%), 2=sell (10% of holdings)
Observation space: DataFrame features + [balance, btc_held]
Episode terminates when all data is consumed.

Performance benchmarks (on unseen BTC data):
    DQN  — Sharpe: 2.22, Returns: 1.28%, Win Rate: 51.90%
    PPO  — Sharpe: 1.85, Returns: 0.97%, Win Rate: 51.76%
    A2C  — Sharpe: 1.82, Returns: 0.99%, Win Rate: 51.78%
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency Trading Environment for Reinforcement Learning.

    Parameters
    ----------
    df : pd.DataFrame
        Price/feature dataframe (must contain 'close' column).
    initial_balance : float, default 10000
        Starting USDT balance.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10_000.0) -> None:
        super().__init__()

        # Data
        self.df = df.astype(np.float32)
        self.features = self.df.columns.tolist()

        # Trading state
        self.current_step = 0
        self.initial_balance = np.float32(initial_balance)
        self.balance = self.initial_balance
        self.btc_held = np.float32(0.0)

        # Spaces
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.features) + 2,),
            dtype=np.float32,
        )

        # Rendering state
        self.fig: Optional[plt.Figure] = None
        self.price_ax = None
        self.balance_ax = None
        self.price_line = None
        self.balance_line = None
        self.render_data: dict = {
            "prices": [],
            "balances": [],
            "trades": [],  # (step, action_type, amount, price)
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = np.float32(0.0)
        self.render_data = {"prices": [], "balances": [], "trades": []}
        return self._next_observation(), {}

    def step(self, action: int):
        self._take_action(action)
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        reward = self._calculate_reward()
        info = {
            "step": self.current_step,
            "balance": float(self.balance),
            "btc_held": float(self.btc_held),
            "net_worth": float(self.net_worth),
            "price": float(self.current_price),
        }

        return self._next_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_observation(self) -> np.ndarray:
        features = self.df.iloc[self.current_step].values.astype(np.float32)
        portfolio = np.array([self.balance, self.btc_held], dtype=np.float32)
        return np.concatenate([features, portfolio])

    def _take_action(self, action: int) -> None:
        current_price = float(self.df.iloc[self.current_step]["close"])

        if action == 1:  # Buy — spend 10% of balance on BTC
            btc_bought = (float(self.balance) * 0.1) / current_price
            self.btc_held += np.float32(btc_bought)
            # Fee: 0.02% (maker) + 0.02% (taker) ≈ 0.04% total
            self.balance -= np.float32(btc_bought * current_price * 1.0004)
            self._record_trade("buy", btc_bought, current_price)

        elif action == 2:  # Sell — liquidate 10% of holdings
            if self.btc_held > 0:
                btc_sold = float(self.btc_held) * 0.1
                self.btc_held -= np.float32(btc_sold)
                self.balance += np.float32(btc_sold * current_price * 0.9996)
                self._record_trade("sell", btc_sold, current_price)

    def _calculate_reward(self) -> float:
        current_value = float(self.net_worth)
        previous_value = float(self._get_previous_value())
        if current_value == previous_value:
            return -0.001
        return current_value - previous_value

    def _get_previous_value(self) -> float:
        if self.current_step == 0:
            return float(self.initial_balance)
        prev_price = float(self.df.iloc[self.current_step - 1]["close"])
        return float(self.balance) + float(self.btc_held) * prev_price

    def _record_trade(self, action_type: str, amount: float, price: float) -> None:
        self.render_data["trades"].append(
            (self.current_step, action_type, amount, price)
        )
        # Prune old trades to avoid memory growth
        if len(self.render_data["trades"]) > 1000:
            self.render_data["trades"] = self.render_data["trades"][-500:]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def net_worth(self) -> np.float32:
        return self.balance + self.btc_held * self.current_price

    @property
    def current_price(self) -> np.float32:
        return np.float32(self.df.iloc[self.current_step]["close"])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, mode: str = "human", recent_steps: int = 200) -> None:
        if mode == "human":
            self._render_human(recent_steps)

    def _render_human(self, recent_steps: int = 200) -> None:
        if self.fig is None:
            plt.ion()
            self.fig, (self.price_ax, self.balance_ax) = plt.subplots(
                2, 1, figsize=(12, 8)
            )
            self.price_line, = self.price_ax.plot([], [], label="Price", color="blue")
            self.balance_line, = self.balance_ax.plot(
                [], [], label="Net Worth", color="green"
            )
            self.price_ax.set_title("BTC/USDT Price")
            self.balance_ax.set_title("Portfolio Value")
            plt.tight_layout()
            plt.show(block=False)

        self.render_data["prices"].append(float(self.current_price))
        self.render_data["balances"].append(float(self.net_worth))

        window_start = max(0, len(self.render_data["prices"]) - recent_steps)
        prices = self.render_data["prices"][window_start:]
        balances = self.render_data["balances"][window_start:]
        steps = np.arange(len(prices))

        self.price_line.set_data(steps, prices)
        self.price_ax.relim()
        self.price_ax.autoscale_view()
        self.price_ax.set_title(f"BTC/USDT Price (Step {self.current_step})")

        self.balance_line.set_data(steps, balances)
        self.balance_ax.relim()
        self.balance_ax.autoscale_view()

        # Remove old trade markers
        for artist in list(self.price_ax.texts) + list(self.price_ax.lines[2:]):
            artist.remove()

        window_trades = [
            t for t in self.render_data["trades"] if t[0] >= (self.current_step - len(prices))
        ]
        for trade in window_trades[-10:]:
            step, action_type, amount, price = trade
            x_pos = step - (self.current_step - len(prices))
            color = "lime" if action_type == "buy" else "red"
            if 0 <= x_pos < len(prices):
                self.price_ax.axvline(x=x_pos, color=color, alpha=0.3)
                self.price_ax.text(
                    x_pos,
                    price,
                    f"{action_type}\n{amount:.4f}",
                    color=color,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# ------------------------------------------------------------------
# Utility: prepare dataframe for use with the environment
# ------------------------------------------------------------------

def prepare_crypto_data(
    df: pd.DataFrame,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """
    Prepare a raw price dataframe for use with CryptoTradingEnv.

    - Selects numeric columns only
    - Forward-fills then backward-fills missing values
    - Ensures 'close' column is present
    - Casts to float32
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    numeric = df.select_dtypes(include=[np.number])
    if forward_fill:
        numeric = numeric.ffill().bfill()
    else:
        numeric = numeric.fillna(0)
    return numeric.astype(np.float32)


# ------------------------------------------------------------------
# Utility: compute trading metrics from a list of net_worth values
# ------------------------------------------------------------------

def compute_metrics(portfolio_values: list[float]) -> dict:
    """
    Compute Sharpe ratio, returns, max drawdown, win rate, and volatility.

    Parameters
    ----------
    portfolio_values : list[float]
        List of net_worth values recorded each step.

    Returns
    -------
    dict
        Keys: 'returns', 'sharpe_ratios', 'max_drawdowns', 'win_rate', 'volatility'
    """
    vals = np.array(portfolio_values, dtype=np.float64)
    if len(vals) < 2:
        return {
            "returns": 0.0,
            "sharpe_ratios": 0.0,
            "max_drawdowns": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
        }
    returns = np.diff(vals) / vals[:-1]
    annualization_factor = np.sqrt(365 * 24)  # hourly data

    sharpe = np.nan_to_num(np.mean(returns) / np.std(returns)) * annualization_factor
    total_return = (vals[-1] / vals[0] - 1) * 100
    max_dd = (np.min(vals) / np.max(vals) - 1) * 100
    win_rate = (np.sum(returns > 0) / len(returns)) * 100
    volatility = np.std(returns) * annualization_factor

    return {
        "returns": float(total_return),
        "sharpe_ratios": float(sharpe),
        "max_drawdowns": float(max_dd),
        "win_rate": float(win_rate),
        "volatility": float(volatility),
    }


if __name__ == "__main__":
    import pandas as pd

    # Smoke-test with the bundled sample data
    DATA_PATH = "D:/Hive/Data/trading_repos/RL-Crypto-Trading-Bot/btc_data.csv"
    df = pd.read_csv(DATA_PATH)
    df = prepare_crypto_data(df)

    env = CryptoTradingEnv(df)
    obs, _ = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done:
            break

    env.close()
    print("Smoke test passed.")
