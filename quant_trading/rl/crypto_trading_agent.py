"""
CryptoTradingAgent — PPO / A2C / DQN agents for BTC/USDT crypto trading.

Adapted from RL-Crypto-Trading-Bot (Vnadh/RL-Crypto-Trading-Bot).
https://github.com/Vnadh/RL-Crypto-Trading-Bot

Performance benchmarks (5000-step backtest on unseen BTC data):
    DQN  — Sharpe: 2.22, Returns: 1.28%, Max Drawdown: -2.31%, Win Rate: 51.90%
    PPO  — Sharpe: 1.85, Returns: 0.97%, Max Drawdown: -1.94%, Win Rate: 51.76%
    A2C  — Sharpe: 1.82, Returns: 0.99%, Max Drawdown: -1.99%, Win Rate: 51.78%

Usage:
    from quant_trading.rl import CryptoTradingAgent, CryptoTradingEnv

    agent = CryptoTradingAgent("DQN")
    agent.train(env, timesteps=500_000)
    agent.save("dqn_crypto")

    # Or use the standalone trainers
    from quant_trading.rl.crypto_trading_agent import train_ppo, train_a2c, train_dqn
"""

from __future__ import annotations

import os
from typing import Optional, Type

import numpy as np
import pandas as pd

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from .crypto_env import CryptoTradingEnv, compute_metrics, prepare_crypto_data


# ------------------------------------------------------------------
# Default hyperparameters (from source repo benchmarks)
# ------------------------------------------------------------------

DEFAULT_MODELS_CONFIG: dict = {
    "PPO": {
        "class": PPO,
        "policy": "MlpPolicy",
        "n_envs": 4,
        "params": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "n_steps": 256,
            "batch_size": 64,
            "ent_coef": 0.01,
        },
        "timesteps": 1_000_000,
    },
    "A2C": {
        "class": A2C,
        "policy": "MlpPolicy",
        "n_envs": 4,
        "params": {
            "learning_rate": 7e-4,
            "gamma": 0.95,
            "n_steps": 128,
            "use_rms_prop": True,
            "ent_coef": 0.01,
        },
        "timesteps": 800_000,
    },
    "DQN": {
        "class": DQN,
        "policy": "MlpPolicy",
        "n_envs": 1,  # DQN does not support vec_env
        "params": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "buffer_size": 100_000,
            "exploration_final_eps": 0.01,
        },
        "timesteps": 500_000,
    },
}


# ------------------------------------------------------------------
# Unified agent class
# ------------------------------------------------------------------

class CryptoTradingAgent:
    """
    Unified PPO / A2C / DQN crypto trading agent.

    Parameters
    ----------
    algorithm : str
        One of "PPO", "A2C", "DQN".
    model_config : dict, optional
        Override the default hyperparameter config for the chosen algorithm.
    verbose : int, default 1
        Verbosity level passed to stable-baselines3.
    """

    SUPPORTED_ALGORITHMS = {"PPO", "A2C", "DQN"}

    def __init__(
        self,
        algorithm: str,
        model_config: Optional[dict] = None,
        verbose: int = 1,
    ) -> None:
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from {self.SUPPORTED_ALGORITHMS}."
            )
        self.algorithm = algorithm
        self.model_config = model_config or DEFAULT_MODELS_CONFIG[algorithm]
        self.verbose = verbose
        self.model: Optional[PPO | A2C | DQN] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        env: CryptoTradingEnv | pd.DataFrame,
        timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        **kwargs,
    ) -> "CryptoTradingAgent":
        """
        Train the agent on the given environment.

        Parameters
        ----------
        env : CryptoTradingEnv or pd.DataFrame
            If DataFrame, it is first wrapped with CryptoTradingEnv.
        timesteps : int, optional
            Override the default total training timesteps.
        callback : BaseCallback, optional
            Stable-baselines3 training callback.
        **kwargs
            Additional arguments forwarded to the underlying sb3 model.

        Returns
        -------
        self
        """
        config = self.model_config
        ts = timesteps or config["timesteps"]

        if isinstance(env, pd.DataFrame):
            env = CryptoTradingEnv(prepare_crypto_data(env))

        n_envs = config["n_envs"]
        if n_envs > 1 and self.algorithm == "DQN":
            n_envs = 1  # DQN requires n_envs=1

        if n_envs > 1:
            train_env = make_vec_env(
                lambda: CryptoTradingEnv(prepare_crypto_data(env.df)),
                n_envs=n_envs,
            )
        else:
            train_env = CryptoTradingEnv(prepare_crypto_data(env.df))

        self.model = config["class"](
            config["policy"],
            train_env,
            verbose=self.verbose,
            **config["params"],
            **kwargs,
        )

        self.model.learn(total_timesteps=ts, callback=callback)
        return self

    # ------------------------------------------------------------------
    # Inference / evaluation
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        """Return the next action for the given observation."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action), None

    def evaluate(
        self,
        env: CryptoTradingEnv,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict:
        """
        Run evaluation episodes and return mean metrics.

        Returns
        -------
        dict
            Mean values for: returns, sharpe_ratios, max_drawdowns, win_rate, volatility
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        all_metrics = {
            "returns": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
            "win_rate": [],
            "volatility": [],
        }

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            portfolio_values = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, _, done, _, info = env.step(action)
                portfolio_values.append(info["net_worth"])

            metrics = compute_metrics(portfolio_values)
            for k, v in metrics.items():
                all_metrics[k].append(v)

        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model to a zip file."""
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str, algorithm: Optional[str] = None) -> "CryptoTradingAgent":
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str
            Path to the saved .zip file (without extension, stable-baselines3 adds it).
        algorithm : str, optional
            Algorithm name used for the agent class. If None, inferred from the filename.
        """
        algo = algorithm or os.path.basename(path).split("_")[0].upper()
        if algo not in cls.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Cannot infer algorithm from '{path}'. Provide algorithm='PPO' etc.")
        agent = cls(algorithm=algo)
        agent.model = cls._load_model(algo, path)
        return agent

    @staticmethod
    def _load_model(algorithm: str, path: str) -> PPO | A2C | DQN:
        """Internal loader dispatch."""
        cls_map: dict[str, Type[PPO | A2C | DQN]] = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        return cls_map[algorithm].load(path)


# ------------------------------------------------------------------
# Standalone trainers (mirror the source repo interface)
# ------------------------------------------------------------------

def train_ppo(
    df: pd.DataFrame,
    timesteps: int = 1_000_000,
    n_envs: int = 4,
    save_path: str = "models/PPO",
    verbose: int = 1,
    **kwargs,
) -> PPO:
    """
    Train a PPO agent on crypto data.

    Parameters
    ----------
    df : pd.DataFrame
        BTC/USDT price/feature data.
    timesteps : int
        Total training timesteps.
    n_envs : int
        Number of parallel environments.
    save_path : str
        Path to save the trained model.
    **kwargs
        Additional params for PPO.

    Returns
    -------
    PPO
    """
    data = prepare_crypto_data(df)
    env = make_vec_env(lambda: CryptoTradingEnv(data), n_envs=n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=256,
        batch_size=64,
        ent_coef=0.01,
        **kwargs,
    )
    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    return model


def train_a2c(
    df: pd.DataFrame,
    timesteps: int = 800_000,
    n_envs: int = 4,
    save_path: str = "models/A2C",
    verbose: int = 1,
    **kwargs,
) -> A2C:
    """Train an A2C agent on crypto data. See train_ppo for parameter docs."""
    data = prepare_crypto_data(df)
    env = make_vec_env(lambda: CryptoTradingEnv(data), n_envs=n_envs)
    model = A2C(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=7e-4,
        gamma=0.95,
        n_steps=128,
        use_rms_prop=True,
        ent_coef=0.01,
        **kwargs,
    )
    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    return model


def train_dqn(
    df: pd.DataFrame,
    timesteps: int = 500_000,
    save_path: str = "models/DQN",
    verbose: int = 1,
    **kwargs,
) -> DQN:
    """
    Train a DQN agent on crypto data.

    Note: DQN does not support vectorized environments (n_envs must be 1).

    Parameters
    ----------
    df : pd.DataFrame
        BTC/USDT price/feature data.
    timesteps : int
        Total training timesteps.
    save_path : str
        Path to save the trained model.
    **kwargs
        Additional params for DQN.

    Returns
    -------
    DQN
    """
    data = prepare_crypto_data(df)
    env = CryptoTradingEnv(data)  # single env for DQN
    model = DQN(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=100_000,
        exploration_final_eps=0.01,
        **kwargs,
    )
    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    return model


# ------------------------------------------------------------------
# Convenience: train all models and compare
# ------------------------------------------------------------------

def train_and_compare(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    save_dir: str = "models",
) -> dict:
    """
    Train PPO, A2C, and DQN on the same data and return evaluation results.

    Parameters
    ----------
    df : pd.DataFrame
        BTC/USDT price/feature data.
    config : dict, optional
        Overrides for DEFAULT_MODELS_CONFIG.
    save_dir : str
        Directory to save model .zip files.

    Returns
    -------
    dict
        Per-algorithm mean evaluation metrics.
    """
    cfg = config or DEFAULT_MODELS_CONFIG
    results = {}

    for name, c in cfg.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")

        data = prepare_crypto_data(df)
        if c["n_envs"] > 1 and name != "DQN":
            env = make_vec_env(lambda d=data: CryptoTradingEnv(d), n_envs=c["n_envs"])
        else:
            env = CryptoTradingEnv(data)

        model = c["class"](c["policy"], env, verbose=1, **c["params"])
        model.learn(total_timesteps=c["timesteps"])

        save_path = os.path.join(save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_path)

        # Evaluate on a fresh single env
        eval_env = CryptoTradingEnv(data)
        agent = CryptoTradingAgent(name, model_config=cfg[name])
        agent.model = model
        metrics = agent.evaluate(eval_env, n_episodes=10)
        results[name] = metrics

        print(f"\n{name} Results:")
        for k, v in metrics.items():
            print(f"  {k:20}: {v:.4f}")

        env.close()
        eval_env.close()

    return results


if __name__ == "__main__":
    import pprint

    DATA_PATH = "D:/Hive/Data/trading_repos/RL-Crypto-Trading-Bot/btc_data.csv"
    df = pd.read_csv(DATA_PATH)

    print("Training all models on sample data...")
    results = train_and_compare(df)
    pprint.pprint(results)
