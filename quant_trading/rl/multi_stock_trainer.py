"""
Multi-stock RL training utilities.

Adapted from MultiStockRLTrading (D:/Hive/Data/trading_repos/MultiStockRLTrading/).

Provides:
- Feature engineering (technical indicators via TA-Lib)
- Data loading for multi-asset CSV files
- Environment factory compatible with stable-baselines3
- Training loop wrappers for PPO
- Evaluation helpers

Dependencies:
    stable-baselines3, pandas, numpy, talib
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from stable_baselines3 import PPO, A2C

from .multi_stock_env import MultiStockTradingEnv
from .cross_attention_policy import CrossAttentionActorCriticPolicy


# ------------------------------------------------------------------
# Default constants
# ------------------------------------------------------------------

DEFAULT_DATA_DIR = "history_data"
DEFAULT_WINDOW_SIZE = 12
DEFAULT_TRAIN_SPLIT = 1500
DEFAULT_INITIAL_AMOUNT = 1_000_000.0
DEFAULT_TRADE_COST = 0.0
DEFAULT_BATCH_SIZE = 256
DEFAULT_TIMESTEPS = 100_000
DEFAULT_MODEL_DIR = Path("saved_models")
DEFAULT_FIGURE_DIR = Path("artifacts")


# ------------------------------------------------------------------
# Technical indicators
# ------------------------------------------------------------------

BASE_INDICATORS = [
    "open", "high", "low", "close", "volume",
    "ret1min", "ret2min", "ret3min", "ret4min", "ret5min",
    "ret6min", "ret7min", "ret8min", "ret9min", "ret10min",
    "sma", "5sma", "20sma",
    "bb_upper", "bb_middle", "bb_lower", "bb_sell", "bb_buy", "bb_squeeze",
    "mom", "adx", "mfi", "rsi", "trange", "bop", "cci", "STOCHRSI",
    "slowk", "slowd", "macd", "macdsignal", "macdhist",
    "NATR", "KAMA", "MAMA", "FAMA",
    "MAMA_buy", "KAMA_buy", "sma_buy", "maco", "rsi_buy", "rsi_sell", "macd_buy_sell",
]


def add_features(tic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and return momentum features.

    Requires ``talib`` to be installed. If unavailable, returns the DataFrame
    unchanged with a warning.
    """
    if not TALIB_AVAILABLE:
        import warnings
        warnings.warn("talib not installed; skipping technical indicator computation")
        return tic_df

    enriched = tic_df.copy()
    close = enriched["close"].values

    # Return features
    for t in range(1, 11):
        enriched[f"ret{t}min"] = enriched["close"].div(enriched["open"].shift(t - 1)).sub(1)

    # Moving averages
    enriched["sma"] = talib.SMA(close)
    enriched["5sma"] = talib.SMA(close, timeperiod=5)
    enriched["20sma"] = talib.SMA(close, timeperiod=20)

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, matype=talib.MA_Type.T3)
    enriched["bb_upper"] = bb_upper
    enriched["bb_middle"] = bb_middle
    enriched["bb_lower"] = bb_lower
    enriched["bb_sell"] = (enriched["close"] > bb_upper).astype(int)
    enriched["bb_buy"] = (enriched["close"] < bb_lower).astype(int)
    enriched["bb_squeeze"] = (bb_upper - bb_middle) / bb_middle

    # Momentum / trend
    enriched["mom"] = talib.MOM(close, timeperiod=10)
    enriched["adx"] = talib.ADX(
        enriched["high"].values, enriched["low"].values, close, timeperiod=10
    )
    enriched["mfi"] = talib.MFI(
        enriched["high"].values, enriched["low"].values, close,
        enriched["volume"].values, timeperiod=10
    )
    enriched["rsi"] = talib.RSI(close, timeperiod=10)
    enriched["trange"] = talib.TRANGE(
        enriched["high"].values, enriched["low"].values, close
    )
    enriched["bop"] = talib.BOP(
        enriched["open"].values, enriched["high"].values,
        enriched["low"].values, close
    )
    enriched["cci"] = talib.CCI(
        enriched["high"].values, enriched["low"].values, close, timeperiod=14
    )

    stoch_rsi = talib.STOCHRSI(close, timeperiod=14, fastk_period=14, fastd_period=3)[0]
    enriched["STOCHRSI"] = stoch_rsi

    slowk, slowd = talib.STOCH(
        enriched["high"].values, enriched["low"].values, close,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )
    enriched["slowk"] = slowk
    enriched["slowd"] = slowd

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    enriched["macd"] = macd
    enriched["macdsignal"] = macdsignal
    enriched["macdhist"] = macdhist

    enriched["NATR"] = talib.NATR(
        enriched["high"].ffill().values, enriched["low"].ffill().values, close
    )
    enriched["KAMA"] = talib.KAMA(close, timeperiod=10)
    mama, fama = talib.MAMA(close)
    enriched["MAMA"] = mama
    enriched["FAMA"] = fama

    enriched["MAMA_buy"] = (mama < fama).astype(int)
    enriched["KAMA_buy"] = (close < enriched["KAMA"].values).astype(int)
    enriched["sma_buy"] = (close < enriched["5sma"].values).astype(int)
    enriched["maco"] = (enriched["5sma"] < enriched["20sma"]).astype(int)
    enriched["rsi_buy"] = (enriched["rsi"] < 30).astype(int)
    enriched["rsi_sell"] = (enriched["rsi"] > 70).astype(int)
    enriched["macd_buy_sell"] = (macd < macdsignal).astype(int)

    return enriched


# ------------------------------------------------------------------
# Policy registry
# ------------------------------------------------------------------

POLICY_REGISTRY: Dict[str, Type] = {
    "cross_attention": CrossAttentionActorCriticPolicy,
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_market_data(
    data_dir: str,
) -> Tuple[List[pd.DataFrame], pd.DataFrame, List[str], List[str]]:
    """
    Load multi-asset CSV files from ``data_dir`` and compute technical indicators.

    Each CSV must contain columns: ``datetime``, ``name``, ``open``, ``high``,
    ``low``, ``close``, ``volume``.

    Returns
    -------
    df_list : List[pd.DataFrame]
        One DataFrame per asset, with technical features added.
    price_df : pd.DataFrame
        Close prices only, columns = asset names.
    names : List[str]
        Asset names in load order.
    indicators : List[str]
        Feature column names used as the observation space.
    """
    dfs = pd.DataFrame()
    names: List[str] = []
    num_assets = 0

    data_files = sorted(glob.glob(os.path.join(data_dir, "*")))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir!r}")

    for filename in data_files:
        df = pd.read_csv(filename)
        if "datetime" not in df.columns or "name" not in df.columns:
            continue

        df["datetime"] = pd.to_datetime(df["datetime"])
        name = df["name"].iloc[0]
        names.append(name)

        # Time-of-day and day-of-week features
        df["ToD"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
        df["DoW"] = df["datetime"].dt.weekday / 6.0
        df.sort_values(["datetime"], inplace=True)

        # Add technical features
        df = add_features(df)

        # Drop non-feature columns
        cols_to_drop = ["datetime", "name"]
        if "token" in df.columns:
            cols_to_drop.append("token")
        for col in cols_to_drop:
            if col in df.columns:
                df.drop([col], axis=1, inplace=True)

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df = df.ffill().bfill().fillna(0)

        # Collect columns for this asset
        cols_per_asset = len(df.columns)
        if num_assets == 0:
            dfs = df.copy()
        else:
            # Append horizontally — assumes all assets share the same index
            dfs = pd.concat([dfs, df], axis=1)
        num_assets += 1

    # Split back into per-asset DataFrames
    cols_per_asset = int(len(dfs.columns) / num_assets)
    df_list: List[pd.DataFrame] = []
    price_df = pd.DataFrame()

    for idx in range(num_assets):
        asset_df = dfs.iloc[:, idx * cols_per_asset : idx * cols_per_asset + cols_per_asset].copy()
        asset_df.index = dfs.index  # preserve index alignment
        price_df[names[idx]] = asset_df["close"]
        df_list.append(asset_df)

    return df_list, price_df, names, BASE_INDICATORS


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

def make_env(
    df_list: List[pd.DataFrame],
    price_df: pd.DataFrame,
    indicators: List[str],
    window_size: int,
    frame_bound: Tuple[int, int],
    initial_amount: float = DEFAULT_INITIAL_AMOUNT,
    trade_cost: float = DEFAULT_TRADE_COST,
    scalers=None,
) -> MultiStockTradingEnv:
    """Factory to construct a MultiStockTradingEnv with given parameters."""
    return MultiStockTradingEnv(
        df_list=df_list,
        price_df=price_df,
        num_stocks=len(df_list),
        initial_amount=initial_amount,
        trade_cost=trade_cost,
        num_features=len(indicators),
        window_size=window_size,
        frame_bound=frame_bound,
        scalers=scalers,
        tech_indicator_list=indicators,
    )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_ppo(
    env: MultiStockTradingEnv,
    policy: Type = CrossAttentionActorCriticPolicy,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 3e-4,
    n_epochs: int = 10,
    gamma: float = 0.99,
    tensorboard_log: Optional[str] = "tb_logs",
    model_name: str = "MultiStockTrader",
    model_dir: Path = DEFAULT_MODEL_DIR,
    verbose: int = 1,
) -> PPO:
    """
    Train a PPO model with the given multi-stock environment.

    Parameters
    ----------
    env : MultiStockTradingEnv
        Pre-initialized and data-processed environment.
    policy : Type
        Policy class (e.g. ``CrossAttentionActorCriticPolicy``).
    total_timesteps : int
        Total training steps.
    batch_size : int
        PPO batch size.
    learning_rate : float
        Initial learning rate (schedule applied automatically).
    n_epochs : int
        PPO epoch count per update.
    gamma : float
        Discount factor.
    tensorboard_log : str, optional
        Directory for TensorBoard logs.
    model_name : str
        Filename stem for the saved model.
    model_dir : Path
        Directory to save the model.
    verbose : int
        Verbosity level passed to stable-baselines3.

    Returns
    -------
    PPO
        Trained model (already saved to disk).
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        policy,
        env,
        verbose=verbose,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        gamma=gamma,
        batch_size=batch_size,
        tensorboard_log=tensorboard_log,
    )
    model.learn(total_timesteps=total_timesteps)

    model_path = model_dir / model_name
    model.save(str(model_path))
    return model


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def run_inference(
    env: MultiStockTradingEnv,
    model: PPO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the trained model through the evaluation episode.

    Parameters
    ----------
    env : MultiStockTradingEnv
        Environment configured for the holdout period.
    model : PPO
        Trained model.

    Returns
    -------
    steps : np.ndarray
        Timestep indices.
    cumulative_rewards : np.ndarray
        Cumulative reward at each step.
    """
    obs, _ = env.reset()
    infer_rewards: List[float] = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        infer_rewards.append(float(reward))
        if terminated or truncated:
            break

    steps = np.arange(len(infer_rewards))
    cumulative_rewards = np.cumsum(np.array(infer_rewards))
    return steps, cumulative_rewards


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-asset RL trader with cross-attention policy."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--policy", choices=sorted(POLICY_REGISTRY.keys()), default="cross_attention")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--train-split", type=int, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--initial-amount", type=float, default=DEFAULT_INITIAL_AMOUNT)
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-name", default="MultiStockTrader")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--tensorboard-log", default="tb_logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    df_list, price_df, names, indicators = load_market_data(args.data_dir)

    train_end = len(price_df) - args.train_split
    if train_end <= args.window_size:
        raise ValueError(
            f"Training split leaves too little history. "
            f"Need > {args.window_size} rows before holdout, got {train_end}."
        )

    # Train environment
    train_env = make_env(
        df_list=df_list,
        price_df=price_df,
        indicators=indicators,
        window_size=args.window_size,
        frame_bound=(args.window_size, train_end),
        initial_amount=args.initial_amount,
        trade_cost=args.trade_cost,
    )
    train_env.process_data()

    policy_cls = POLICY_REGISTRY[args.policy]
    model = train_ppo(
        env=train_env,
        policy=policy_cls,
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    print(f"Saved model to {DEFAULT_MODEL_DIR / args.model_name}")

    if args.skip_inference:
        return

    # Evaluate on holdout
    eval_env = make_env(
        df_list=df_list,
        price_df=price_df,
        indicators=indicators,
        window_size=args.window_size,
        frame_bound=(train_end, len(price_df)),
        initial_amount=args.initial_amount,
        trade_cost=args.trade_cost,
        scalers=train_env.scalers,
    )
    eval_env.process_data()

    loaded_model = PPO.load(str(DEFAULT_MODEL_DIR / args.model_name))
    infer_steps, cumulative_rewards = run_inference(eval_env, loaded_model)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 6))
    plt.title(args.model_name)
    plt.plot(infer_steps, cumulative_rewards, color="red", label="Cumulative Reward")
    plt.legend()
    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = DEFAULT_FIGURE_DIR / f"{args.model_name}_inference.png"
    plt.savefig(figure_path)
    print(f"Saved inference plot to {figure_path}")


if __name__ == "__main__":
    main()
