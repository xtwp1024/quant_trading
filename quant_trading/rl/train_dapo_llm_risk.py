#!/usr/bin/env python
# coding: utf-8
"""
DAPO LLM Risk Trading Training Script.

Adapted from FinRL-DAPO-SR for quant_trading package.

Usage:
    python train_dapo_llm_risk.py --epochs 100 --hid 512 --l 2 --seed 0
    python train_dapo_llm_risk.py --adjustment_type both --alpha 1.0 --beta 1.0
    python train_dapo_llm_risk.py --adjustment_type sentiment --alpha 1.5
    python train_dapo_llm_risk.py --adjustment_type risk --beta 1.2
    python train_dapo_llm_risk.py --adjustment_type none  # No LLM adjustment

Required Data:
    - Stock price data with technical indicators (macd, rsi, cci, dx, boll, closebb, volbb)
    - LLM risk scores (llm_risk, scale 1-5)
    - LLM sentiment scores (llm_sentiment, scale 1-5)

State Space:
    [cash] + [close_prices] + [num_shares] + [technical_indicators per stock]
    + [llm_sentiment per stock] + [llm_risk per stock]

Note:
    Uses MPI for distributed training if available (mpirun -np 4 python train_dapo_llm_risk.py).
    Falls back to single-process if MPI is unavailable.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from env_stocktrading_llm_risk import StockTradingEnv

# Default technical indicators (must match data columns)
DEFAULT_INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
    "boll",
    "closebb",
    "volbb",
]


def create_trading_env(
    df: pd.DataFrame,
    stock_dim: int,
    hmax: int = 100,
    initial_amount: int = 1000000,
    buy_cost_pct: float = 0.001,
    sell_cost_pct: float = 0.001,
    reward_scaling: float = 1e-4,
    tech_indicator_list: list = None,
):
    """
    Create a StockTradingEnv with LLM risk signals.

    Args:
        df: DataFrame with columns: date, tic, close, [tech_indicators], llm_sentiment, llm_risk
        stock_dim: Number of stocks to trade
        hmax: Maximum number of shares per action
        initial_amount: Starting cash
        buy_cost_pct: Buy transaction cost percentage
        sell_cost_pct: Sell transaction cost percentage
        reward_scaling: Reward scaling factor
        tech_indicator_list: List of technical indicator column names

    Returns:
        StockTradingEnv instance
    """
    if tech_indicator_list is None:
        tech_indicator_list = DEFAULT_INDICATORS

    # Calculate state space
    # 1 (cash) + stock_dim (prices) + stock_dim (shares) + (2+len(indicators))*stock_dim (indicators+LLM)
    state_space = (
        1
        + 2 * stock_dim
        + (2 + len(tech_indicator_list)) * stock_dim
    )

    buy_cost_list = sell_cost_list = [buy_cost_pct] * stock_dim
    num_stock_shares = [0] * stock_dim

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dim,
        "reward_scaling": reward_scaling,
    }

    e_train_gym = StockTradingEnv(df=df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    return env_train, e_train_gym


def prepare_data(
    price_data_path: str,
    risk_data_path: str = None,
    sentiment_data_path: str = None,
    start_date: str = None,
    end_date: str = None,
):
    """
    Load and merge price, risk, and sentiment data.

    Args:
        price_data_path: Path to price + indicators CSV
        risk_data_path: Optional path to LLM risk scores CSV (columns: date, tic, llm_risk)
        sentiment_data_path: Optional path to LLM sentiment CSV (columns: date, tic, llm_sentiment)
        start_date: Optional filter start date
        end_date: Optional filter end date

    Returns:
        Merged DataFrame with all signals
    """
    # Load price data
    train = pd.read_csv(price_data_path)

    if start_date:
        train = train[train["date"] >= start_date]
    if end_date:
        train = train[train["date"] <= end_date]

    # Merge LLM risk if provided
    if risk_data_path and os.path.exists(risk_data_path):
        risk_data = pd.read_csv(risk_data_path)
        train = pd.merge(train, risk_data, on=["date", "tic"], how="left")
        train["llm_risk"].fillna(3, inplace=True)  # Neutral risk score
    else:
        # Add dummy column if not present
        if "llm_risk" not in train.columns:
            train["llm_risk"] = 3.0

    # Merge LLM sentiment if provided
    if sentiment_data_path and os.path.exists(sentiment_data_path):
        sentiment_data = pd.read_csv(sentiment_data_path)
        train = pd.merge(
            train, sentiment_data, on=["date", "tic"], how="left", suffixes=("", "_sentiment")
        )
        # Handle possible column name collision
        if "llm_sentiment_sentiment" in train.columns:
            train["llm_sentiment"] = train["llm_sentiment_sentiment"]
            train.drop("llm_sentiment_sentiment", axis=1, inplace=True)
        train["llm_sentiment"].fillna(3, inplace=True)  # Neutral sentiment
    else:
        if "llm_sentiment" not in train.columns:
            train["llm_sentiment"] = 3.0

    # Create a proper date index
    unique_dates = train["date"].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    return train


def train_dapo(
    env_train,
    stock_dim: int,
    tech_indicator_list: list,
    env_kwargs: dict,
    num_samples_per_state: int = 10,
    epochs: int = 100,
    hid: int = 512,
    l: int = 2,
    seed: int = 0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
    adjustment_type: str = "both",
    alpha: float = 1.0,
    beta: float = 1.0,
    checkpoint_dir: str = "./checkpoint",
    exp_name: str = "dapo",
    save_freq: int = 10,
):
    """
    Train DAPO agent on the trading environment.

    Args:
        env_train: Training environment
        stock_dim: Number of stocks
        tech_indicator_list: Technical indicators used
        env_kwargs: Environment configuration
        num_samples_per_state: DAPO action samples per state
        epochs: Training epochs
        hid: Hidden layer size
        l: Number of hidden layers
        seed: Random seed
        epsilon_low: Lower clip ratio
        epsilon_high: Upper clip ratio
        adjustment_type: LLM adjustment ('both', 'sentiment', 'risk', 'none')
        alpha: Sentiment exponent
        beta: Risk exponent
        checkpoint_dir: Where to save models
        exp_name: Experiment name
        save_freq: Checkpoint frequency

    Returns:
        Trained DAPO actor-critic model
    """
    from dapo_algorithm import dapo, MLPActorCritic

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cpu")
    print("Using CPU for training")

    # Try to setup logger
    try:
        from spinup.utils.run_utils import setup_logger_kwargs

        logger_kwargs = setup_logger_kwargs(exp_name, seed)
    except ImportError:
        logger_kwargs = dict()

    trained_dapo = dapo(
        lambda: env_train,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[hid] * l),
        seed=seed,
        logger_kwargs=logger_kwargs,
        num_samples_per_state=num_samples_per_state,
        epochs=epochs,
        env_kwargs=env_kwargs,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        adjustment_type=adjustment_type,
        alpha=alpha,
        beta=beta,
        force_cpu=True,
    )

    # Build model filename based on adjustment parameters
    if adjustment_type == "both":
        model_name = f"agent_dapo_{adjustment_type}_a{alpha}_b{beta}.pth"
    elif adjustment_type == "sentiment":
        model_name = f"agent_dapo_{adjustment_type}_a{alpha}.pth"
    elif adjustment_type == "risk":
        model_name = f"agent_dapo_{adjustment_type}_b{beta}.pth"
    else:
        model_name = "agent_dapo_no_adjustment.pth"

    final_model_path = os.path.join(checkpoint_dir, model_name)
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": trained_dapo.state_dict(),
            "adjustment_type": adjustment_type,
            "alpha": alpha,
            "beta": beta,
            "stock_dim": stock_dim,
            "tech_indicator_list": tech_indicator_list,
        },
        final_model_path,
    )
    print(f"\nTraining finished. Final model saved in {final_model_path}")

    return trained_dapo


def main():
    parser = argparse.ArgumentParser(
        description="Train DAPO agent with LLM risk signals"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/train_data.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--risk_path",
        type=str,
        default=None,
        help="Path to LLM risk data CSV",
    )
    parser.add_argument(
        "--sentiment_path",
        type=str,
        default=None,
        help="Path to LLM sentiment data CSV",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2013-01-01",
        help="Training start date",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2018-12-31",
        help="Training end date",
    )
    parser.add_argument(
        "--hid", type=int, default=512, help="Hidden layer size"
    )
    parser.add_argument(
        "--l", type=int, default=2, help="Number of hidden layers"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Action samples per state for DAPO",
    )
    parser.add_argument(
        "--epsilon_low",
        type=float,
        default=0.2,
        help="Lower clip ratio (DAPO)",
    )
    parser.add_argument(
        "--epsilon_high",
        type=float,
        default=0.28,
        help="Upper clip ratio (DAPO)",
    )
    parser.add_argument(
        "--adjustment_type",
        type=str,
        default="both",
        choices=["both", "sentiment", "risk", "none"],
        help="LLM adjustment type",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Sentiment exponent"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Risk exponent"
    )
    parser.add_argument(
        "--initial_amount",
        type=int,
        default=1000000,
        help="Initial capital",
    )
    parser.add_argument(
        "--hmax", type=int, default=100, help="Max shares per trade"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoint",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--exp_name", type=str, default="dapo", help="Experiment name"
    )
    args = parser.parse_args()

    # Load and prepare data
    print(f"Loading data from {args.data_path}...")
    train = prepare_data(
        args.data_path,
        risk_data_path=args.risk_path,
        sentiment_data_path=args.sentiment_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Setup environment
    stock_dimension = len(train.tic.unique())
    print(f"Stock Dimension: {stock_dimension}")

    env_kwargs = {
        "hmax": args.hmax,
        "initial_amount": args.initial_amount,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": 1
        + 2 * stock_dimension
        + (2 + len(DEFAULT_INDICATORS)) * stock_dimension,
        "stock_dim": stock_dimension,
        "tech_indicator_list": DEFAULT_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    print(
        f"State Space: {env_kwargs['state_space']}, Action Space: {env_kwargs['action_space']}"
    )

    # Train DAPO
    trained_dapo = train_dapo(
        env_train=env_train,
        stock_dim=stock_dimension,
        tech_indicator_list=DEFAULT_INDICATORS,
        env_kwargs=env_kwargs,
        num_samples_per_state=args.num_samples,
        epochs=args.epochs,
        hid=args.hid,
        l=args.l,
        seed=args.seed,
        epsilon_low=args.epsilon_low,
        epsilon_high=args.epsilon_high,
        adjustment_type=args.adjustment_type,
        alpha=args.alpha,
        beta=args.beta,
        checkpoint_dir=args.checkpoint_dir,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
