#!/usr/bin/env python
# coding: utf-8
"""
DAPO Trading Backtest Script.

Adapted from FinRL-DAPO-SR for quant_trading package.

Usage:
    python backtest_dapo.py --model_path ./checkpoint/agent_dapo_final.pth
    python backtest_dapo.py --model_path ./checkpoint/agent_dapo_both_a1.0_b1.0.pth
    python backtest_dapo.py --data_path ./dataset/trade_data.csv --start_date 2019-01-01

Requires:
    - Trained DAPO model checkpoint
    - Trade data with price, technical indicators, and LLM signals
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from env_stocktrading_llm_risk import StockTradingEnv

DEFAULT_INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
    "boll",
    "closebb",
    "volbb",
]


def load_dapo_model(model_path: str, observation_space, action_space, hidden_sizes=(256, 128)):
    """
    Load a trained DAPO model from checkpoint.

    Args:
        model_path: Path to .pth checkpoint
        observation_space: Gym observation space
        action_space: Gym action space
        hidden_sizes: Hidden layer sizes (must match training)

    Returns:
        Loaded MLPActorCritic model
    """
    from dapo_algorithm import MLPActorCritic

    ac = MLPActorCritic(observation_space, action_space, hidden_sizes=hidden_sizes)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        ac.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
        return ac
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")


def run_backtest(
    model,
    df: pd.DataFrame,
    stock_dim: int,
    hmax: int = 100,
    initial_amount: int = 1000000,
    tech_indicator_list: list = None,
    model_name: str = "DAPO",
    mode: str = "test",
    iteration: str = "0",
):
    """
    Run backtest with a trained model.

    Args:
        model: Trained actor-critic model
        df: Trade DataFrame with date, tic, close, indicators, llm_sentiment, llm_risk
        stock_dim: Number of stocks
        hmax: Max shares per trade
        initial_amount: Starting capital
        tech_indicator_list: Technical indicator columns
        model_name: Model identifier for output files
        mode: 'train' or 'test'
        iteration: Iteration identifier

    Returns:
        Dictionary with backtest results
    """
    if tech_indicator_list is None:
        tech_indicator_list = DEFAULT_INDICATORS

    state_space = 1 + 2 * stock_dim + (2 + len(tech_indicator_list)) * stock_dim

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
        "model_name": model_name,
        "mode": mode,
        "iteration": iteration,
    }

    # Create environment
    e_trade_gym = StockTradingEnv(df=df, **env_kwargs)
    trade_env, _ = e_trade_gym.get_sb_env()

    # Run episode
    obs, _ = trade_env.reset()
    done = False
    episode_returns = []
    actions_history = []

    while not done:
        # Get action from model (continuous, scaled -1 to 1)
        action = model.act(obs)
        actions_history.append(action)

        # Step environment
        obs, reward, done, truncated, info = trade_env.step(action)
        episode_returns.append(reward)

        if truncated or done:
            break

    # Collect results
    account_value = e_trade_gym.save_asset_memory()
    actions_df = e_trade_gym.save_action_memory()

    # Calculate metrics
    account_value["daily_return"] = account_value["account_value"].pct_change(1)
    sharpe = 0
    if account_value["daily_return"].std() != 0:
        sharpe = (
            (252**0.5)
            * account_value["daily_return"].mean()
            / account_value["daily_return"].std()
        )

    total_return = (
        account_value["account_value"].iloc[-1] - initial_amount
    ) / initial_amount

    results = {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "final_value": account_value["account_value"].iloc[-1],
        "initial_value": initial_amount,
        "account_value": account_value,
        "actions": actions_df,
        "total_trades": e_trade_gym.trades,
        "total_cost": e_trade_gym.cost,
    }

    return results


def print_backtest_summary(results: dict, model_name: str = "DAPO"):
    """Print a formatted backtest summary."""
    print("\n" + "=" * 50)
    print(f"  {model_name} Backtest Summary")
    print("=" * 50)
    print(f"  Initial Value:     ${results['initial_value']:,.2f}")
    print(f"  Final Value:      ${results['final_value']:,.2f}")
    print(f"  Total Return:     {results['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Total Cost:       ${results['total_cost']:,.2f}")
    print("=" * 50)


def prepare_trade_data(
    price_data_path: str,
    risk_data_path: str = None,
    sentiment_data_path: str = None,
    start_date: str = None,
    end_date: str = None,
):
    """
    Load and merge trade data with LLM signals.

    Similar to training data preparation but for out-of-sample testing.
    """
    train = pd.read_csv(price_data_path)

    if start_date:
        train = train[train["date"] >= start_date]
    if end_date:
        train = train[train["date"] <= end_date]

    if risk_data_path and os.path.exists(risk_data_path):
        risk_data = pd.read_csv(risk_data_path)
        train = pd.merge(train, risk_data, on=["date", "tic"], how="left")
        train["llm_risk"].fillna(3, inplace=True)
    else:
        if "llm_risk" not in train.columns:
            train["llm_risk"] = 3.0

    if sentiment_data_path and os.path.exists(sentiment_data_path):
        sentiment_data = pd.read_csv(sentiment_data_path)
        train = pd.merge(
            train, sentiment_data, on=["date", "tic"], how="left", suffixes=("", "_sentiment")
        )
        if "llm_sentiment_sentiment" in train.columns:
            train["llm_sentiment"] = train["llm_sentiment_sentiment"]
            train.drop("llm_sentiment_sentiment", axis=1, inplace=True)
        train["llm_sentiment"].fillna(3, inplace=True)
    else:
        if "llm_sentiment" not in train.columns:
            train["llm_sentiment"] = 3.0

    unique_dates = train["date"].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    return train


def main():
    parser = argparse.ArgumentParser(description="Backtest DAPO trading model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/trade_data.csv",
        help="Path to trade data CSV",
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
        default="2019-01-01",
        help="Trade start date",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2023-12-31",
        help="Trade end date",
    )
    parser.add_argument(
        "--initial_amount",
        type=int,
        default=1000000,
        help="Initial capital",
    )
    parser.add_argument(
        "--hmax",
        type=int,
        default=100,
        help="Max shares per trade",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Load trade data
    print(f"Loading trade data from {args.data_path}...")
    trade_df = prepare_trade_data(
        args.data_path,
        risk_data_path=args.risk_path,
        sentiment_data_path=args.sentiment_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    stock_dimension = len(trade_df.tic.unique())
    print(f"Stock Dimension: {stock_dimension}")

    # Create dummy environment to get spaces
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

    e_dummy = StockTradingEnv(df=trade_df, **env_kwargs)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_dapo_model(
        args.model_path,
        e_dummy.observation_space,
        e_dummy.action_space,
        hidden_sizes=tuple(args.hidden_sizes),
    )
    model.eval()

    # Run backtest
    print("Running backtest...")
    results = run_backtest(
        model=model,
        df=trade_df,
        stock_dim=stock_dimension,
        hmax=args.hmax,
        initial_amount=args.initial_amount,
        tech_indicator_list=DEFAULT_INDICATORS,
        model_name="DAPO",
        mode="test",
    )

    # Print summary
    print_backtest_summary(results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results["account_value"].to_csv(
        os.path.join(args.output_dir, "backtest_account_value.csv"), index=False
    )
    results["actions"].to_csv(
        os.path.join(args.output_dir, "backtest_actions.csv")
    )
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
