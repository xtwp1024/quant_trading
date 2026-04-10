#!/usr/bin/env python
# coding: utf-8
"""
RL Agent Backtest Comparison - Fully Isolated Version

This script runs in complete isolation from the quant_trading package
to avoid circular import issues.

Usage:
    # Run as module from parent directory
    cd D:/量化交易系统/量化之神
    python -m quant_trading.experiments.rl_backtest_demo
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import from quant_trading package (run as module to use these imports)
from quant_trading.rl.crypto_env import TradingEnvironment, prepare_crypto_data, compute_trading_metrics


def run_rl_simulation(df, config, n_episodes=3):
    """
    Run RL-style simulation (random actions with action masking).

    A trained RL agent would perform better than random,
    but this demonstrates the framework is working.
    """
    results = []

    for episode in range(n_episodes):
        env = TradingEnvironment(
            df=df,
            initial_balance=config['initial_balance'],
            buy_pct=config['buy_pct'],
            sell_pct=config['sell_pct'],
            commission=config['commission'],
            use_indicators=True,
            stop_loss_pct=config['stop_loss_pct'],
            take_profit_pct=config['take_profit_pct'],
        )

        obs, info = env.reset()
        done = False

        portfolio_values = [info['portfolio_value']]
        actions = []
        trades = []

        while not done:
            # Use action mask to avoid invalid actions
            mask = env.get_action_mask()
            valid_actions = np.where(mask == 1)[0]

            if len(valid_actions) > 0:
                action = int(np.random.choice(valid_actions))
            else:
                action = 0  # Hold if no valid actions

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            portfolio_values.append(info['portfolio_value'])
            actions.append(action)

        trades = env.trades
        env.close()

        # Compute metrics
        metrics = compute_trading_metrics(portfolio_values)
        metrics['n_trades'] = len(trades)
        metrics['episode'] = episode
        metrics['final_value'] = portfolio_values[-1]

        # Action distribution
        unique, counts = np.unique(actions, return_counts=True)
        action_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        metrics['action_distribution'] = action_dist

        results.append({
            'metrics': metrics,
            'portfolio_values': portfolio_values,
            'actions': actions,
        })

    return results


def run_buy_and_hold(df, config):
    """Run buy-and-hold baseline."""
    initial_balance = config['initial_balance']
    commission = config['commission']

    initial_price = float(df.iloc[0]["close"])
    final_price = float(df.iloc[-1]["close"])

    # Buy at beginning, hold until end
    btc_bought = (initial_balance * (1 - commission)) / initial_price
    final_value = btc_bought * final_price * (1 - commission)

    # Portfolio values over time
    prices = df["close"].values
    portfolio_values = list(btc_bought * prices * (1 - commission))

    # Metrics
    metrics = compute_trading_metrics(portfolio_values)
    metrics['final_value'] = final_value
    metrics['initial_value'] = initial_balance

    return {
        'metrics': metrics,
        'portfolio_values': portfolio_values,
        'prices': list(prices),
    }


def print_comparison(rl_results, bnh_results):
    """Print comparison table."""
    # Average metrics across episodes - only numeric fields
    numeric_keys = [k for k in rl_results[0]['metrics'].keys()
                   if isinstance(rl_results[0]['metrics'][k], (int, float, np.floating, np.integer))]
    rl_metrics = {k: np.mean([r['metrics'][k] for r in rl_results]) for k in numeric_keys}
    bnh_metrics = bnh_results['metrics']

    print("\n" + "=" * 70)
    print("  RL AGENT vs BUY-AND-HOLD COMPARISON REPORT")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'RL Agent':>15} {'Buy & Hold':>15} {'Difference':>15}")
    print("-" * 70)

    rl_return = float(rl_metrics['total_return'])
    bnh_return = float(bnh_metrics['total_return'])
    rl_outperform = rl_return - bnh_return

    print(f"{'Total Return':<25} {rl_return:>14.2f}% {bnh_return:>14.2f}% {rl_outperform:>+14.2f}%")

    rl_sharpe = float(rl_metrics['sharpe_ratio'])
    bnh_sharpe = float(bnh_metrics['sharpe_ratio'])
    print(f"{'Sharpe Ratio':<25} {rl_sharpe:>15.3f} {bnh_sharpe:>15.3f} {rl_sharpe - bnh_sharpe:>+15.3f}")

    rl_dd = float(rl_metrics['max_drawdown'])
    bnh_dd = float(bnh_metrics['max_drawdown'])
    print(f"{'Max Drawdown':<25} {rl_dd:>14.2f}% {bnh_dd:>14.2f}% {rl_dd - bnh_dd:>+14.2f}%")

    rl_wr = float(rl_metrics['win_rate'])
    bnh_wr = float(bnh_metrics['win_rate'])
    print(f"{'Win Rate':<25} {rl_wr:>14.2f}% {bnh_wr:>14.2f}% {rl_wr - bnh_wr:>+14.2f}%")

    rl_vol = float(rl_metrics['volatility'])
    bnh_vol = float(bnh_metrics['volatility'])
    print(f"{'Volatility':<25} {rl_vol:>15.3f} {bnh_vol:>15.3f} {rl_vol - bnh_vol:>+15.3f}")

    rl_final = float(np.mean([r['metrics']['final_value'] for r in rl_results]))
    bnh_final = float(bnh_metrics['final_value'])
    print(f"{'Final Value':<25} ${rl_final:>14,.2f} ${bnh_final:>14,.2f} ${rl_final - bnh_final:>+14,.2f}")

    rl_trades = int(np.mean([r['metrics']['n_trades'] for r in rl_results]))
    print(f"{'Num Trades':<25} {rl_trades:>15} {'N/A':>15} {'N/A':>15}")

    print("\n" + "-" * 70)

    # Action distribution
    rl_actions = rl_results[0]['metrics'].get('action_distribution', {})
    print("\nAction Distribution (RL Agent):")
    total_actions = sum(rl_actions.values()) if rl_actions else 0
    for action, count in sorted(rl_actions.items()):
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        action_name = {0: 'Hold', 1: 'Buy', 2: 'Sell'}.get(action, f'Action {action}')
        print(f"  {action_name:<10}: {count:>6} ({pct:>5.1f}%)")

    print("\n" + "=" * 70)

    # Winner determination
    if rl_sharpe > bnh_sharpe:
        print("  WINNER: RL Agent (higher Sharpe ratio)")
    elif bnh_sharpe > rl_sharpe:
        print("  WINNER: Buy-and-Hold (higher Sharpe ratio)")
    else:
        print("  TIE: Both strategies have equal Sharpe ratio")

    if rl_return > bnh_return:
        print(f"  RL Agent outperformed by {rl_outperform:.2f}% in total return")
    else:
        print(f"  Buy-and-Hold outperformed by {-rl_outperform:.2f}% in total return")

    print("=" * 70)

    return {
        'rl_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in rl_metrics.items()},
        'bnh_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in bnh_metrics.items()},
        'outperformance': rl_outperform,
        'sharpe_diff': rl_sharpe - bnh_sharpe,
    }


def main():
    print("=" * 70)
    print("  RL TRADING AGENT - BACKTEST & COMPARISON")
    print("=" * 70)

    # Configuration
    config = {
        'initial_balance': 10000.0,
        'buy_pct': 0.10,
        'sell_pct': 0.10,
        'commission': 0.0004,
        'stop_loss_pct': -0.05,
        'take_profit_pct': 0.10,
    }

    # Load data
    DATA_PATH = 'D:/Hive/Data/trading_repos/RL-Crypto-Trading-Bot/btc_data.csv'
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows of data")

    # Use last 20% for testing (simulating held-out data)
    train_end = int(len(df) * 0.8)
    test_df = df.iloc[train_end:].reset_index(drop=True)
    print(f"Test set: {len(test_df)} rows")

    # Save date info before prepare_crypto_data removes non-numeric columns
    test_start_date = str(test_df.iloc[0]['timestamp'])
    test_end_date = str(test_df.iloc[-1]['timestamp'])
    print(f"Test period: {test_start_date} to {test_end_date}")

    # Prepare data (removes non-numeric columns like timestamp)
    test_df = prepare_crypto_data(test_df)

    # Run RL simulation (random agent with action masking)
    print("\nRunning RL simulation (3 episodes)...")
    rl_results = run_rl_simulation(test_df, config, n_episodes=3)

    # Run buy-and-hold
    print("Running Buy-and-Hold baseline...")
    bnh_results = run_buy_and_hold(test_df, config)

    # Print comparison
    comparison = print_comparison(rl_results, bnh_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "D:/量化交易系统/量化之神/quant_trading/experiments"

    results_data = {
        'timestamp': timestamp,
        'config': config,
        'test_data_info': {
            'n_rows': len(test_df),
            'start_date': test_start_date,
            'end_date': test_end_date,
        },
        'comparison': comparison,
        'rl_episodes': [
            {
                'episode': r['metrics'].get('episode', 0),
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r['metrics'].items()},
                'n_actions': len(r['actions']),
            }
            for r in rl_results
        ],
        'buy_hold': {
            'metrics': comparison['bnh_metrics'],
        }
    }

    # Save JSON
    json_path = os.path.join(output_dir, f"rl_backtest_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, f"rl_portfolio_comparison_{timestamp}.csv")
    portfolio_df = pd.DataFrame({
        'step': range(len(bnh_results['portfolio_values'])),
        'buy_hold': bnh_results['portfolio_values'],
    })
    # Add RL episode values
    for i, r in enumerate(rl_results):
        max_len = min(len(r['portfolio_values']), len(portfolio_df))
        portfolio_df[f'rl_ep{i+1}'] = [None] * len(portfolio_df)
        portfolio_df.loc[:max_len-1, f'rl_ep{i+1}'] = r['portfolio_values'][:max_len]

    portfolio_df.to_csv(csv_path, index=False)
    print(f"Portfolio values saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("  BACKTEST COMPLETE")
    print("=" * 70)

    return results_data


if __name__ == "__main__":
    main()
