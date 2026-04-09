#!/usr/bin/env python3
"""
量化之神 CLI - Command Line Interface
Unified Quant Trading System
"""

import sys
import argparse
from typing import Optional

from quant_trading.options.pricing.black_scholes import BlackScholes
from quant_trading.optimization.kelly_calculator import KellyCalculator
from quant_trading.backtest import BacktestEngine
from quant_trading.exchanges import BinanceAdapter, OKXAdapter
from quant_trading.signal.stock_pool import StockPoolManager, PoolType, AStockDataProvider
from quant_trading.strategy.advanced.v36_strategy import V36Backtester, V36Params, DEFAULT_STOCK_POOL


def cmd_info(args):
    """Show system information"""
    print("=" * 60)
    print("量化之神 (God of Quant Trading)")
    print("=" * 60)
    print("Version: 1.0.0")
    print("Author: Quant God Team")
    print()
    print("Available Modules:")
    print("  - BlackScholes: Options pricing model")
    print("  - KellyCalculator: Kelly criterion position sizing")
    print("  - BacktestEngine: Strategy backtesting")
    print("  - BinanceAdapter: Binance exchange")
    print("  - OKXAdapter: OKX exchange")
    print("  - StockTradingEnv: RL trading environment (Gymnasium)")
    print("  - AkshareMarketDataClient: A-share market data")
    print("  - IndicatorEngine: TA-Lib compatible indicator engine")
    print("  - MarketService: MCP tool interface for market data")
    print("  - Optopsy: Options backtesting (38 strategies, 90+ signals)")
    print("  - CognitiveUnit, DreamEngine: Five Force cognitive modules")
    print("  - GeneBank, StrategyBreeder: Gene Lab evolution modules")
    print()
    print("Working Features:")
    print("  [X] Black-Scholes options pricing + Greeks")
    print("  [X] Kelly Criterion calculator")
    print("  [X] BacktestEngine")
    print("  [X] Exchange adapters (Binance, OKX)")
    print("  [X] DAPO RL training (StockTradingEnv + MLPActorCritic)")
    print("  [X] A-share data via akshare (Eastmoney API)")
    print("  [X] Indicator engine (pandas fallback, TA-Lib optional)")
    print("  [X] MCP market data service (Kline, RSI, MACD, MA, etc.)")
    print("  [X] Optopsy options strategies (long_straddles, iron_condor, ...)")
    print("  [X] Optopsy signals (RSI, IV_rank, MACD, Supertrend, ...)")
    print("  [ ] Five Force cognitive architecture (partial)")
    print("  [ ] Gene Lab full evolution pipeline (partial)")
    print()


def cmd_black_scholes(args):
    """Calculate Black-Scholes option price"""
    bs = BlackScholes(
        S=args.spot,
        K=args.strike,
        T=args.time,
        r=args.rate,
        sigma=args.volatility
    )
    if args.call:
        price = bs.call_price()
        print(f"Call option price: {price:.4f}")
    elif args.put:
        price = bs.put_price()
        print(f"Put option price: {price:.4f}")
    else:
        print(f"Call option price: {bs.call_price():.4f}")
        print(f"Put option price: {bs.put_price():.4f}")
        print(f"Delta (call): {bs.call_delta():.4f}")
        print(f"Gamma: {bs.gamma():.4f}")
        print(f"Vega: {bs.vega():.4f}")
        print(f"Theta (call): {bs.theta_call():.4f}")


def cmd_kelly(args):
    """Calculate Kelly criterion"""
    kelly = KellyCalculator()
    fraction = kelly.calculate_kelly_fraction(
        win_rate=args.win_rate,
        avg_win=args.avg_win,
        avg_loss=args.avg_loss
    )
    print(f"Kelly fraction: {fraction:.4f} ({fraction*100:.1f}%)")
    print(f"Recommended position: {fraction*100:.1f}% of capital")


def cmd_backtest(args):
    """Run backtest"""
    print("BacktestEngine initialized")
    print(f"  Initial capital: {args.capital}")
    print(f"  Commission: {args.commission}")
    print(f"  Slippage: {args.slippage}")
    print()
    print("To run a backtest, use the Python API:")
    print("  from quant_trading.backtest import BacktestEngine")
    print("  engine = BacktestEngine(initial_capital=10000)")
    print("  results = engine.run(data, signals)")


def cmd_exchange(args):
    """Test exchange connection"""
    config = {
        "apiKey": args.api_key or "",
        "secret": args.api_secret or "",
        "testnet": args.testnet,
    }

    if args.exchange == "binance":
        adapter = BinanceAdapter(config)
        print(f"BinanceAdapter created (testnet={args.testnet})")
        print("  Methods: connect(), fetch_ohlcv(), fetch_ticker(), create_order()")
    elif args.exchange == "okx":
        adapter = OKXAdapter(config)
        print(f"OKXAdapter created (testnet={args.testnet})")
        print("  Methods: connect(), fetch_ohlcv(), fetch_ticker(), create_order()")


def cmd_stock_pool(args):
    """Stock Pool Classification with Real A-Share Data"""
    import numpy as np

    manager = StockPoolManager()

    # 如果指定了股票代码，使用真实数据
    if args.symbols:
        print("正在连接 akshare 获取真实A股数据...")
        data_provider = AStockDataProvider()

        symbols = args.symbols
        print(f"获取 {len(symbols)} 只股票数据...")

        symbols_ohlcv, volume_ratios = data_provider.fetch_batch(symbols)

        if not symbols_ohlcv:
            print("错误: 未能获取任何股票数据")
            return

        print(f"成功获取 {len(symbols_ohlcv)} 只股票数据")

    elif args.popular:
        # 获取热门股票
        print("正在获取热门股票（成交额排名）...")
        data_provider = AStockDataProvider()
        top_stocks = data_provider.get_top_stocks_by_amount(limit=args.popular)

        if not top_stocks:
            print("错误: 未能获取热门股票")
            return

        print(f"获取到 {len(top_stocks)} 只热门股票")

        symbols = [s["symbol"] for s in top_stocks]
        symbols_ohlcv, volume_ratios = data_provider.fetch_batch(symbols)

        if not symbols_ohlcv:
            print("错误: 未能获取股票数据")
            return

        print(f"成功获取 {len(symbols_ohlcv)} 只股票数据")

    else:
        # 使用模拟数据演示
        print("使用模拟数据进行演示 (可添加 --symbols 代码 或 --popular N 获取真实数据)")
        print()

        def gen_data(pool_type, n=200, base=100.0):
            np.random.seed(hash(pool_type) % 2**32)
            if pool_type == "scalping":
                vol, trend = 0.03, 0.0001
            elif pool_type == "trend":
                vol, trend = 0.015, 0.002
            else:
                vol, trend = 0.01, -0.0001

            closes = [base]
            for i in range(1, n):
                closes.append(closes[-1] * (1 + np.random.normal(trend, vol)))
            closes = np.array(closes)

            opens = closes * (1 + np.random.uniform(-0.005, 0.005, n))
            highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, vol * 0.5, n)))
            lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, vol * 0.5, n)))
            volumes = np.random.uniform(1000, 5000, n)
            return np.column_stack([opens, highs, lows, closes, volumes])

        test_stocks = {
            "贵州茅台": ("trend", 1800),
            "宁德时代": ("trend", 200),
            "比亚迪": ("scalping", 250),
            "中国平安": ("watch", 45),
            "万科A": ("watch", 8),
            "东方财富": ("trend", 15),
            "隆基绿能": ("watch", 25),
            "紫金矿业": ("scalping", 12),
        }

        symbols_ohlcv = {}
        volume_ratios = {}
        for name, (ptype, price) in test_stocks.items():
            symbols_ohlcv[name] = gen_data(ptype, 200, price)
            volume_ratios[name] = np.random.uniform(0.8, 2.0)

    manager.update_pool(symbols_ohlcv, volume_ratios)
    summary = manager.get_pool_summary()

    print()
    print("=" * 60)
    print("Stock Pool Classification Results")
    print("=" * 60)

    for pool_name, data in summary.items():
        print(f"\n{pool_name}类 ({data['count']}只):")
        if data["details"]:
            for detail in data["details"]:
                signal = detail.get("signal", "-")
                print(f"  - {detail['symbol']}: {signal}")
        else:
            print("  (empty)")

    # 弱势企稳候选
    candidates = manager.get_stabilization_candidates()
    if candidates:
        print("\n[Stabilization Candidates]")
        for c in candidates:
            print(f"  - {c['symbol']}: {c['signal']}")

    # 趋势标的
    trend_stocks = manager.get_trend_stocks()
    if trend_stocks:
        print("\n[Trend Stocks]")
        for s in trend_stocks[:3]:
            print(f"  - {s['symbol']}: {s['signal']} (momentum={s['momentum']:+.1f}%)")


def cmd_v36(args):
    """V36 A股趋势策略回测"""
    import os

    print("=" * 70)
    print("V36 A股趋势策略 - 企稳 + 回踩动态支撑 + 强势回踩5日线")
    print("=" * 70)

    # 创建V36回测器
    params = V36Params(
        cash=args.capital,
        max_pos=args.max_pos,
        stop=-args.stop_loss,
        take=args.take_profit,
        time_stop=args.time_stop,
    )

    tester = V36Backtester(params)

    # 数据路径
    json_data_path = "data/ashare/all_stocks_data.json"
    csv_data_path = args.data or "a_stock_daily_data.csv"

    # 尝试多种数据格式
    data_loaded = False

    # 1. 尝试JSON格式 (A股JSON)
    if os.path.exists(json_data_path):
        print(f"\n从JSON加载数据: {json_data_path}")
        try:
            tester.load_ashare(json_data_path)
            data_loaded = True
            print(f"成功从JSON加载 {len(tester.pool.stocks)} 只股票")
        except Exception as e:
            print(f"JSON加载失败: {e}")

    # 2. 尝试CSV格式
    if not data_loaded and os.path.exists(csv_data_path):
        print(f"\n从CSV加载数据: {csv_data_path}")
        try:
            tester.load_csv(csv_data_path)
            data_loaded = True
            print(f"成功从CSV加载 {len(tester.pool.stocks)} 只股票")
        except Exception as e:
            print(f"CSV加载失败: {e}")

    if not data_loaded:
        print(f"\n错误: 未找到数据文件")
        print(f"  - JSON: {json_data_path}")
        print(f"  - CSV: {csv_data_path}")
        return

    if len(tester.pool.stocks) == 0:
        print("\n错误: 股票池为空，请检查股票代码是否匹配")
        print("\n默认股票池:")
        for code, name in list(DEFAULT_STOCK_POOL.items())[:10]:
            print(f"  {code}: {name}")
        return

    print(f"\n股票池数量: {len(tester.pool.stocks)}")
    print(f"初始资金: {params.cash:,.0f}")
    print(f"最大持仓: {params.max_pos}")
    print(f"止损: {params.stop*100:.0f}%")
    print(f"止盈: {params.take*100:.0f}%")
    print(f"时间止损: {params.time_stop}天")
    print()

    # 运行回测
    print("运行回测...")
    nav_df = tester.run()
    result = tester.get_result()

    # 输出结果
    print()
    print("=" * 70)
    print("回测结果")
    print("=" * 70)
    print(f"交易次数: {result['trade_count']}")
    print(f"收益率: {result['ret']:.1f}%")
    print(f"最大回撤: {result['max_drawdown']:.1f}%")
    print(f"最终资产: {result['final_asset']:,.0f}")
    print("=" * 70)

    # 保存净值曲线
    if args.output:
        nav_df.to_csv(args.output, index=False)
        print(f"\n净值曲线已保存: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="量化之神 - God of Quant Trading")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Show system information")

    # Black-Scholes command
    bs_parser = subparsers.add_parser("black-scholes", help="Black-Scholes options pricing")
    bs_parser.add_argument("--spot", type=float, default=100, help="Spot price")
    bs_parser.add_argument("--strike", type=float, default=105, help="Strike price")
    bs_parser.add_argument("--time", type=float, default=0.25, help="Time to expiry (years)")
    bs_parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    bs_parser.add_argument("--volatility", type=float, default=0.2, help="Volatility")
    bs_parser.add_argument("--call", action="store_true", help="Show call price only")
    bs_parser.add_argument("--put", action="store_true", help="Show put price only")
    bs_parser.set_defaults(func=cmd_black_scholes)

    # Kelly command
    kelly_parser = subparsers.add_parser("kelly", help="Kelly criterion calculation")
    kelly_parser.add_argument("--win-rate", type=float, default=0.55, help="Win rate (0-1)")
    kelly_parser.add_argument("--avg-win", type=float, default=100, help="Average win amount")
    kelly_parser.add_argument("--avg-loss", type=float, default=50, help="Average loss amount")
    kelly_parser.set_defaults(func=cmd_kelly)

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    backtest_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    backtest_parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate")
    backtest_parser.set_defaults(func=cmd_backtest)

    # Exchange command
    exchange_parser = subparsers.add_parser("exchange", help="Test exchange connection")
    exchange_parser.add_argument("exchange", choices=["binance", "okx"], help="Exchange name")
    exchange_parser.add_argument("--api-key", type=str, help="API key")
    exchange_parser.add_argument("--api-secret", type=str, help="API secret")
    exchange_parser.add_argument("--testnet", action="store_true", help="Use testnet")
    exchange_parser.set_defaults(func=cmd_exchange)

    # Stock Pool command
    stock_pool_parser = subparsers.add_parser("stock-pool", help="Stock pool classification")
    stock_pool_parser.add_argument("--symbols", nargs="*", help="Stock symbols to analyze (e.g., 000001 600519)")
    stock_pool_parser.add_argument("--popular", type=int, metavar="N", help="Get top N stocks by trading volume")
    stock_pool_parser.set_defaults(func=cmd_stock_pool)

    # V36 command
    v36_parser = subparsers.add_parser("v36", help="V36 A-share trend strategy backtest")
    v36_parser.add_argument("--data", type=str, default="a_stock_daily_data.csv", help="Data file path")
    v36_parser.add_argument("--capital", type=float, default=1000000, help="Initial capital")
    v36_parser.add_argument("--max-pos", type=int, default=8, help="Max positions")
    v36_parser.add_argument("--stop-loss", type=float, default=0.07, help="Stop loss rate")
    v36_parser.add_argument("--take-profit", type=float, default=0.20, help="Take profit rate")
    v36_parser.add_argument("--time-stop", type=int, default=6, help="Time stop days")
    v36_parser.add_argument("--output", type=str, help="Output CSV file for nav")
    v36_parser.set_defaults(func=cmd_v36)

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "black-scholes":
        cmd_black_scholes(args)
    elif args.command == "kelly":
        cmd_kelly(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "exchange":
        cmd_exchange(args)
    elif args.command == "stock-pool":
        cmd_stock_pool(args)
    elif args.command == "v36":
        cmd_v36(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
