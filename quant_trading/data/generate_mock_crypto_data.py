# -*- coding: utf-8 -*-
"""
生成模拟Crypto数据用于V36策略测试

Usage:
    python generate_mock_crypto_data.py
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

import numpy as np


def generate_ohlcv(
    initial_price: float,
    n_days: int,
    volatility: float = 0.03,
    trend: float = 0.0,
) -> List[List]:
    """
    生成模拟OHLCV数据

    Args:
        initial_price: 初始价格
        n_days: 天数
        volatility: 波动率 (日波动)
        trend: 趋势 (正=上涨, 负=下跌)
    """
    ohlcv = []
    price = initial_price

    for day in range(n_days):
        # 生成随机涨跌
        daily_return = np.random.normal(trend, volatility)
        close = price * (1 + daily_return)

        # 生成日内OHLC
        high_mult = abs(np.random.normal(1.01, 0.01))
        low_mult = abs(np.random.normal(0.99, 0.01))

        high = max(price, close) * high_mult
        low = min(price, close) * low_mult
        open_price = price * (1 + np.random.normal(0, volatility * 0.5))

        # 生成成交量 (与价格变动相关)
        base_volume = initial_price * 1000
        volume_mult = 1 + abs(daily_return) * 10
        volume = base_volume * volume_mult * np.random.uniform(0.5, 1.5)

        # 时间戳 (毫秒)
        timestamp = int((datetime.now() - timedelta(days=n_days - day - 1)).timestamp() * 1000)

        ohlcv.append([
            timestamp,
            float(open_price),
            float(high),
            float(low),
            float(close),
            float(volume)
        ])

        price = close

    return ohlcv


def generate_crypto_data(
    output_file: str = None,
    n_days: int = 365,
) -> Dict[str, Any]:
    """
    生成多个交易对的模拟数据
    """
    # 交易对配置
    symbols_config = {
        "BTC/USDT": {"name": "Bitcoin", "initial_price": 45000, "volatility": 0.025, "trend": 0.001},
        "ETH/USDT": {"name": "Ethereum", "initial_price": 2500, "volatility": 0.03, "trend": 0.0008},
        "BNB/USDT": {"name": "BNB", "initial_price": 350, "volatility": 0.035, "trend": 0.0005},
        "SOL/USDT": {"name": "Solana", "initial_price": 100, "volatility": 0.05, "trend": 0.001},
        "XRP/USDT": {"name": "Ripple", "initial_price": 0.55, "volatility": 0.04, "trend": 0.0003},
        "ADA/USDT": {"name": "Cardano", "initial_price": 0.45, "volatility": 0.045, "trend": 0.0002},
        "DOGE/USDT": {"name": "Dogecoin", "initial_price": 0.08, "volatility": 0.06, "trend": 0.0001},
        "AVAX/USDT": {"name": "Avalanche", "initial_price": 35, "volatility": 0.05, "trend": 0.0006},
    }

    data = {}

    print("=" * 60)
    print("Crypto 模拟数据生成")
    print("=" * 60)
    print(f"交易对数量: {len(symbols_config)}")
    print(f"数据天数: {n_days}")
    print("-" * 60)

    random.seed(42)
    np.random.seed(42)

    for symbol, config in symbols_config.items():
        name = config["name"]
        initial_price = config["initial_price"]
        volatility = config["volatility"]
        trend = config["trend"]

        print(f"生成 {symbol} ({name})...")

        ohlcv = generate_ohlcv(
            initial_price=initial_price,
            n_days=n_days,
            volatility=volatility,
            trend=trend,
        )

        latest_price = ohlcv[-1][4] if ohlcv else initial_price

        data[symbol] = {
            "symbol": symbol,
            "name": name,
            "ohlcv": ohlcv,
            "count": len(ohlcv),
            "latest_price": latest_price,
            "timeframe": "1d",
            "fetch_time": datetime.now().isoformat(),
        }

        print(f"  {symbol}: {len(ohlcv)} 条K线, 最新价格: {latest_price:.4f}")

    # 保存到JSON
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), 'crypto', 'crypto_data.json')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"数据保存至: {output_file}")
    print(f"总计: {sum(d['count'] for d in data.values())} 条K线")

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成Crypto模拟数据')
    parser.add_argument('--days', '-d', type=int, default=365,
                        help='数据天数 (默认: 365)')
    parser.add_argument('--output', '-o', type=str,
                        help='输出文件路径')

    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(__file__),
            'crypto',
            'crypto_data.json'
        )

    generate_crypto_data(output_file, args.days)