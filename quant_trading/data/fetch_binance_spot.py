# -*- coding: utf-8 -*-
"""
Binance 现货数据获取脚本 - 使用原生 requests

直接使用 Binance 现货 API，避免 ccxt 的期货市场加载问题

Usage:
    python fetch_binance_spot.py
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

import requests


# 默认交易对配置
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
]

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'crypto', 'binance_spot_data.json')


def fetch_klines(symbol: str, interval: str = '1d', limit: int = 500) -> List[List]:
    """
    获取K线数据

    Args:
        symbol: 交易对，如 'BTCUSDT'
        interval: K线周期，如 '1d', '1h', '15m'
        limit: 返回数量 (max 1000)

    Returns:
        K线数据列表
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_spot_data(
    symbols: List[str] = None,
    days: int = 500,
    output_file: str = None,
) -> Dict[str, Any]:
    """
    获取现货数据

    Args:
        symbols: 交易对列表
        days: 数据天数
        output_file: 输出文件路径

    Returns:
        包含所有交易对数据的字典
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    if output_file is None:
        output_file = OUTPUT_FILE

    print("=" * 60)
    print("Binance 现货数据获取")
    print("=" * 60)
    print(f"交易对数量: {len(symbols)}")
    print(f"数据天数: {days}")
    print("-" * 60)

    data = {}
    success_count = 0

    for symbol in symbols:
        # 格式化显示名称
        display_name = symbol.replace('USDT', '/USDT') if not symbol.endswith('/USDT') else symbol

        print(f"获取 {display_name}...", end=" ")

        try:
            # 计算起始时间
            limit = min(days, 1000)  # Binance max 1000

            klines = fetch_klines(symbol, interval='1d', limit=limit)

            if klines:
                # 获取最新价格
                latest = fetch_ticker(symbol)
                latest_price = latest.get('lastPrice', 0)

                data[symbol] = {
                    "symbol": display_name,
                    "raw_symbol": symbol,
                    "ohlcv": klines,
                    "count": len(klines),
                    "latest_price": float(latest_price),
                    "timeframe": "1d",
                    "fetch_time": datetime.now().isoformat(),
                }

                print(f"[OK] {len(klines)} 条K线, 价格: {float(latest_price):.4f}")
                success_count += 1

                # 避免频率限制
                time.sleep(0.2)
            else:
                print("[EMPTY] 无数据")

        except Exception as e:
            print(f"[FAIL] 错误: {str(e)[:50]}")

    # 保存数据
    if data:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"获取完成: {success_count}/{len(symbols)} 个交易对")
    if data:
        print(f"数据保存至: {output_file}")

    return data


def fetch_ticker(symbol: str) -> Dict:
    """获取24小时ticker数据"""
    url = f"https://api.binance.com/api/v3/ticker/24hr"
    params = {'symbol': symbol}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='获取Binance现货数据')
    parser.add_argument('--symbols', '-s', nargs='+', help='交易对，如: BTCUSDT ETHUSDT')
    parser.add_argument('--days', '-d', type=int, default=500, help='天数 (默认: 500)')

    args = parser.parse_args()

    if args.symbols:
        symbols = [s.upper().replace('/', '') for s in args.symbols]
    else:
        symbols = DEFAULT_SYMBOLS

    fetch_spot_data(symbols, args.days)


if __name__ == '__main__':
    main()