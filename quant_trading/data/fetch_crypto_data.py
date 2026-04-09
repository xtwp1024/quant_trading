# -*- coding: utf-8 -*-
"""
Crypto 数据获取脚本 - 从Binance获取历史K线数据

Usage:
    python fetch_crypto_data.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("错误: 请先安装 ccxt")
    print("  pip install ccxt")
    sys.exit(1)


# 默认交易对配置
DEFAULT_SYMBOLS = {
    "BTC/USDT": {"name": "Bitcoin", "max_daily_bars": 500},
    "ETH/USDT": {"name": "Ethereum", "max_daily_bars": 500},
    "BNB/USDT": {"name": "BNB", "max_daily_bars": 500},
    "SOL/USDT": {"name": "Solana", "max_daily_bars": 500},
    "XRP/USDT": {"name": "Ripple", "max_daily_bars": 500},
    "ADA/USDT": {"name": "Cardano", "max_daily_bars": 500},
    "DOGE/USDT": {"name": "Dogecoin", "max_daily_bars": 500},
    "AVAX/USDT": {"name": "Avalanche", "max_daily_bars": 500},
    "DOT/USDT": {"name": "Polkadot", "max_daily_bars": 500},
    "MATIC/USDT": {"name": "Polygon", "max_daily_bars": 500},
}

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'crypto')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'crypto_data.json')


async def fetch_symbol_data(exchange, symbol: str, max_bars: int = 500) -> Dict[str, Any]:
    """获取单个交易对的K线数据"""
    config = DEFAULT_SYMBOLS.get(symbol, {"name": symbol, "max_daily_bars": max_bars})
    name = config["name"]

    print(f"  获取 {symbol} ({name})...")

    try:
        # 设置时间范围：最近 max_bars 天
        end_time = exchange.milliseconds()
        start_time = end_time - (max_bars * 86400 * 1000)  # 按天计算

        # 获取日线数据
        ohlcv = await exchange.fetch_ohlcv(
            symbol,
            timeframe='1d',
            since=start_time,
            limit=max_bars
        )

        # 获取最新价格
        ticker = await exchange.fetch_ticker(symbol)
        latest_price = ticker.get('last', 0)

        result = {
            "symbol": symbol,
            "name": name,
            "ohlcv": ohlcv,
            "count": len(ohlcv),
            "latest_price": latest_price,
            "timeframe": "1d",
            "fetch_time": datetime.now().isoformat(),
        }

        print(f"    {symbol}: {len(ohlcv)} 条K线, 最新价格: {latest_price:.4f}")
        return symbol, result

    except Exception as e:
        print(f"    {symbol} 获取失败: {e}")
        return symbol, None


async def fetch_all_data(symbols: Dict[str, Dict] = None, output_file: str = OUTPUT_FILE) -> Dict:
    """获取所有交易对的数据"""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    print("=" * 60)
    print("Crypto 数据获取")
    print("=" * 60)
    print(f"交易所: Binance")
    print(f"时间范围: 最近 {max(v['max_daily_bars'] for v in symbols.values())} 天")
    print(f"交易对数量: {len(symbols)}")
    print("-" * 60)

    # 创建交易所实例 (明确使用现货)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # 现货模式，避免连接期货 API
            'broker': 'YOUR_BROKER',  # 可选：添加经纪商标记
        },
        # 明确指定现货 API URL
        'urls': {
            'api': {
                'public': 'https://api.binance.com/api/v3',
                'private': 'https://api.binance.com/api/v3',
            },
        },
    })

    try:
        # 加载市场数据
        await exchange.load_markets()
        print(f"市场加载完成: {len(exchange.markets)} 个交易对")
        print("-" * 60)

        # 获取所有数据
        tasks = [
            fetch_symbol_data(exchange, symbol, config.get('max_daily_bars', 500))
            for symbol, config in symbols.items()
        ]
        results = await asyncio.gather(*tasks)

        # 构建数据字典
        data = {}
        success_count = 0
        for symbol, result in results:
            if result is not None:
                data[symbol] = result
                success_count += 1

        # 保存到JSON
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("-" * 60)
        print(f"获取完成: {success_count}/{len(symbols)} 个交易对")
        print(f"数据保存至: {output_file}")

        # 输出摘要
        print("-" * 60)
        print("数据摘要:")
        for symbol, crypto_data in data.items():
            print(f"  {symbol}: {crypto_data['count']} 条K线, "
                  f"价格范围: {min(o[4] for o in crypto_data['ohlcv']):.2f} ~ "
                  f"{max(o[4] for o in crypto_data['ohlcv']):.2f}")

        return data

    finally:
        await exchange.close()


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description='获取Crypto历史K线数据')
    parser.add_argument('--symbols', '-s', nargs='+',
                        help='指定交易对，例如: BTC/USDT ETH/USDT')
    parser.add_argument('--days', '-d', type=int, default=500,
                        help='获取天数 (默认: 500)')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_FILE,
                        help=f'输出文件路径 (默认: {OUTPUT_FILE})')

    args = parser.parse_args()

    # 如果指定了交易对，只获取指定的
    if args.symbols:
        symbols = {s: {"name": s.split('/')[0], "max_daily_bars": args.days}
                   for s in args.symbols}
    else:
        symbols = DEFAULT_SYMBOLS

    # 更新天数
    for s in symbols:
        symbols[s]['max_daily_bars'] = args.days

    asyncio.run(fetch_all_data(symbols, args.output))


if __name__ == '__main__':
    main()