# -*- coding: utf-8 -*-
"""
Crypto 数据加载器 - 从JSON格式加载数据用于V36回测
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd


def load_crypto_json(json_path: str) -> pd.DataFrame:
    """
    从JSON文件加载Crypto数据

    支持两种格式:
    1. 自定义格式:
       {"BTC/USDT": {"symbol": "BTC/USDT", "name": "Bitcoin", "ohlcv": [[ts, o, h, l, c, v], ...], ...}}

    2. Binance 格式:
       {"BTCUSDT": {"symbol": "BTC/USDT", "raw_symbol": "BTCUSDT", "ohlcv": [[ts, o, h, l, c, v], ...], ...}}

    Returns:
        DataFrame with columns: date, code, name, open, high, low, close, volume
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []

    for symbol, crypto_data in data.items():
        ohlcv = crypto_data.get('ohlcv', [])
        count = crypto_data.get('count', len(ohlcv))
        # 支持多种名称格式
        name = crypto_data.get('name', crypto_data.get('symbol', symbol))
        raw_symbol = crypto_data.get('raw_symbol', symbol)

        if not ohlcv:
            continue

        for ohlcv_item in ohlcv:
            if len(ohlcv_item) >= 6:
                ts = ohlcv_item[0]
                # Binance API 返回的是毫秒时间戳
                dt = datetime.fromtimestamp(ts / 1000) if ts > 1e10 else datetime.fromtimestamp(ts)
                # 标准化 symbol 格式 (BTCUSDT -> BTC/USDT)
                if '/' not in symbol:
                    normalized_symbol = symbol[:-4] + '/' + symbol[-4:]  # BTCUSDT -> BTC/USDT
                else:
                    normalized_symbol = symbol
                records.append({
                    'date': dt.strftime('%Y-%m-%d'),
                    'code': symbol.replace('/', '_'),  # BTC/USDT -> BTC_USDT
                    'symbol': normalized_symbol,
                    'name': name or normalized_symbol.split('/')[0],  # 使用 symbol 部分作为 name
                    'open': float(ohlcv_item[1]),
                    'high': float(ohlcv_item[2]),
                    'low': float(ohlcv_item[3]),
                    'close': float(ohlcv_item[4]),
                    'volume': float(ohlcv_item[5]),
                })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_crypto_directory(directory: str) -> pd.DataFrame:
    """
    从目录加载所有Crypto JSON文件

    Args:
        directory: 包含JSON文件的目录

    Returns:
        合并的DataFrame
    """
    json_path = os.path.join(directory, 'crypto_data.json')
    if os.path.exists(json_path):
        return load_crypto_json(json_path)

    raise FileNotFoundError(f"未找到数据文件: {json_path}")


class CryptoDataLoader:
    """Crypto策略数据加载器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # 使用相对于 quant_trading 目录的路径
            quant_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(quant_dir, 'data', 'crypto')
        else:
            self.data_dir = data_dir

    def load(self) -> pd.DataFrame:
        """加载数据"""
        return load_crypto_directory(self.data_dir)

    def filter_symbols(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """根据交易对过滤数据"""
        # 标准化symbol格式
        normalized = [s.replace('_', '/') for s in df['symbol'].unique()]
        return df[df['symbol'].isin(symbols)]

    def get_symbols_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取交易对数据"""
        df = self.load()
        return self.filter_symbols(df, symbols)


if __name__ == '__main__':
    # 测试加载
    loader = CryptoDataLoader('data/crypto')
    try:
        df = loader.load()
        print(f"加载数据: {len(df)} 条")
        print(f"交易对数量: {df['symbol'].nunique()}")
        print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
        print(df.head())
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 fetch_crypto_data.py 获取数据")