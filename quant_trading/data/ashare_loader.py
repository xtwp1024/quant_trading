# -*- coding: utf-8 -*-
"""
A股数据加载器 - 从JSON格式加载数据用于V36回测
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def load_ashare_json(json_path: str) -> pd.DataFrame:
    """
    从JSON文件加载A股数据

    JSON格式:
    {
        "000001": {
            "symbol": "000001",
            "name": "平安银行",
            "ohlcv": [[open, high, low, close, volume], ...],
            "volume_ratio": 1.5,
            "count": 200,
            "latest_price": 12.34
        },
        ...
    }

    Returns:
        DataFrame with columns: date, code, name, open, high, low, close, volume
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    latest_date = datetime(2026, 4, 7)  # 假设最新数据日期

    for code, stock_data in data.items():
        ohlcv = stock_data.get('ohlcv', [])
        count = stock_data.get('count', len(ohlcv))
        name = stock_data.get('name', code)

        if not ohlcv:
            continue

        # 生成日期（假设是日线数据，从最新日期往前推）
        dates = [latest_date - timedelta(days=i) for i in range(count - 1, -1, -1)]

        for dt, ohlcv_item in zip(dates, ohlcv):
            if len(ohlcv_item) >= 5:
                records.append({
                    'date': dt.strftime('%Y-%m-%d'),
                    'code': code,
                    'name': name,
                    'open': ohlcv_item[0],
                    'high': ohlcv_item[1],
                    'low': ohlcv_item[2],
                    'close': ohlcv_item[3],
                    'volume': ohlcv_item[4],
                })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_ashare_directory(directory: str) -> pd.DataFrame:
    """
    从目录加载所有A股JSON文件

    Args:
        directory: 包含JSON文件的目录

    Returns:
        合并的DataFrame
    """
    json_path = os.path.join(directory, 'all_stocks_data.json')
    if os.path.exists(json_path):
        return load_ashare_json(json_path)

    # 否则合并所有cache_*.json文件
    all_data = []
    for filename in os.listdir(directory):
        if filename.startswith('cache_') and filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                df = load_ashare_json(filepath)
                all_data.append(df)
            except Exception as e:
                print(f"加载 {filename} 失败: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    raise FileNotFoundError(f"未找到数据文件: {json_path}")


class V36DataLoader:
    """V36策略数据加载器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # 默认路径
            self.data_dir = os.path.join(
                os.path.dirname(__file__),
                'ashare'
            )
        else:
            self.data_dir = data_dir

    def load(self) -> pd.DataFrame:
        """加载数据"""
        return load_ashare_directory(self.data_dir)

    def filter_stocks(self, df: pd.DataFrame, stock_pool: Dict[str, str]) -> pd.DataFrame:
        """根据股票池过滤数据"""
        codes = list(stock_pool.keys())
        return df[df['code'].isin(codes)]

    def get_stock_pool_data(self, stock_pool: Dict[str, str]) -> pd.DataFrame:
        """获取股票池数据"""
        df = self.load()
        return self.filter_stocks(df, stock_pool)


if __name__ == '__main__':
    # 测试加载
    loader = V36DataLoader('data/ashare')
    df = loader.load()
    print(f"加载数据: {len(df)} 条")
    print(f"股票数量: {df['code'].nunique()}")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(df.head())
