"""TimesFM 集成测试

Usage:
    cd D:/量化交易系统/量化之神
    .venv\Scripts\python quant_trading\experiments\test_timesfm.py
"""

import sys
import os

# 确保 quant_trading 模块可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from quant_trading.signal import TimesFMGenerator


def create_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成合成 OHLCV 数据用于测试"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = low + np.random.rand(n) * (high - low)
    volume = np.random.randint(1000, 10000, n)

    return pd.DataFrame(
        {
            "timestamp": dates.astype(np.int64) // 10**6,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_timesfm_import():
    """测试 TimesFM 能否正常导入"""
    print("=" * 50)
    print("Test 1: TimesFM 模型导入")
    print("=" * 50)
    try:
        import timesfm
        print(f"✓ TimesFM version: {timesfm.__version__ if hasattr(timesfm, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        print(f"✗ TimesFM 导入失败: {e}")
        return False


def test_signal_generation():
    """测试信号生成"""
    print("\n" + "=" * 50)
    print("Test 2: TimesFMGenerator 信号生成")
    print("=" * 50)

    df = create_synthetic_ohlcv(n=300)
    print(f"✓ 生成测试数据: {len(df)} 条 OHLCV")

    gen = TimesFMGenerator(
        symbol="TEST/USDT",
        horizon=12,
        context_len=128,
        buy_threshold=0.01,
        sell_threshold=-0.01,
    )
    print("✓ TimesFMGenerator 实例化成功")

    print("\n首次调用（需加载模型，约10-30秒）...")
    signals = gen.generate(df)

    if not signals:
        print("  (无信号，预测收益在阈值内)")
    else:
        for sig in signals:
            print(f"\n✓ 信号生成:")
            print(f"  type: {sig.type.value}")
            print(f"  price: {sig.price:.4f}")
            print(f"  strength: {sig.strength:.4f}")
            print(f"  reason: {sig.reason}")
            print(f"  pred_return: {sig.metadata.get('pred_return', 'N/A')}")

    return True


def test_get_forecast():
    """测试纯预测接口"""
    print("\n" + "=" * 50)
    print("Test 3: TimesFMGenerator.get_forecast()")
    print("=" * 50)

    df = create_synthetic_ohlcv(n=200)
    gen = TimesFMGenerator(symbol="TEST/USDT", horizon=24, context_len=100)

    pred_return, point, quantile = gen.get_forecast(df)
    print(f"✓ 预测收益: {pred_return*100:.4f}%")
    print(f"  point_forecast shape: {point.shape}")
    print(f"  quantile_forecast shape: {quantile.shape}")

    return True


if __name__ == "__main__":
    ok1 = test_timesfm_import()
    if ok1:
        test_signal_generation()
        test_get_forecast()

    print("\n" + "=" * 50)
    print("所有测试完成！")
    print("=" * 50)
