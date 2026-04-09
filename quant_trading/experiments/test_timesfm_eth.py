"""TimesFM + ETH 实时测试

Usage:
    cd D:/量化交易系统/量化之神
    .venv\Scripts\python quant_trading\experiments\test_timesfm_eth.py
"""

import sys, os
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import importlib
import numpy as np
import pandas as pd
import requests
import time

# ── 数据获取 ──────────────────────────────────────────────────────────────────

BINANCE_API = "https://api.binance.com"
BINANCE_KEY = "8FRhMccDU4IX47Fe1FPcObwcOnPoRhXzLamLywSyxrhlgI8wvlNZjZXlkIbkK6B"


def fetch_with_retry(url, params=None, max_retries=4, timeout=20):
    headers = {"X-MBX-APIKEY": BINANCE_KEY}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return None


def fetch_eth_klines(interval="1h", limit=500):
    """获取 ETHUSDT K线数据"""
    intvl_map = {"1h": "1h", "4h": "4h", "1d": "1d", "15m": "15m"}
    intvl = intvl_map.get(interval, "1h")

    url = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": "ETHUSDT", "interval": intvl, "limit": limit}
    r = fetch_with_retry(url, params)
    if r is None or r.status_code != 200:
        print(f"  数据获取失败，status={r.status_code if r else None}")
        return None

    data = json.loads(r.text)
    rows = []
    for x in data:
        rows.append({
            "timestamp": int(x[0]),
            "open": float(x[1]),
            "high": float(x[2]),
            "low": float(x[3]),
            "close": float(x[4]),
            "volume": float(x[5]),
        })
    return pd.DataFrame(rows)


# ── TimesFM 测试 ─────────────────────────────────────────────────────────────

def test_timesfm_on_eth():
    print("=" * 60)
    print("TimesFM + ETHUSDT 实时测试")
    print("=" * 60)

    # 1. 获取数据
    print("\n[1] 获取 ETHUSDT K线数据...")
    df = fetch_eth_klines("1h", 500)
    if df is None:
        print("  无法获取数据，退出")
        return
    print(f"  ✓ 获取 {len(df)} 条 K线")
    print(f"  时间范围: {pd.to_datetime(df['timestamp'], unit='ms').iloc[0]} ~ {pd.to_datetime(df['timestamp'], unit='ms').iloc[-1]}")
    print(f"  最新价格: {df['close'].iloc[-1]:.2f}")

    # 2. 导入 TimesFMGenerator (绕过 quant_trading.__init__ 的 redis 依赖)
    print("\n[2] 导入 TimesFMGenerator...")
    try:
        import importlib.util
        # 用脚本所在目录的相对路径
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(os.path.dirname(_script_dir))
        # 如果路径解析出错（Desktop 拷贝干扰），用备用路径
        _signal_path = os.path.join(_project_root, "signal", "timesfm_generator.py")
        if not os.path.exists(_signal_path):
            _project_root = r"D:\量化交易系统\量化之神"
            _signal_path = os.path.join(_project_root, "quant_trading", "signal", "timesfm_generator.py")
        spec = importlib.util.spec_from_file_location(
            "timesfm_gen",
            _signal_path,
        )
        timesfm_mod = importlib.util.module_from_spec(spec)
        # 注入 mock types
        import sys
        class MockSignalType:
            BUY = "buy"
            SELL = "sell"
        class MockSignalDirection:
            LONG = 1
            SHORT = -1
            NEUTRAL = 0
        mock_types = type(sys)("quant_trading.signal.types")
        mock_types.SignalType = MockSignalType
        mock_types.SignalDirection = MockSignalDirection
        mock_types.Signal = object  # abstract enough
        sys.modules["quant_trading.signal.types"] = mock_types
        sys.modules["quant_trading.signal"] = mock_types
        spec.loader.exec_module(timesfm_mod)
        TimesFMGenerator = timesfm_mod.TimesFMGenerator
        print("  [OK] 导入成功")
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return

    # 3. 生成预测
    print("\n[3] TimesFM 预测 (horizon=24h, context=256)...")
    gen = TimesFMGenerator(
        symbol="ETH/USDT",
        horizon=24,
        context_len=256,
        buy_threshold=0.01,
        sell_threshold=-0.01,
    )

    try:
        signals = gen.generate(df)
    except Exception as e:
        print(f"  ✗ 预测失败: {e}")
        return

    print(f"  信号数量: {len(signals)}")

    if not signals:
        print("  (当前无信号，预测收益在阈值内)")
    else:
        for sig in signals:
            print(f"\n  ✓ 信号: {sig.type.value.upper()}")
            print(f"    价格: {sig.price:.2f}")
            print(f"    强度: {sig.strength:.4f}")
            print(f"    原因: {sig.reason}")
            print(f"    预测收益: {sig.metadata.get('pred_return', 'N/A'):.4f}")
            if sig.metadata.get('quantile_forecast'):
                q = sig.metadata['quantile_forecast']
                print(f"    分位数预测 (P50序列): {q[:5]}...")

    # 4. 获取详细预测
    print("\n[4] 获取详细预测数据...")
    try:
        pred_return, point_fc, quantile_fc = gen.get_forecast(df)
        print(f"  预测收益 (24h): {pred_return*100:.4f}%")
        print(f"  Point forecast shape: {point_fc.shape}")
        print(f"  Quantile forecast shape: {quantile_fc.shape}")
    except Exception as e:
        print(f"  详细预测失败: {e}")

    # 5. 对比：传统技术指标
    print("\n[5] 对比: 传统技术指标信号...")
    try:
        # 同样绕过主包导入
        gens_path = os.path.join(_project_root, "signal", "generators.py")
        if not os.path.exists(gens_path):
            gens_path = os.path.join(r"D:\量化交易系统\量化之神", "quant_trading", "signal", "generators.py")
        gens_spec = importlib.util.spec_from_file_location("gens", gens_path)
        gens_mod = importlib.util.module_from_spec(gens_spec)
        gens_mod.SignalType = MockSignalType
        gens_mod.SignalDirection = MockSignalDirection
        gens_mod.Signal = object
        gens_spec.loader.exec_module(gens_mod)
        RSIGenerator = gens_mod.RSIGenerator
        MACDGenerator = gens_mod.MACDGenerator

        rsi_gen = RSIGenerator(period=14, symbol="ETH/USDT")
        macd_gen = MACDGenerator(symbol="ETH/USDT")
        rsi_signals = rsi_gen.generate(df)
        macd_signals = macd_gen.generate(df)
        print(f"  RSI: {len(rsi_signals)} signals, last: {rsi_signals[-1].reason if rsi_signals else 'none'}")
        print(f"  MACD: {len(macd_signals)} signals, last: {macd_signals[-1].reason if macd_signals else 'none'}")
    except Exception as e:
        print(f"  Traditional indicators failed: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_timesfm_on_eth()
