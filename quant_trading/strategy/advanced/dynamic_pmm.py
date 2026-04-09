"""
动态PMM做市策略
==============

基于 hummingbot pmm_dynamic 核心逻辑:
- MACD 动态价格偏移: 使用 MACD 信号标准化后偏移参考价格
- NATR 动态价差: 价差与市场波动率挂钩

参考价格公式:
    reference_price = close * (1 + price_multiplier)
    price_multiplier = ((0.5 * macd_signal + 0.5 * macdh_signal) * max_price_shift)
    max_price_shift = natr / 2
    spread_multiplier = natr

MACD 信号:
    macd_signal = - (macd - macd.mean()) / macd.std()  # 标准化
    macdh_signal = 1 if macdh > 0 else -1  # MACD直方图方向

使用方式:
    from quant_trading.strategy.advanced.dynamic_pmm import DynamicPMMStrategy, DynamicPMMParams

    strategy = DynamicPMMStrategy("BTC-USDT", DynamicPMMParams())
    signals = strategy.generate_signals(candles_df)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class DynamicPMMParams(StrategyParams):
    """动态PMM策略参数"""
    # MACD 参数
    macd_fast: int = 21
    macd_slow: int = 42
    macd_signal: int = 9
    # NATR 参数
    natr_length: int = 14
    # 多层报价参数
    buy_spreads: List[float] = None  # e.g. [0.001, 0.002, 0.004]
    sell_spreads: List[float] = None
    # 仓位参数
    position_size_pct: float = 0.1  # 每次下单占账户的比例

    def __post_init__(self):
        if self.buy_spreads is None:
            self.buy_spreads = [0.001, 0.002, 0.004]
        if self.sell_spreads is None:
            self.sell_spreads = [0.001, 0.002, 0.004]


class DynamicPMMStrategy(BaseStrategy):
    """
    动态PMM做市策略

    使用 MACD 信号动态调整参考价格，NATR 动态调整价差。
    核心思想：当市场上涨趋势强时，提高买入价格；当市场下跌趋势强时，降低卖出价格。
    """

    name = "dynamic_pmm"

    def __init__(
        self,
        symbol: str,
        params: Optional[DynamicPMMParams] = None,
    ) -> None:
        super().__init__(symbol, params or DynamicPMMParams())
        self._max_records = max(
            self.params.macd_slow,
            self.params.macd_fast,
            self.params.macd_signal,
            self.params.natr_length
        ) + 100

        self._last_signal: Optional[Signal] = None
        self._reference_price: float = 0.0
        self._spread_multiplier: float = 0.0

    def _compute_macd(
        self,
        close: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算 MACD 指标

        Returns:
            (macd, macd_signal, macdh) 元组
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macdh = macd - macd_signal

        return macd, macd_signal, macdh

    def _compute_natr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int
    ) -> pd.Series:
        """
        计算 Normalized Average True Range (NATR)

        NATR = ATR / Close * 100
        用于衡量市场波动性并动态调整价差
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=length).mean()
        natr = (atr / close) * 100

        return natr

    def _calculate_dynamic_price(
        self,
        candles: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        计算动态参考价格和价差倍数

        Returns:
            (reference_price, spread_multiplier)
        """
        close = candles["close"]
        high = candles["high"]
        low = candles["low"]

        # 计算 MACD
        macd, macd_signal_line, macdh = self._compute_macd(
            close,
            self.params.macd_fast,
            self.params.macd_slow,
            self.params.macd_signal
        )

        # 标准化 MACD 信号
        macd_std = macd.std()
        if macd_std > 0:
            macd_signal_normalized = -(macd - macd.mean()) / macd_std
        else:
            macd_signal_normalized = 0

        # MACD 直方图方向信号
        macdh_signal = macdh.apply(lambda x: 1 if x > 0 else -1)

        # 计算 NATR 作为最大价格偏移
        natr = self._compute_natr(
            high, low, close, self.params.natr_length
        )
        max_price_shift = natr / 2

        # 计算价格偏移量
        last_macd_signal = macd_signal_normalized.iloc[-1]
        last_macdh_signal = macdh_signal.iloc[-1]
        last_nat = natr.iloc[-1] / 100  # 转换为小数

        price_multiplier = (0.5 * last_macd_signal + 0.5 * last_macdh_signal) * (last_nat / 2)

        # 参考价格
        reference_price = close.iloc[-1] * (1 + price_multiplier)

        # 价差倍数
        spread_multiplier = last_nat

        return float(reference_price), float(spread_multiplier)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成做市信号

        Args:
            data: K线数据，必须包含 high, low, close 列

        Returns:
            Signal 列表，包含参考价格和价差信息
        """
        signals = []

        if len(data) < self._max_records:
            return signals

        df = data.copy()

        # 计算动态价格
        reference_price, spread_multiplier = self._calculate_dynamic_price(df)

        self._reference_price = reference_price
        self._spread_multiplier = spread_multiplier

        # 获取当前价格方向
        close = df["close"]
        close_change = close.iloc[-1] - close.iloc[-2]
        price_direction = SignalType.NEUTRAL
        if close_change > 0:
            price_direction = SignalType.BULLISH
        elif close_change < 0:
            price_direction = SignalType.BEARISH

        # 创建信号
        signal = Signal(
            symbol=self.symbol,
            signal_type=SignalType.MARKET_MAKING,
            strength=abs(close_change / close.iloc[-1]) if close.iloc[-1] > 0 else 0,
            timestamp=df.index[-1],
            metadata={
                "reference_price": reference_price,
                "spread_multiplier": spread_multiplier,
                "buy_spreads": [s * spread_multiplier for s in self.params.buy_spreads],
                "sell_spreads": [s * spread_multiplier for s in self.params.sell_spreads],
                "price_direction": price_direction,
                "natr": spread_multiplier * 100,  # 存储为百分比形式
            }
        )

        signals.append(signal)
        self._last_signal = signal

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        计算仓位大小

        基于组合配置比例和风险参数计算仓位
        """
        if signal.strength <= 0:
            return 0.0

        # 基础仓位百分比
        base_size = self.params.position_size_pct

        # 根据波动率调整
        natr = signal.metadata.get("natr", 10)
        volatility_adjustment = min(2.0, max(0.5, 10 / natr)) if natr > 0 else 1.0

        # 根据价格方向调整
        direction = signal.metadata.get("price_direction", SignalType.NEUTRAL)
        if direction == SignalType.BULLISH:
            # 上涨趋势时多买少卖
            buy_bias = 1.2
            sell_bias = 0.8
        elif direction == SignalType.BEARISH:
            # 下跌趋势时少买多卖
            buy_bias = 0.8
            sell_bias = 1.2
        else:
            buy_bias = sell_bias = 1.0

        return base_size * volatility_adjustment * buy_bias

    def get_required_history(self) -> int:
        """获取所需历史数据长度"""
        return self._max_records

    def get_bid_ask_prices(
        self,
        reference_price: Optional[float] = None
    ) -> Tuple[List[float], List[float]]:
        """
        获取多层买卖报价

        Returns:
            (bid_prices, ask_prices)
        """
        if reference_price is None:
            reference_price = self._reference_price

        if reference_price <= 0:
            return [], []

        bid_prices = []
        ask_prices = []

        for spread in self.params.buy_spreads:
            actual_spread = spread * self._spread_multiplier
            bid_prices.append(reference_price * (1 - actual_spread))

        for spread in self.params.sell_spreads:
            actual_spread = spread * self._spread_multiplier
            ask_prices.append(reference_price * (1 + actual_spread))

        return bid_prices, ask_prices

    def get_quoting_levels(self) -> dict:
        """
        获取当前报价层级信息

        Returns:
            dict: 包含买单和卖单层级的详细信息
        """
        bid_prices, ask_prices = self.get_bid_ask_prices()

        levels = {
            "reference_price": self._reference_price,
            "spread_multiplier": self._spread_multiplier,
            "bids": [],
            "asks": []
        }

        for i, (bid, ask) in enumerate(zip(bid_prices, ask_prices)):
            level_spread = self.params.buy_spreads[i] * self._spread_multiplier
            levels["bids"].append({
                "level": i + 1,
                "price": bid,
                "spread_pct": level_spread * 100,
                "size_pct": self.params.position_size_pct
            })
            level_spread = self.params.sell_spreads[i] * self._spread_multiplier
            levels["asks"].append({
                "level": i + 1,
                "price": ask,
                "spread_pct": level_spread * 100,
                "size_pct": self.params.position_size_pct
            })

        return levels

    @classmethod
    def from_dict(cls, data: dict) -> "DynamicPMMStrategy":
        """从字典创建策略"""
        from quant_trading.strategy.advanced import DynamicPMMParams

        params = DynamicPMMParams(**data.get("params", {}))
        return cls(
            symbol=data.get("symbol", "UNKNOWN"),
            params=params
        )


# ---------------------------------------------------------------------------
# 辅助函数：直接计算PMM价格（不依赖策略类）
# ---------------------------------------------------------------------------

def compute_pmm_prices(
    candles: pd.DataFrame,
    macd_fast: int = 21,
    macd_slow: int = 42,
    macd_signal: int = 9,
    natr_length: int = 14,
) -> Tuple[float, float]:
    """
    直接计算PMM参考价格和价差倍数

    Args:
        candles: K线数据
        macd_fast: MACD快线周期
        macd_slow: MACD慢线周期
        macd_signal: MACD信号线周期
        natr_length: NATR周期

    Returns:
        (reference_price, spread_multiplier)

    Example:
        >>> ref_price, spread_mult = compute_pmm_prices(df)
        >>> print(f"参考价格: {ref_price}, 价差倍数: {spread_mult}")
    """
    strategy = DynamicPMMStrategy("TEMP", DynamicPMMParams(
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        natr_length=natr_length,
    ))
    return strategy._calculate_dynamic_price(candles)


# ---------------------------------------------------------------------------
# Main / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # 模拟价格走势
    returns = np.random.randn(n) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    high = price * (1 + np.random.rand(n) * 0.01)
    low = price * (1 - np.random.rand(n) * 0.01)
    close = price

    df = pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
    }, index=dates)

    # 测试策略
    print("测试 DynamicPMMStrategy...")
    strategy = DynamicPMMStrategy("TEST-USDT", DynamicPMMParams())

    signals = strategy.generate_signals(df)
    if signals:
        s = signals[-1]
        print(f"  参考价格: {s.metadata['reference_price']:.4f}")
        print(f"  价差倍数: {s.metadata['spread_multiplier']:.4f}")
        print(f"  买单层级: {s.metadata['buy_spreads']}")
        print(f"  卖单层级: {s.metadata['sell_spreads']}")

    # 测试报价生成
    bid_prices, ask_prices = strategy.get_bid_ask_prices()
    print(f"  买入报价: {bid_prices}")
    print(f"  卖出报价: {ask_prices}")

    # 测试层级信息
    levels = strategy.get_quoting_levels()
    print(f"  报价层级数: {len(levels['bids'])} 买单, {len(levels['asks'])} 卖单")

    # 测试独立函数
    print("\n测试 compute_pmm_prices 函数...")
    ref, spread = compute_pmm_prices(df)
    print(f"  参考价格: {ref:.4f}, 价差倍数: {spread:.4f}")
