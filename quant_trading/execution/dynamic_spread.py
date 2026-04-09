"""
Dynamic Spread Pricing — 动态价差做市算法
==========================================

Absorbed from hummingbot's pmm_dynamic controller.

核心算法：
1. NATR (Normalized ATR) 计算归一化波动率
2. MACD 信号标准化 + histogram 方向判断趋势
3. 动态参考价 = close * (1 + price_multiplier)
4. 价差根据波动率自适应调整

适用于：
- 加密货币做市商
- 波动率自适应价差策略
- 趋势感知型挂单

Usage:
    from quant_trading.execution import DynamicSpreadPricing

    dsp = DynamicSpreadPricing(
        macd_fast=21,
        macd_slow=42,
        macd_signal=9,
        natr_length=14
    )
    reference_price, spread_multiplier = dsp.calculate(high, low, close)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from quant_trading.indicators.mytt import ATR, MACD


__all__ = ["DynamicSpreadPricing", "DynamicSpreadConfig"]


@dataclass
class DynamicSpreadConfig:
    """动态价差配置 / Dynamic spread configuration."""
    macd_fast: int = 21
    macd_slow: int = 42
    macd_signal: int = 9
    natr_length: int = 14
    # 波动率对价差的影响系数 (volatility to spread multiplier)
    volatility_spread_coef: float = 1.0
    # 趋势信号对价格偏移的影响系数
    trend_price_coef: float = 0.5


class DynamicSpreadPricing:
    """
    动态价差做市算法 / Dynamic Spread Market Making Algorithm.

    使用 MACD 趋势信号和 NATR 波动率动态调整：
    - 参考价偏移 (trend-aware mid-price)
    - 挂单价差 (volatility-adjusted spread)

    Algorithm:
        NATR = ATR(high, low, close, length) / close * 100
        MACD_hist = MACD(close).histogram
        macd_signal = -(DIF - mean(DIF)) / std(DIF)
        macdh_direction = sign(MACD_hist)

        max_price_shift = NATR / 2
        price_multiplier = (trend_coef * macd_signal + trend_coef * macdh_direction) * max_price_shift
        reference_price = close * (1 + price_multiplier)

        spread_multiplier = NATR * volatility_spread_coef
    """

    def __init__(self, config: DynamicSpreadConfig | None = None):
        self.config = config or DynamicSpreadConfig()
        self._last_macd: Tuple[float, float, float] | None = None

    def calculate_natr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        计算 NATR (Normalized ATR) — 归一化波动率指标.

        NATR = ATR(high, low, close, length) / close * 100

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组

        Returns:
            NATR 数组 (百分比形式, e.g., 2.5 表示 2.5% 波动率)
        """
        atr_values = ATR(close, high, low, self.config.natr_length)
        # 避免除零
        close_safe = np.where(close == 0, np.nan, close)
        natr = atr_values / close_safe * 100
        # 填充零值和NaN
        natr = np.nan_to_num(natr, nan=0.0, posinf=0.0, neginf=0.0)
        return natr

    def calculate_macd_signal(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算标准化 MACD 信号和 histogram 方向.

        Args:
            close: 收盘价数组

        Returns:
            (macd_signal_norm, macdh_direction) — 标准化 MACD 信号和方向数组
        """
        dif, dea, macd_hist = MACD(
            close,
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal
        )

        # 标准化 DIF (类似 z-score)
        macd_mean = np.mean(dif)
        macd_std = np.std(dif)
        if macd_std > 1e-10:
            macd_signal_norm = -(dif - macd_mean) / macd_std
        else:
            macd_signal_norm = np.zeros_like(dif)

        # MACD histogram 方向: >0 为 1, <=0 为 -1
        macdh_direction = np.where(macd_hist > 0, 1.0, -1.0)

        return macd_signal_norm, macdh_direction

    def calculate(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        计算动态参考价和价差乘数.

        Args:
            high: 最高价数组 (numpy array)
            low: 最低价数组 (numpy array)
            close: 收盘价数组 (numpy array)

        Returns:
            (reference_price, spread_multiplier, natr) — 参考价、价差乘数、NATR
        """
        if len(close) < max(self.config.macd_slow, self.config.natr_length) + 10:
            raise ValueError(
                f"数据长度不足，需要至少 {max(self.config.macd_slow, self.config.natr_length) + 10} 根K线"
            )

        # 1. 计算 NATR 归一化波动率
        natr = self.calculate_natr(high, low, close)

        # 2. 计算 MACD 信号
        macd_signal_norm, macdh_direction = self.calculate_macd_signal(close)

        # 3. 获取最新值
        natr_val = natr[-1]
        macd_sig = macd_signal_norm[-1]
        macdh_dir = macdh_direction[-1]
        close_val = close[-1]

        # 4. 计算最大价格偏移
        max_price_shift = natr_val / 2 / 100  # 转换为小数

        # 5. 计算价格乘数
        price_multiplier = (
            self.config.trend_price_coef * macd_sig +
            self.config.trend_price_coef * macdh_dir
        ) * max_price_shift

        # 6. 参考价 = close * (1 + price_multiplier)
        reference_price = close_val * (1 + price_multiplier)

        # 7. 价差乘数 = NATR * 系数
        spread_multiplier = natr_val * self.config.volatility_spread_coef / 100

        self._last_macd = (macd_sig, macdh_dir, natr_val)

        return reference_price, spread_multiplier, natr_val

    def calculate_levels(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        level_spreads: list[float],
        side: str = "both"
    ) -> dict:
        """
        计算多层级挂单价格.

        Args:
            high, low, close: 价格数据
            level_spreads: 各层级价差列表 (e.g., [0.001, 0.002, 0.004])
            side: 'buy', 'sell', 或 'both'

        Returns:
            dict with 'reference_price', 'buy_prices', 'sell_prices'
        """
        reference_price, spread_mult, _ = self.calculate(high, low, close)

        buy_prices = []
        sell_prices = []

        for spread_pct in level_spreads:
            # 调整后的价差 = 基础价差 * (1 + 波动率调整)
            adjusted_spread = spread_pct * (1 + spread_mult)
            buy_prices.append(reference_price * (1 - adjusted_spread))
            sell_prices.append(reference_price * (1 + adjusted_spread))

        result = {
            "reference_price": reference_price,
            "spread_multiplier": spread_mult,
        }
        if side in ("buy", "both"):
            result["buy_prices"] = buy_prices
        if side in ("sell", "both"):
            result["sell_prices"] = sell_prices

        return result

    def get_last_macd_state(self) -> Tuple[float, float, float] | None:
        """返回最新的 MACD 状态 (signal_norm, histogram_direction, natr)."""
        return self._last_macd
