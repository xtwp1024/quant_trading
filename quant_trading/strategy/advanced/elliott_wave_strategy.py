"""
Elliott Wave Strategy — 艾略特波浪策略适配器
==========================================

基于 `quant_trading.strategy.elliott_wave.ElliottWaveAnalyzer` 的 BaseStrategy 适配器。

艾略特波浪理论核心概念：
- 推动浪(Impulse): 1, 2, 3, 4, 5
- 调整浪(Correction): A, B, C
- 斐波那契比率分析

规则：
1. 浪2不能回撤浪1的100%以上
2. 浪3不能是最短的推动浪
3. 浪4不能与浪1重叠
4. 交替原则：浪2和浪4的形态交替

Classes
-------
ElliottWaveStrategy
    BaseStrategy adapter for Elliott Wave trading.
ElliottWaveParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal, SignalDirection, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.elliott_wave import (
    ElliottWaveAnalyzer,
    WaveDegree,
    WaveType,
)


@dataclass
class ElliottWaveParams(StrategyParams):
    """艾略特波浪策略参数"""
    min_wave_size: int = 5
    """波浪最小K线数"""
    strict_mode: bool = True
    """严格模式，强制执行艾略特规则"""
    entry_threshold: float = 0.6
    """信号强度阈值"""
    prefer_wave_3: bool = True
    """优先交易浪3（最强劲的推动浪）"""


class ElliottWaveStrategy(BaseStrategy):
    """
    艾略特波浪策略 (Elliott Wave Strategy)

    基于艾略特波浪理论的分析和交易策略。
    识别推动浪和调整浪，在特定波浪位置产生信号。

    信号逻辑：
    - 浪3突破浪1高点 -> 买入（强劲趋势）
    - 浪4低点反弹 -> 买入（回调入场）
    - 浪5突破浪3高点 -> 买入（趋势延续）

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : ElliottWaveParams, optional
        策略参数
    """

    name = "elliott_wave"

    def __init__(
        self,
        symbol: str,
        params: Optional[ElliottWaveParams] = None,
    ) -> None:
        super().__init__(symbol, params or ElliottWaveParams())
        self._analyzer = ElliottWaveAnalyzer(
            min_wave_size=self.params.min_wave_size,
            strict_mode=self.params.strict_mode,
        )
        self._signals: List[Signal] = []

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        基于艾略特波浪生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            K线数据，需包含 open, high, low, close, timestamp 列

        Returns
        -------
        List[Signal]
            交易信号列表
        """
        # 加载数据到分析器
        self._analyzer.load_data(data)

        # 执行完整分析
        self._analyzer.analyze()

        # 生成信号
        raw_signals = self._analyzer.generate_signals()

        signals = []
        for sig in raw_signals:
            # 过滤低强度信号
            if sig.strength < self.params.entry_threshold:
                continue

            # 波浪3信号优先
            wave_num = sig.metadata.get('wave', 0)
            if self.params.prefer_wave_3 and wave_num != 3:
                continue

            # 艾略特波浪信号方向由原始信号决定
            # 原始信号中 direction 字段是 SignalDirection 类型
            sig_type = SignalType.BUY  # 艾略特波浪的买入信号

            # 获取当前最新K线信息
            last_row = data.iloc[-1] if len(data) > 0 else None
            timestamp = int(last_row.get("timestamp", 0)) if last_row is not None else 0
            price = float(last_row.get("close", sig.price or 0.0)) if last_row is not None else 0.0

            signals.append(
                Signal(
                    type=sig_type,
                    symbol=self.symbol,
                    timestamp=timestamp,
                    price=price,
                    strength=sig.strength,
                    metadata={
                        "strategy": "elliott_wave",
                        "type": sig.metadata.get("type", "elliott_wave"),
                        "wave": wave_num,
                        "reason": sig.metadata.get("reason", ""),
                    },
                )
            )

        self._signals = signals
        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        计算仓位大小。

        Parameters
        ----------
        signal : Signal
            交易信号
        context : StrategyContext
            策略上下文

        Returns
        -------
        float
            仓位大小
        """
        # 基于信号强度和波浪位置计算仓位
        # 浪3信号更强，给予更大仓位
        wave_num = signal.metadata.get("wave", 0)
        wave_multiplier = 1.5 if wave_num == 3 else 1.0

        base_size = context.available_cash * signal.strength * 0.1
        return base_size * wave_multiplier

    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """
        处理单根K线数据。

        Parameters
        ----------
        bar : pd.Series
            K线数据

        Returns
        -------
        Optional[Signal]
            交易信号（如果有）
        """
        df = pd.DataFrame([bar])
        signals = self.generate_signals(df)
        return signals[-1] if signals else None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        处理tick数据。

        Parameters
        ----------
        tick : Dict[str, Any]
            Tick数据

        Returns
        -------
        Optional[Signal]
            交易信号（如果有）
        """
        return None

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        pass

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        # 艾略特波浪需要足够的数据来识别多个波浪
        return max(self.params.min_wave_size * 10, 200)

    def get_analyzer(self) -> ElliottWaveAnalyzer:
        """获取底层分析器（用于诊断）"""
        return self._analyzer

    def get_summary(self) -> Dict[str, Any]:
        """获取波浪分析摘要"""
        return self._analyzer.get_summary()

    def get_current_wave_position(self) -> Optional[Dict]:
        """获取当前波浪位置"""
        return self._analyzer.get_current_wave_position()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "min_wave_size": self.params.min_wave_size,
                "strict_mode": self.params.strict_mode,
                "entry_threshold": self.params.entry_threshold,
                "prefer_wave_3": self.params.prefer_wave_3,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElliottWaveStrategy":
        """从字典反序列化"""
        params = ElliottWaveParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
