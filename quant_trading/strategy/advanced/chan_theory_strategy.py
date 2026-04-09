"""
Chan Theory Strategy — 缠论策略适配器
======================================

基于 `quant_trading.strategy.chan_theory.ChanTheoryAnalyzer` 的 BaseStrategy 适配器。

缠论核心概念：
- 笔 (Bi): 相邻顶底分型，中间不少于5根K线
- 线段 (XSegment): 连续三笔构成的一段走势
- 中枢 (ZS): 至少三个线段的重叠区域
- 走势类型: 上涨、下跌、盘整
- 背驰 (BeiChi): 趋势力度减弱

Classes
-------
ChanTheoryStrategy
    BaseStrategy adapter for Chan Theory trading.
ChanTheoryParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal, SignalDirection, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.chan_theory import (
    ChanTheoryAnalyzer,
    KLineDirection,
)


@dataclass
class ChanTheoryParams(StrategyParams):
    """缠论策略参数"""
    min_bi_klines: int = 5
    """笔的最少K线数（默认5）"""
    segment_min_bi: int = 3
    """线段的最少笔数（默认3）"""
    require_new_high: bool = True
    """构建笔时是否需要创新高/新低"""
    beichi_method: str = 'price'
    """背驰判断方法: 'price', 'volume', 'macd'"""
    entry_threshold: float = 0.7
    """信号强度阈值（低于此值不入场）"""


class ChanTheoryStrategy(BaseStrategy):
    """
    缠论策略 (Chan Theory Strategy)

    基于缠论的技术分析策略，识别笔、线段、中枢和背驰，
    在趋势转折点产生交易信号。

    信号逻辑：
    - 底背驰后向上突破中枢上轨 -> 买入
    - 顶背驰后向下突破中枢下轨 -> 卖出

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : ChanTheoryParams, optional
        策略参数
    """

    name = "chan_theory"

    def __init__(
        self,
        symbol: str,
        params: Optional[ChanTheoryParams] = None,
    ) -> None:
        super().__init__(symbol, params or ChanTheoryParams())
        self._analyzer = ChanTheoryAnalyzer(
            min_bi_klines=self.params.min_bi_klines,
            segment_min_bi=self.params.segment_min_bi,
            require_new_high=self.params.require_new_high,
        )
        self._signals: List[Signal] = []
        self._last_direction: Optional[KLineDirection] = None

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        基于缠论生成交易信号。

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
            if sig.strength < self.params.entry_threshold:
                continue

            # 转换方向：SignalDirection -> SignalType
            sig_type = (
                SignalType.BUY if sig.direction == SignalDirection.LONG
                else SignalType.SELL
            )

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
                        "strategy": "chan_theory",
                        "type": sig.metadata.get("type", "chan_entry"),
                        "reason": sig.metadata.get("reason", ""),
                        "beichi": sig.metadata.get("beichi", ""),
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

        使用上下文中的风险参数决定仓位。

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
        # 基于信号强度和可用资金计算仓位
        base_size = context.available_cash * signal.strength * 0.1
        return base_size

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
        # 缠论策略基于K线，不处理tick
        return None

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        pass

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        # 缠论需要足够的数据来构建笔和线段
        return max(self.params.min_bi_klines * 10, 200)

    def get_analyzer(self) -> ChanTheoryAnalyzer:
        """获取底层分析器（用于诊断）"""
        return self._analyzer

    def get_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return self._analyzer.get_summary()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "min_bi_klines": self.params.min_bi_klines,
                "segment_min_bi": self.params.segment_min_bi,
                "require_new_high": self.params.require_new_high,
                "beichi_method": self.params.beichi_method,
                "entry_threshold": self.params.entry_threshold,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChanTheoryStrategy":
        """从字典反序列化"""
        params = ChanTheoryParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
