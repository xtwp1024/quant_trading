"""微观结构流动性真空吃反弹策略 (Microstructure Rebound)"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class MicrostructureParams(StrategyParams):
    """微观结构错杀策略参数"""
    window_size: int = 15  # 放宽取样窗口
    drop_threshold: float = 0.012  # 将暴跌门槛从 5% 降至 1.2%，极大提升灵敏度
    
    # 移动止损 (Trailing Stop) 参数
    trailing_activation_pct: float = 0.015  # 盈利达到 1.5% 后激活追踪止损
    trailing_callback_pct: float = 0.005    # 激活后，从最高点回撤 0.5% 即平仓
    
    # 硬止损 (兜底)
    hard_stop_loss_pct: float = 0.03
    
    # ATR 自适应参数
    atr_period: int = 14
    position_size: float = 0.1  # 基础仓位


class MicrostructureReboundStrategy(BaseStrategy):
    """包含极速移动止损与 ATR 波动率自适应的微观吃反弹策略"""
    
    name = "microstructure_rebound_adaptive"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[MicrostructureParams] = None,
    ) -> None:
        super().__init__(symbol, params or MicrostructureParams())
        self.params: MicrostructureParams = self.params
        
        # trade_states 记录每个币种的持仓风控状态
        # 格式: { symbol: {"entry_price": float, "highest_price": float, "trailing_active": bool} }
        self.trade_states: Dict[str, Dict[str, Any]] = {}

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """计算真实波动率 (ATR)"""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift(1)).abs()
        low_close = (data['low'] - data['close'].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.params.atr_period).mean()
        return atr

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """兼容 BaseStrategy 接口，多币种请走 generate_signals_for_symbol"""
        return self.generate_signals_for_symbol(self.symbol, data)
    
    def generate_signals_for_symbol(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        signals = []
        if len(data) < max(self.params.window_size, self.params.atr_period) + 1:
            return signals
            
        # 附加 ATR
        atr_series = self._calculate_atr(data)
        
        for i in range(max(self.params.window_size, self.params.atr_period), len(data)):
            current = data.iloc[i]
            current_price = float(current["close"])
            current_atr = float(atr_series.iloc[i])
            # ATR 百分比波幅 (当前 ATR 占币价的比例)
            atr_pct = current_atr / current_price if current_price > 0 else 0
            
            # --- 1. 风控：持仓状态机检查 (止损 / 移动止盈) ---
            if symbol in self.trade_states:
                state = self.trade_states[symbol]
                entry_price = state["entry_price"]
                highest_price = state["highest_price"]
                
                # 更新最高价
                if current_price > highest_price:
                    state["highest_price"] = current_price
                    highest_price = current_price
                
                pnl_pct = (current_price - entry_price) / entry_price
                
                # 动态自适应：如果当前波动率极大，适当放宽回撤容忍度，防插针
                adapted_callback = self.params.trailing_callback_pct
                if atr_pct > 0.01: # 分钟级别 ATR > 1%，说明处于疯牛暴力波动
                    adapted_callback = self.params.trailing_callback_pct * 1.5

                # 检查是否激活 Trailing Stop
                if not state["trailing_active"] and pnl_pct >= self.params.trailing_activation_pct:
                    state["trailing_active"] = True
                    # logger.info(f"[{symbol}] 盈利穿透 {self.params.trailing_activation_pct*100}%，已激活移动止损机制！")
                    
                # 触发 Trailing Stop 出局
                if state["trailing_active"]:
                    drawdown_from_high = (highest_price - current_price) / highest_price
                    if drawdown_from_high >= adapted_callback:
                        signals.append(
                            Signal(
                                type=SignalType.EXIT_LONG,
                                symbol=symbol,
                                timestamp=int(current["timestamp"]),
                                price=current_price,
                                reason=f"Trailing Stop Hit (Drawdown: {drawdown_from_high*100:.2f}%, PnL: {pnl_pct*100:.2f}%)",
                            )
                        )
                        del self.trade_states[symbol]
                        continue
                        
                # 触发硬止损 (兜底保护)
                elif pnl_pct <= -self.params.hard_stop_loss_pct:
                    signals.append(
                        Signal(
                            type=SignalType.EXIT_LONG,
                            symbol=symbol,
                            timestamp=int(current["timestamp"]),
                            price=current_price,
                            reason=f"Hard Stop Loss Hit ({pnl_pct*100:.2f}%)",
                        )
                    )
                    del self.trade_states[symbol]
                    continue
            
            # --- 2. 进攻：短线暴跌结构扫描 ---
            if symbol not in self.trade_states:
                window_start = data.iloc[i - self.params.window_size]
                start_price = float(window_start["close"])
                
                drop_pct = (current_price - start_price) / start_price
                
                # 自适应跌幅阈值：如果市场死水一潭(ATR低)，稍微跌一点(4%)就算断层；
                # 如果市场处于狂暴状态(ATR高)，正常的波动可能就3%，必须要跌幅 > 6% 才算真实错杀。
                adapted_drop_threshold = self.params.drop_threshold
                if atr_pct > 0.015:
                    adapted_drop_threshold = self.params.drop_threshold * 1.5 # 拉高门槛
                elif atr_pct < 0.005:
                    adapted_drop_threshold = self.params.drop_threshold * 0.8 # 降低门槛

                if drop_pct <= -adapted_drop_threshold:
                    # 发生微观流动性断裂 -> 买入抢反弹
                    signals.append(
                        Signal(
                            type=SignalType.BUY,
                            symbol=symbol,
                            timestamp=int(current["timestamp"]),
                            price=current_price,
                            reason=f"Adaptive Flash Crash: {drop_pct*100:.2f}% drop (Thresh: {adapted_drop_threshold*100:.2f}%)",
                        )
                    )
                    # 记录入场状态
                    self.trade_states[symbol] = {
                        "entry_price": current_price,
                        "highest_price": current_price,
                        "trailing_active": False
                    }
                    
        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        if signal.type == SignalType.BUY:
            position_value = context.portfolio_value * self.params.position_size
            amount = position_value / signal.price
            return max(amount, 0.0)
        else:
            return 0.0
