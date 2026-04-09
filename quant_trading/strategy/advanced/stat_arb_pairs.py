"""跨资产统计套利：配对交易与 Z-Score 均值回归策略"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams


@dataclass
class StatArbParams(StrategyParams):
    """配对统计套利参数"""
    window_limit: int = 120  # 计算 Z-Score 的历史滑动窗口（如 120 分钟）
    z_score_long_entry: float = -2.0  # Z-score 低于此值，做多 Target (预期 Target 补涨或 Base 补跌)
    z_score_short_entry: float = 2.0  # Z-score 高于此值，做空 Target (预期 Target 回调或 Base 赶上)
    z_score_exit: float = 0.0         # Z-score 回归到 0 附近时平仓
    
    # 止损控制
    hard_stop_loss_pct: float = 0.05
    # 追踪止损机制复用
    trailing_activation_pct: float = 0.015
    trailing_callback_pct: float = 0.005
    position_size: float = 0.1


class StatArbPairsStrategy(BaseStrategy):
    """基于双资产比价协整性的均值回归吃大盘 Beta 策略
    
    使用场景：
    我们设定一个 Base 资产（如大盘 ETH），然后监控 Target 资产（如 OP）。
    由于两者处于强关联，它们的比价 Ratio = Target_Price / Base_Price 在常态下应该围绕一个均值波动。
    当 Ratio 剧烈下跌（例如 ETH 暴涨而 OP 没涨），导致 Z-Score < -2.0 时，买入做多 OP。
    """
    
    name = "stat_arb_pairs"
    
    def __init__(
        self,
        symbol: str, # 在这个策略里，symbol 通常是 'DUAL_CORE' 或统揽全局的名称
        base_symbol: str = "ETH/USDT:USDT",
        params: Optional[StatArbParams] = None,
    ) -> None:
        super().__init__(symbol, params or StatArbParams())
        self.params: StatArbParams = self.params
        self.base_symbol = base_symbol
        
        # trade_states 记录各个山寨币的做单状态
        # { "OP/USDT:USDT": { "entry_price": 2.5, "highest_price": 2.6, "side": "long", "trailing_active": False } }
        self.trade_states: Dict[str, Dict[str, Any]] = {}
        
        # 我们需要在策略内部维持目标币种对其 Base 的历史 Ratio 序列，用于计算 Z-Score
        # 结构: { "OP/USDT:USDT": [0.0012, 0.0013, ...] }
        self.ratio_history: Dict[str, List[float]] = {}


    def generate_signals_with_global_bus(
        self, 
        target_symbol: str, 
        target_df: pd.DataFrame, 
        global_latest_prices: Dict[str, float]
    ) -> List[Signal]:
        """
        依赖全局行情总线的信号生成器。
        每次 Target 资产更新时调用此方法。
        
        target_symbol: 触发计算的山寨币（如 OP/USDT:USDT）
        target_df: 该山寨币的历史 DataFrame
        global_latest_prices: 内存中缓存的所有币种的最新秒级价格，用于获取 Base 的价格
        """
        signals = []
        
        # 1. 预检
        if target_symbol == self.base_symbol:
            # 基准自身不需要跟自己算配对
            return signals
            
        if self.base_symbol not in global_latest_prices:
            # 如果我们连基准的大盘价格都没建立，拒绝运行
            return signals
            
        base_current_price = global_latest_prices[self.base_symbol]
        target_current_price = target_df.iloc[-1]['close']
        current_ts = int(target_df.iloc[-1]['timestamp'])
        
        # 2. 计算当前的比值 Ratio
        current_ratio = target_current_price / base_current_price
        
        # 3. 维护 Ratio 历史数组，用于计算均值(mu)和标准差(sigma)
        if target_symbol not in self.ratio_history:
            self.ratio_history[target_symbol] = []
            
        history = self.ratio_history[target_symbol]
        history.append(current_ratio)
        
        # 保证数组不无限发涨
        if len(history) > self.params.window_limit:
            history.pop(0)
            
        # 如果样本不够（比如前几十根K线），无法计算可靠的标准差，先跳过
        if len(history) < min(30, self.params.window_limit):
            return signals
            
        # 4. 计算 Z-Score
        ratio_array = np.array(history)
        mu = np.mean(ratio_array)
        sigma = np.std(ratio_array)
        
        if sigma == 0:
            return signals # 防止除以 0
            
        z_score = (current_ratio - mu) / sigma
        
        # 打印一下调试信息？可以在外面引擎层打印，策略层保持纯净
        # logger.info(f"[{target_symbol}] Ratio: {current_ratio:.6f}, Z-Score: {z_score:.2f}")

        # 5. 风控与平仓逻辑 (优先处理已经在持仓中的标的)
        if target_symbol in self.trade_states:
            state = self.trade_states[target_symbol]
            entry_price = state["entry_price"]
            side = state["side"]
            
            # --- 做多持仓的风控 ---
            if side == "long":
                # 更新跟踪止损最高点
                if target_current_price > state["highest_price"]:
                    state["highest_price"] = target_current_price
                    
                profit_pct = (target_current_price - entry_price) / entry_price
                highest_profit_pct = (state["highest_price"] - entry_price) / entry_price
                
                # 硬止损
                if profit_pct <= -self.params.hard_stop_loss_pct:
                    signals.append(Signal(type=SignalType.EXIT_LONG, symbol=target_symbol, timestamp=current_ts, price=target_current_price, reason="StatArb Hard Stop Loss Drop"))
                    return signals
                    
                # 追踪止损激活
                if not state.get("trailing_active", False) and highest_profit_pct >= self.params.trailing_activation_pct:
                    state["trailing_active"] = True
                    
                # 追踪回撤平仓
                if state.get("trailing_active", False):
                    drawdown_from_high = (state["highest_price"] - target_current_price) / state["highest_price"]
                    if drawdown_from_high >= self.params.trailing_callback_pct:
                        signals.append(Signal(type=SignalType.EXIT_LONG, symbol=target_symbol, timestamp=current_ts, price=target_current_price, reason="StatArb Trailing Stop Fallback"))
                        return signals
                        
                # 均线回归平仓 (核心逻辑：Z-Score 回归 0 或超过 0)
                if z_score >= self.params.z_score_exit:
                    signals.append(Signal(type=SignalType.EXIT_LONG, symbol=target_symbol, timestamp=current_ts, price=target_current_price, reason="Z-Score Mean Reversion Achieved"))
                    return signals


            # --- 做空持仓的风控 ---
            elif side == "short":
                # 更新做空的最有利极值 (最低价)
                if target_current_price < state["lowest_price"]:
                    state["lowest_price"] = target_current_price
                    
                profit_pct = (entry_price - target_current_price) / entry_price
                highest_profit_pct = (entry_price - state["lowest_price"]) / entry_price
                
                # 硬止损
                if profit_pct <= -self.params.hard_stop_loss_pct:
                    signals.append(Signal(type=SignalType.EXIT_SHORT, symbol=target_symbol, timestamp=current_ts, price=target_current_price, reason="StatArb Short Hard Stop Loss Hit"))
                    return signals
                    
                # 追踪止损激活
                if not state.get("trailing_active", False) and highest_profit_pct >= self.params.trailing_activation_pct:
                    state["trailing_active"] = True
                    
                # 追踪回撤平仓 (对于做空，价格反弹则是回撤)
                if state.get("trailing_active", False):
                    drawdown_from_low = (target_current_price - state["lowest_price"]) / state["lowest_price"]
                    if drawdown_from_low >= self.params.trailing_callback_pct:
                        signals.append(Signal(type=SignalType.EXIT_SHORT, symbol=target_symbol, timestamp=current_ts, price=target_current_price, reason="StatArb Short Trailing Stop Fallback"))
                        return signals
                        
                # 均线回归平仓 (核心逻辑：Z-Score 回归 0 或低于 0)
                if z_score <= self.params.z_score_exit:
                    signals.append(Signal(type=SignalType.EXIT_SHORT, symbol=target_symbol, timestamp=current_ts, price=target_current_price, reason="Short Z-Score Mean Reversion Achieved"))
                    return signals

            # 仍在持仓安全区内，不触发新方向
            return signals

        # 6. 未持仓，探寻入场信号
        # 当 target 和 base 极度脱离，且 Target 明显落后 (跌出 Z < -2.0)，做多 Target 吃补涨
        if z_score < self.params.z_score_long_entry:
            signals.append(Signal(
                type=SignalType.BUY, 
                symbol=target_symbol, 
                timestamp=current_ts, 
                price=target_current_price, 
                reason=f"StatArb Z-Score Dive (Z={z_score:.2f} < {self.params.z_score_long_entry})"
            ))
            # 标记模拟入场状态
            self.trade_states[target_symbol] = {
                "entry_price": target_current_price,
                "highest_price": target_current_price,
                "side": "long",
                "trailing_active": False
            }
            
        # 当 Target 相比 Base 极度超出 (涨出 Z > 2.0)，且你允许双向套利的话，做空 Target
        elif z_score > self.params.z_score_short_entry:
            signals.append(Signal(
                type=SignalType.SELL, 
                symbol=target_symbol, 
                timestamp=current_ts, 
                price=target_current_price, 
                reason=f"StatArb Z-Score Spike (Z={z_score:.2f} > {self.params.z_score_short_entry})"
            ))
            # 标记模拟入场状态
            self.trade_states[target_symbol] = {
                "entry_price": target_current_price,
                "lowest_price": target_current_price,
                "side": "short",
                "trailing_active": False
            }

        return signals


    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """兼容接口，如果不用全局行情总线调用的话返回空"""
        return []

    def calculate_position_size(self, signal: Signal, context: Any) -> float:
        """从 BaseStrategy 继承的必须实现的方法 (计算下单数量)"""
        # 这个方法如果在引擎层自己直接手写了 amount 计算，那么这里只是为了满足接口存在
        # 当前 StatArbEngine 中已经亲自根据 free_usdt 计算了数量，所以这个接口暂时作兼容
        target_value = context.portfolio_value * self.params.position_size
        return target_value / signal.price if signal.price > 0 else 0
