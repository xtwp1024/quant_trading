"""
PA-AMM做市策略

基于论文: arXiv:2602.09887
"Partially Active Automated Market Makers"

核心思想:
- 每个区块只开放部分流动性(λ)给交易
- 降低LP被套利的LVR损失
- 平衡LVR损失与价格跟踪误差
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class PAAMMParams(StrategyParams):
    """PA-AMM策略参数"""
    lambda_activeness: float = 0.5  # 活跃度 [0, 1]
    fee_rate: float = 0.003  # 手续费率
    gamma: float = 1.0  # LVR权重参数
    inventory_target: float = 0.5  # 目标库存比例
    max_inventory: float = 1.0  # 最大库存
    min_spread: float = 0.001  # 最小价差
    position_size: float = 0.1


class PAAMMMaker:
    """
    部分活跃自动做市商
    
    在每个时间窗口只使用部分流动性
    """
    
    def __init__(
        self,
        lambda_activeness: float = 0.5,
        fee_rate: float = 0.003,
    ):
        self.lambda_activeness = lambda_activeness
        self.fee_rate = fee_rate
        
        self.active_reserves: Dict[str, float] = {}
        self.passive_reserves: Dict[str, float] = {}
    
    def set_reserves(
        self,
        token_a: str,
        amount_a: float,
        token_b: str,
        amount_b: float,
    ) -> None:
        """设置储备"""
        active_a = amount_a * self.lambda_activeness
        passive_a = amount_a - active_a
        
        active_b = amount_b * self.lambda_activeness
        passive_b = amount_b - active_b
        
        self.active_reserves[token_a] = active_a
        self.passive_reserves[token_a] = passive_a
        self.active_reserves[token_b] = active_b
        self.passive_reserves[token_b] = passive_b
    
    def get_spot_price(self, token_a: str, token_b: str) -> float:
        """获取现货价格"""
        if token_a not in self.active_reserves or token_b not in self.active_reserves:
            return 0.0
        
        return self.active_reserves[token_b] / self.active_reserves[token_a]
    
    def calculate_output(
        self,
        input_token: str,
        output_token: str,
        input_amount: float,
    ) -> float:
        """计算输出数量"""
        if input_token not in self.active_reserves:
            return 0.0
        
        r_in = self.active_reserves[input_token]
        r_out = self.active_reserves[output_token]
        
        if r_in <= 0 or r_out <= 0:
            return 0.0
        
        k = r_in * r_in + r_out * r_out
        
        new_r_in = r_in + input_amount * (1 - self.fee_rate)
        new_r_out = math.sqrt(k - new_r_in * new_r_in)
        
        output_amount = r_out - new_r_out
        
        return max(0, output_amount)
    
    def optimal_lambda(self, gamma: float) -> float:
        """
        计算最优活跃度
        
        λ* = (1 + √(1 + 2γ)) / (1 + γ + √(1 + 2γ))
        """
        sqrt_term = math.sqrt(1 + 2 * gamma)
        optimal = (1 + sqrt_term) / (1 + gamma + sqrt_term)
        
        return max(0.01, min(0.99, optimal))


class LVRTracker:
    """
    LVR（Loss-Versus-Rebalancing）追踪器
    """
    
    def __init__(self):
        self.lvr_history: List[float] = []
    
    def calculate_lvr(
        self,
        entry_price: float,
        exit_price: float,
        amount: float,
    ) -> float:
        """
        计算LVR损失
        
        LVR = (P_external - P_AMM) * Amount
        """
        lvr = abs(exit_price - entry_price) * amount
        self.lvr_history.append(lvr)
        return lvr
    
    def get_total_lvr(self) -> float:
        """获取总LVR"""
        return sum(self.lvr_history)


class PAAMMStrategy(BaseStrategy):
    """
    PA-AMM做市策略
    
    通过部分活跃机制降低LVR损失
    """
    
    name = "pa_amm"
    
    def __init__(
        self,
        symbol: str,
        params: Optional[PAAMMParams] = None,
    ) -> None:
        super().__init__(symbol, params or PAAMMParams())
        
        base, quote = symbol.split("/")
        self.base_token = base
        self.quote_token = quote
        
        self.maker = PAAMMMaker(
            lambda_activeness=params.lambda_activeness if params else 0.5,
            fee_rate=params.fee_rate if params else 0.003,
        )
        
        self.lvr_tracker = LVRTracker()
        
        self._inventory: float = 0.0
        self._cash: float = 0.0
        self._last_price: Optional[float] = None
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        df = data.copy()
        
        df["mid_price"] = (df["high"] + df["low"]) / 2
        df["spread"] = df["high"] - df["low"]
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        
        optimal_lambda = self.maker.optimal_lambda(self.params.gamma)
        
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            current_price = current["close"]
            spread = current["spread"]
            volatility = current["volatility"]
            
            mid_price = current["mid_price"]
            half_spread = max(spread / 2, mid_price * self.params.min_spread)
            
            bid_price = mid_price - half_spread
            ask_price = mid_price + half_spread
            
            inventory_ratio = self._inventory / self.params.max_inventory if self.params.max_inventory > 0 else 0
            
            if inventory_ratio > self.params.inventory_target:
                bid_price *= 0.99
                ask_price *= 0.99
            
            elif inventory_ratio < self.params.inventory_target:
                bid_price *= 1.01
                ask_price *= 1.01
            
            if self._last_price is not None:
                price_change = abs(current_price - self._last_price) / self._last_price
                
                if price_change > volatility * 2:
                    signals.append(
                        Signal(
                            type=SignalType.SELL if self._inventory > 0 else SignalType.BUY,
                            symbol=self.symbol,
                            timestamp=int(current["timestamp"]),
                            price=float(current_price),
                            strength=optimal_lambda,
                            reason=f"Price movement detected, rebalancing (λ={optimal_lambda:.2f})",
                            metadata={
                                "lambda": optimal_lambda,
                                "inventory_ratio": inventory_ratio,
                                "price_change": float(price_change),
                            },
                        )
                    )
            
            self._last_price = current_price
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位"""
        base_size = context.portfolio_value * self.params.position_size
        
        lambda_factor = signal.metadata.get("lambda", 0.5) if signal.metadata else 0.5
        
        adjusted_size = base_size * lambda_factor * signal.strength
        
        return adjusted_size / signal.price
    
    def update_inventory(self, amount: float, price: float, side: str) -> None:
        """更新库存"""
        if side == "buy":
            self._inventory += amount
            self._cash -= amount * price
        else:
            self._inventory -= amount
            self._cash += amount * price


"""
参考文献:
---------
[1] arXiv:2602.09887 - Partially Active Automated Market Makers
    Author: Sunghun Ko

关键概念:
---------
1. 部分活跃机制:
   - 每个区块只开放λ比例的流动性
   - λ=1 时退化为传统AMM
   - λ<1 时降低LVR损失

2. LVR (Loss-Versus-Rebalancing):
   - LP因信息不对称遭受的损失
   - 与套利者的逆向选择成本

3. 价格跟踪误差:
   - AMM价格与外部价格的偏离
   - λ越小，偏离越大

4. 最优活跃度:
   - λ* = (1 + √(1 + 2γ)) / (1 + γ + √(1 + 2γ))
   - γ越大（越重视LVR），λ*越小

数学公式:
---------
1. 价格偏离方差:
   E[g²] ≈ σ²Δt / (λ(2-λ))

2. LVR下降:
   与 1/(2-λ) 相关

3. 最优λ:
   权衡LVR损失与跟踪误差

实现注意事项:
-------------
- 本实现为概念验证版本
- 完整实现需要链上数据
- 需要考虑Gas费用
- 需要考虑MEV风险
- 需要实现动态λ调整
"""
