"""跨维度动量策略：链上异动 -> CEX执行信号 (On-Chain Momentum)"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd

from quant_trading.signal import Signal, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.logger.logger import get_logger

logger = get_logger("onchain_momentum")

@dataclass
class OnChainMomentumParams(StrategyParams):
    """链上动量截获参数"""
    whale_usd_threshold: float = 50000.0  # 定义链上“聪明钱”单笔大单的阈值 (比如 5万美元)
    eth_price_usd: float = 2300.0         # 用于估值转换的以太坊参考基准价 (在引擎中会实时更新)
    
    # 策略风控
    position_size: float = 0.1            # 单笔开仓占总资金的比例
    hard_stop_loss_pct: float = 0.05
    trailing_activation_pct: float = 0.015
    trailing_callback_pct: float = 0.005


class OnChainMomentumStrategy(BaseStrategy):
    """基于链上巨鲸流动性池 Swap 事件的动量跟随引擎
    
    核心范式：
    监控诸如 PEPE/WETH 的 Uniswap V3 流动池。
    如果有人向池子里砸入了价值 > $50,000 的 WETH 并带走 PEPE，说明链上巨鲸买入 PEPE。
    此时抛出包含价格和方向的信号，传递给下游的 OKX/Gate 执行！
    """
    
    name = "onchain_momentum"
    
    def __init__(
        self,
        symbol: str, 
        params: Optional[OnChainMomentumParams] = None,
    ) -> None:
        super().__init__(symbol, params or OnChainMomentumParams())
        self.params: OnChainMomentumParams = self.params
        
        # trade_states 拦截和接管持仓风控
        self.trade_states: Dict[str, Dict[str, Any]] = {}


    def generate_signals_from_onchain(self, swap_event: Dict[str, Any], cex_symbol: str, current_cex_price: float) -> List[Signal]:
        """
        接收链上解析好的 Swap 事件，输出 CEX 交易信号
        swap_event 的格式 (来源于 evm_ws_client.py):
        {
            "tx_hash": "0x123...",
            "pool_address": "0x...",
            "token0_symbol": "WETH",
            "token1_symbol": "PEPE",
            "real_amount0": 25.5,  # WETH
            "real_amount1": -18000000000 # PEPE
        }
        
        解析规则示例（如果 base token 是 WETH (token0)）:
        - real_amount0 > 0: 交易者卖出 WETH 丢进池子 (流入)
        - real_amount1 < 0: 交易者带走了 PEPE (买入)
        因此，这笔交易反映的是买入 PEPE!
        """
        signals: List[Signal] = []
        
        # 实时动态风控维护 (每次更新都走一遍，不管有没有 Swap)
        self._update_risk_management(cex_symbol, current_cex_price, signals)
        
        token0 = swap_event.get("token0_symbol", "")
        token1 = swap_event.get("token1_symbol", "")
        amount0 = swap_event.get("real_amount0", 0.0)
        amount1 = swap_event.get("real_amount1", 0.0)
        
        # 我们需要判断谁是 Base(U/WETH)，谁是 Target(山寨币)
        # 为了通用，我们看谁的价值被拿来做计价参考。这里假设 token0 是 WETH。
        if token0 == "WETH":
            base_amount = amount0
            base_price = self.params.eth_price_usd
            is_buying_altcoin = amount0 > 0 and amount1 < 0 # WETH流入，山寨币流出
            is_selling_altcoin = amount0 < 0 and amount1 > 0 # WETH流出，山寨币流入
        elif token1 == "WETH":
            base_amount = amount1
            base_price = self.params.eth_price_usd
            is_buying_altcoin = amount1 > 0 and amount0 < 0 
            is_selling_altcoin = amount1 < 0 and amount0 > 0
        else:
            # TODO: 支持纯 USDC 等非 ETH 池
            return signals
            
        # 换算这笔交易对应的美金价值
        tx_usd_value = abs(base_amount) * base_price
        
        # 如果不是大单，忽略
        if tx_usd_value < self.params.whale_usd_threshold:
            return signals
            
        short_hash = f"{swap_event['tx_hash'][:6]}..{swap_event['tx_hash'][-4:]}"
        
        if is_buying_altcoin:
            logger.info(f"[🐋 链上核弹警告] 巨鲸在 DEX 爆买价值 ${tx_usd_value:,.0f} 的 {cex_symbol.split('/')[0]}! (Tx: {short_hash})")
            
            # 如果我们还没持仓，发出买入指令追单
            if cex_symbol not in self.trade_states or self.trade_states[cex_symbol]["side"] != "long":
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=cex_symbol,
                    timestamp=0, # 我们在此体系中忽略精确时间戳，执行引擎按“立刻执行”处理
                    price=current_cex_price,
                    reason=f"OnChain Whale BUY (${tx_usd_value:,.0f})"
                ))
                self.trade_states[cex_symbol] = {
                    "entry_price": current_cex_price,
                    "highest_price": current_cex_price,
                    "side": "long",
                    "trailing_active": False
                }
                
        elif is_selling_altcoin:
            logger.info(f"[🚨 链上塌方警告] 巨鲸在 DEX 清仓抛售 ${tx_usd_value:,.0f} 的 {cex_symbol.split('/')[0]}! (Tx: {short_hash})")
            
            # 此处如果你允许做双向交易，你可以做空。此版为了稳妥也可以仅仅当作平仓信号
            if cex_symbol in self.trade_states and self.trade_states[cex_symbol]["side"] == "long":
                signals.append(Signal(
                    type=SignalType.EXIT_LONG,
                    symbol=cex_symbol,
                    timestamp=0,
                    price=current_cex_price,
                    reason=f"OnChain Whale SETUP PANIC (${tx_usd_value:,.0f})"
                ))
            elif cex_symbol not in self.trade_states or self.trade_states[cex_symbol]["side"] != "short":
                signals.append(Signal(
                    type=SignalType.SELL,
                    symbol=cex_symbol,
                    timestamp=0,
                    price=current_cex_price,
                    reason=f"OnChain Whale SELL (${tx_usd_value:,.0f})"
                ))
                self.trade_states[cex_symbol] = {
                    "entry_price": current_cex_price,
                    "lowest_price": current_cex_price,
                    "side": "short",
                    "trailing_active": False
                }

        return signals

    def _update_risk_management(self, cex_symbol: str, current_price: float, pending_signals: List[Signal]):
        """管理现有持仓的波幅止损逻辑"""
        if cex_symbol not in self.trade_states:
            return
            
        state = self.trade_states[cex_symbol]
        entry_price = state["entry_price"]
        side = state["side"]
        
        if side == "long":
            if current_price > state["highest_price"]:
                state["highest_price"] = current_price
                
            profit_pct = (current_price - entry_price) / entry_price
            highest_profit_pct = (state["highest_price"] - entry_price) / entry_price
            
            if profit_pct <= -self.params.hard_stop_loss_pct:
                pending_signals.append(Signal(type=SignalType.EXIT_LONG, symbol=cex_symbol, timestamp=0, price=current_price, reason="Whale Momentum Stop Loss Hit"))
                del self.trade_states[cex_symbol]
                return
                
            if not state.get("trailing_active", False) and highest_profit_pct >= self.params.trailing_activation_pct:
                state["trailing_active"] = True
                
            if state.get("trailing_active", False):
                drawdown = (state["highest_price"] - current_price) / state["highest_price"]
                if drawdown >= self.params.trailing_callback_pct:
                    pending_signals.append(Signal(type=SignalType.EXIT_LONG, symbol=cex_symbol, timestamp=0, price=current_price, reason="Whale Momentum Trailing Exit"))
                    del self.trade_states[cex_symbol]

        elif side == "short":
            if current_price < state.get("lowest_price", entry_price):
                state["lowest_price"] = current_price
                
            profit_pct = (entry_price - current_price) / entry_price
            highest_profit_pct = (entry_price - state.get("lowest_price", entry_price)) / entry_price
            
            if profit_pct <= -self.params.hard_stop_loss_pct:
                pending_signals.append(Signal(type=SignalType.EXIT_SHORT, symbol=cex_symbol, timestamp=0, price=current_price, reason="Whale Momentum Short Stop Loss"))
                del self.trade_states[cex_symbol]
                return
                
            if not state.get("trailing_active", False) and highest_profit_pct >= self.params.trailing_activation_pct:
                state["trailing_active"] = True
                
            if state.get("trailing_active", False):
                drawdown = (current_price - state["lowest_price"]) / state["lowest_price"]
                if drawdown >= self.params.trailing_callback_pct:
                    pending_signals.append(Signal(type=SignalType.EXIT_SHORT, symbol=cex_symbol, timestamp=0, price=current_price, reason="Whale Momentum Short Trailing Exit"))
                    del self.trade_states[cex_symbol]


    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """BaseStrategy 默认接口兼容"""
        return []

    def calculate_position_size(self, signal: Signal, context: Any) -> float:
        """从 BaseStrategy 继承的必须实现的方法 (计算下单数量)"""
        target_value = context.portfolio_value * self.params.position_size
        return target_value / signal.price if signal.price > 0 else 0
