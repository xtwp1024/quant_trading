"""
Options Backtest Engine - 期权回测引擎
Phase 2: 回测框架 + 基础策略
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

from .data.models import OptionContract, OptionType
from .pricing.black_scholes import BlackScholes, bs_price, bs_greeks
from .pricing.greeks import calculate_greeks


class OptionPositionSide(Enum):
    """期权持仓方向"""
    LONG = 1   # 买入
    SHORT = -1  # 卖出
    FLAT = 0    # 空仓


@dataclass
class OptionPosition:
    """期权持仓"""
    contract: OptionContract  # 期权合约信息
    side: OptionPositionSide  # 持仓方向
    size: float              # 持仓数量（合约数）
    entry_premium: float    # 开仓期权费
    entry_timestamp: int     # 开仓时间戳

    @property
    def entry_value(self) -> float:
        """开仓价值（权利金支出/收入）"""
        sign = 1 if self.side == OptionPositionSide.LONG else -1
        return sign * self.size * self.entry_premium

    @property
    def premium(self) -> float:
        """当前期权费"""
        return self.contract.mark_price


@dataclass
class OptionTrade:
    """期权交易记录"""
    timestamp: int
    symbol: str
    option_type: str  # "call" or "put"
    strike_price: float
    expiration_timestamp: int
    side: str  # "OPEN", "CLOSE"
    position_side: str  # "LONG", "SHORT"
    size: float
    premium: float
    pnl: float = 0.0
    commission: float = 0.0


@dataclass
class OptionBacktestConfig:
    """期权回测配置"""
    symbol: str = "ETH"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000.0
    commission: float = 0.0015  # 0.15% 手续费
    slippage: float = 0.001  # 0.1% 滑点
    risk_free_rate: float = 0.05  # 5% 无风险利率


@dataclass
class OptionBacktestResult:
    """期权回测结果"""
    config: OptionBacktestConfig
    trades: List[OptionTrade] = field(default_factory=list)
    final_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_premium_paid: float = 0.0
    total_premium_received: float = 0.0

    # 期权特有指标
    max_correlation_exposure: float = 0.0  # 最大相关资产暴露
    avg_days_to_expiry: float = 0.0  # 平均持仓天数

    def to_dict(self) -> Dict:
        return {
            "final_equity": round(self.final_equity, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": f"{self.max_drawdown_pct * 100:.2f}%",
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": f"{self.win_rate * 100:.2f}%",
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


class OptionBacktestEngine:
    """期权回测引擎"""

    def __init__(
        self,
        config: OptionBacktestConfig = None,
        historical_data: Dict = None,
    ):
        self.config = config or OptionBacktestConfig()
        self.historical_data = historical_data or {}  # {date: {strike: OptionContract}}
        self.cash = self.config.initial_balance
        self.peak_cash = self.cash
        self.positions: List[OptionPosition] = []
        self.trades: List[OptionTrade] = []
        self.equity_curve: List[Tuple[int, float]] = []
        self.daily_pnl: List[float] = []

    def open_position(
        self,
        timestamp: int,
        symbol: str,
        option_type: str,
        strike_price: float,
        expiration_timestamp: int,
        size: float,
        premium: float,
        position_side: OptionPositionSide = OptionPositionSide.LONG,
    ):
        """开仓"""
        # 创建模拟合约
        contract = OptionContract(
            symbol=f"{symbol}-{strike_price}-{option_type}",
            option_type=OptionType.CALL if option_type.lower() == "call" else OptionType.PUT,
            strike_price=strike_price,
            expiration_timestamp=expiration_timestamp,
            mark_price=premium,
            bid_price=premium * 0.99,
            ask_price=premium * 1.01,
            underlying_price=self._get_underlying_price(timestamp),
        )

        position = OptionPosition(
            contract=contract,
            side=position_side,
            size=size,
            entry_premium=premium,
            entry_timestamp=timestamp,
        )

        self.positions.append(position)

        # 记录交易
        # 买入：付权利金；卖出：收权利金
        if position_side == OptionPositionSide.LONG:
            cost = size * premium * (1 + self.config.commission + self.config.slippage)
            self.cash -= cost
            trade_side = "LONG"
        else:
            proceeds = size * premium * (1 - self.config.commission - self.config.slippage)
            self.cash += proceeds
            trade_side = "SHORT"

        trade = OptionTrade(
            timestamp=timestamp,
            symbol=symbol,
            option_type=option_type,
            strike_price=strike_price,
            expiration_timestamp=expiration_timestamp,
            side="OPEN",
            position_side=trade_side,
            size=size,
            premium=premium,
            commission=size * premium * self.config.commission,
        )
        self.trades.append(trade)

    def close_position(
        self,
        position_idx: int,
        timestamp: int,
        premium: float,
    ):
        """平仓"""
        if position_idx >= len(self.positions):
            return

        position = self.positions[position_idx]

        # 计算盈亏
        if position.side == OptionPositionSide.LONG:
            # 买入期权平仓：卖出收钱
            pnl = (premium - position.entry_premium) * position.size
            proceeds = premium * position.size * (1 - self.config.commission - self.config.slippage)
            self.cash += proceeds
        else:
            # 卖出期权平仓：买回付钱
            pnl = (position.entry_premium - premium) * position.size
            cost = premium * position.size * (1 + self.config.commission + self.config.slippage)
            self.cash -= cost

        # 记录交易
        trade = OptionTrade(
            timestamp=timestamp,
            symbol=position.contract.symbol,
            option_type=position.contract.option_type.value,
            strike_price=position.contract.strike_price,
            expiration_timestamp=position.contract.expiration_timestamp,
            side="CLOSE",
            position_side=position.side.value,
            size=position.size,
            premium=premium,
            pnl=pnl,
            commission=position.size * premium * self.config.commission,
        )
        self.trades.append(trade)

        # 从持仓中移除
        self.positions.pop(position_idx)

        return pnl

    def update_positions(self, timestamp: int, underlying_price: float):
        """更新持仓价格（按市价重估）"""
        total_unrealized_pnl = 0.0

        for position in self.positions:
            # 更新合约的当前价格（使用 Black-Scholes 重估）
            position.contract.underlying_price = underlying_price

            T = (position.contract.expiration_timestamp - timestamp) / (365 * 24 * 60 * 60 * 1000)
            if T <= 0:
                T = 1 / 365  # 至少1天

            try:
                current_premium = bs_price(
                    S=underlying_price,
                    K=position.contract.strike_price,
                    T=T,
                    r=self.config.risk_free_rate,
                    sigma=0.80,  # 使用固定 IV
                    option_type=position.contract.option_type.value,
                )
                position.contract.mark_price = current_premium
            except Exception as e:
                # 记录错误而不是静默忽略
                import logging
                logging.warning(f"Failed to calculate BS price for {position.contract.symbol}: {e}")

            # 计算未实现盈亏
            if position.side == OptionPositionSide.LONG:
                unrealized = (position.contract.mark_price - position.entry_premium) * position.size
            else:
                unrealized = (position.entry_premium - position.contract.mark_price) * position.size

            total_unrealized_pnl += unrealized

        # 记录权益曲线
        current_equity = self.cash + total_unrealized_pnl
        if current_equity > self.peak_cash:
            self.peak_cash = current_equity
        self.equity_curve.append((timestamp, current_equity))

    def check_expiration(self, timestamp: int, underlying_price: float) -> List[int]:
        """检查期权到期，返回需要平仓的持仓索引"""
        expired_indices = []

        for i, position in enumerate(self.positions):
            if position.contract.expiration_timestamp <= timestamp:
                # 期权到期，计算最终价值
                # ATM 期权可能被行权，其他自动归零
                K = position.contract.strike_price
                S = underlying_price
                opt_type = position.contract.option_type.value

                # 简单判断：ITM 则可能被行权
                if opt_type == "call" and S > K:
                    # Call ITM: 按 (S-K) 结算
                    if position.side == OptionPositionSide.LONG:
                        # Long Call 被行权，获得 S-K
                        pnl = (S - K - position.entry_premium) * position.size
                    else:
                        # Short Call 被行权，损失 S-K
                        pnl = (position.entry_premium - (S - K)) * position.size
                elif opt_type == "put" and S < K:
                    # Put ITM: 按 (K-S) 结算
                    if position.side == OptionPositionSide.LONG:
                        pnl = (K - S - position.entry_premium) * position.size
                    else:
                        pnl = (position.entry_premium - (K - S)) * position.size
                else:
                    # OTM 期权归零
                    if position.side == OptionPositionSide.LONG:
                        pnl = -position.entry_premium * position.size
                    else:
                        pnl = position.entry_premium * position.size

                # 执行平仓
                self.cash += pnl
                expired_indices.append(i)

                # 记录到期交易
                trade = OptionTrade(
                    timestamp=timestamp,
                    symbol=position.contract.symbol,
                    option_type=opt_type,
                    strike_price=K,
                    expiration_timestamp=position.contract.expiration_timestamp,
                    side="EXPIRED",
                    position_side=position.side.value,
                    size=position.size,
                    premium=0,
                    pnl=pnl,
                )
                self.trades.append(trade)

        # 逆序删除（避免索引混乱）
        for i in sorted(expired_indices, reverse=True):
            self.positions.pop(i)

        return expired_indices

    def _get_underlying_price(self, timestamp: int) -> float:
        """获取标的资产价格"""
        # 从历史数据或使用默认价格
        return 2177.0  # 默认 ETH 价格

    def get_portfolio_greeks(self, timestamp: int) -> Dict:
        """获取组合 Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for position in self.positions:
            T = (position.contract.expiration_timestamp - timestamp) / (365 * 24 * 60 * 60 * 1000)
            if T <= 0:
                T = 1 / 365

            greeks = calculate_greeks(
                S=position.contract.underlying_price,
                K=position.contract.strike_price,
                T=T,
                r=self.config.risk_free_rate,
                sigma=0.80,
                option_type=position.contract.option_type.value,
            )

            size = position.size * (1 if position.side == OptionPositionSide.LONG else -1)
            total_delta += greeks.delta * size
            total_gamma += greeks.gamma * size
            total_vega += greeks.vega * size
            total_theta += greeks.theta * size

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta,
        }

    def calculate_results(self) -> OptionBacktestResult:
        """计算回测结果"""
        closed_trades = [t for t in self.trades if t.side in ("CLOSE", "EXPIRED")]

        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing)) if losing else 0

        # 计算最大回撤
        equity_values = [e[1] for e in self.equity_curve]
        max_equity = self.config.initial_balance
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        for eq in equity_values:
            if eq > max_equity:
                max_equity = eq
            dd = max_equity - eq
            if dd > max_drawdown:
                max_drawdown = dd
                max_drawdown_pct = dd / max_equity if max_equity > 0 else 0

        # 计算夏普比率
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_eq = self.equity_curve[i-1][1]
            curr_eq = self.equity_curve[i][1]
            if prev_eq > 0:
                ret = (curr_eq - prev_eq) / prev_eq
                returns.append(ret)

        sharpe = 0.0
        if returns and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        result = OptionBacktestResult(
            config=self.config,
            trades=self.trades,
            final_equity=self.cash,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(closed_trades) if closed_trades else 0,
            avg_win=total_wins / len(winning) if winning else 0,
            avg_loss=total_losses / len(losing) if losing else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else 0,
            sharpe_ratio=sharpe,
        )

        return result


class OptionStrategy:
    """期权策略基类"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def generate_signals(
        self,
        timestamp: int,
        underlying_price: float,
        option_chain: Dict,
    ) -> List[Dict]:
        """
        生成交易信号

        Returns:
            List of signals, each with:
            {
                "action": "OPEN" or "CLOSE",
                "option_type": "call" or "put",
                "strike_price": float,
                "expiration_timestamp": int,
                "size": float,
                "side": "LONG" or "SHORT",
            }
        """
        raise NotImplementedError


class LongStraddleStrategy(OptionStrategy):
    """Long Straddle（跨式组合）"""
    # 同时买入相同行权价的 Call 和 Put

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.strike_pct = self.config.get("strike_pct", 0)  # ATM=0, OTM%=0.05

    def generate_signals(self, timestamp, underlying_price, option_chain):
        signals = []

        # 找 ATM 行权价
        strikes = list(option_chain.keys())
        if not strikes:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        if self.strike_pct != 0:
            # 找 OTM 行权价
            direction = 1 if self.strike_pct > 0 else -1
            atm_strike = atm_strike * (1 + self.strike_pct) if direction > 0 else atm_strike * (1 + self.strike_pct)

        call_price = option_chain.get(atm_strike, {}).get("call_price", 0)
        put_price = option_chain.get(atm_strike, {}).get("put_price", 0)

        if call_price > 0 and put_price > 0:
            # 开仓 Long Straddle
            signals.append({
                "action": "OPEN",
                "option_type": "call",
                "strike_price": atm_strike,
                "premium": call_price,
                "side": "LONG",
                "size": 1,
            })
            signals.append({
                "action": "OPEN",
                "option_type": "put",
                "strike_price": atm_strike,
                "premium": put_price,
                "side": "LONG",
                "size": 1,
            })

        return signals


class IronCondorStrategy(OptionStrategy):
    """Iron Condor（铁秃鹰）"""
    # 卖出一个 OTM Call + 买一个更 OTM Call
    # 卖出一个 OTM Put + 买一个更 OTM Put

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.call_spread = self.config.get("call_spread", 0.05)  # 5% OTM
        self.put_spread = self.config.get("put_spread", 0.05)      # 5% OTM
        self.width = self.config.get("width", 0.02)               # 价差宽度 2%

    def generate_signals(self, timestamp, underlying_price, option_chain):
        signals = []

        strikes = sorted(option_chain.keys())
        if len(strikes) < 4:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))

        # 找行权价
        otm_call_strike = atm_strike * (1 + self.call_spread)
        otm_call_strike = min(strikes, key=lambda x: abs(x - otm_call_strike))

        otm_put_strike = atm_strike * (1 - self.put_spread)
        otm_put_strike = min(strikes, key=lambda x: abs(x - otm_put_strike))

        # 找更远的行权价作为保护
        call_protection = otm_call_strike * (1 + self.width)
        put_protection = otm_put_strike * (1 - self.width)

        try:
            call_protection = min(strikes, key=lambda x: abs(x - call_protection))
            put_protection = min(strikes, key=lambda x: abs(x - put_protection))
        except Exception as e:
            import logging
            logging.warning(f"Failed to calculate protection strikes for Iron Condor: {e}")
            call_protection = otm_call_strike * 1.05
            put_protection = otm_put_strike * 0.95

        # 获取价格
        short_call_price = option_chain.get(otm_call_strike, {}).get("call_price", 0)
        long_call_price = option_chain.get(call_protection, {}).get("call_price", 0)
        short_put_price = option_chain.get(otm_put_strike, {}).get("put_price", 0)
        long_put_price = option_chain.get(put_protection, {}).get("put_price", 0)

        if short_call_price > 0 and long_call_price > 0 and short_put_price > 0 and long_put_price > 0:
            # 卖出 Call Spread (Short Call + Long Call 保护)
            signals.append({
                "action": "OPEN",
                "option_type": "call",
                "strike_price": otm_call_strike,
                "premium": short_call_price,
                "side": "SHORT",
                "size": 1,
            })
            signals.append({
                "action": "OPEN",
                "option_type": "call",
                "strike_price": call_protection,
                "premium": long_call_price,
                "side": "LONG",
                "size": 1,
            })

            # 卖出 Put Spread (Short Put + Long Put 保护)
            signals.append({
                "action": "OPEN",
                "option_type": "put",
                "strike_price": otm_put_strike,
                "premium": short_put_price,
                "side": "SHORT",
                "size": 1,
            })
            signals.append({
                "action": "OPEN",
                "option_type": "put",
                "strike_price": put_protection,
                "premium": long_put_price,
                "side": "LONG",
                "size": 1,
            })

        return signals


class CoveredCallStrategy(OptionStrategy):
    """Covered Call（备兑看涨期权）"""
    # 持有标的资产，卖出 OTM Call

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.strike_pct = self.config.get("strike_pct", 0.03)  # 3% OTM

    def generate_signals(self, timestamp, underlying_price, option_chain):
        signals = []

        strikes = sorted(option_chain.keys())
        if not strikes:
            return signals

        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))

        # 找 OTM Call
        target_strike = atm_strike * (1 + self.strike_pct)
        target_strike = min(strikes, key=lambda x: abs(x - target_strike))

        call_price = option_chain.get(target_strike, {}).get("call_price", 0)

        if call_price > 0:
            signals.append({
                "action": "OPEN",
                "option_type": "call",
                "strike_price": target_strike,
                "premium": call_price,
                "side": "SHORT",
                "size": 1,
            })

        return signals


if __name__ == "__main__":
    # 简单测试
    engine = OptionBacktestEngine()

    # 模拟：ETH $2177，买入 ATM Straddle，30天后到期
    now = int(datetime.now().timestamp() * 1000)
    expiry = int((datetime.now() + timedelta(days=30)).timestamp() * 1000)

    # 买入 Call
    engine.open_position(
        timestamp=now,
        symbol="ETH",
        option_type="call",
        strike_price=2200,
        expiration_timestamp=expiry,
        size=1,
        premium=192.48,  # BS 计算的理论价
    )

    # 买入 Put
    engine.open_position(
        timestamp=now,
        symbol="ETH",
        option_type="put",
        strike_price=2150,
        expiration_timestamp=expiry,
        size=1,
        premium=200.0,
    )

    print(f"Opened positions: {len(engine.positions)}")
    print(f"Cash: ${engine.cash:.2f}")

    # 模拟价格变动
    engine.update_positions(now + 86400000 * 7, 2200)  # 7天后 ETH=$2200
    print(f"After 7 days - Cash: ${engine.cash:.2f}")
