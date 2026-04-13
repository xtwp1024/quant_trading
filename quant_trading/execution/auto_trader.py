# -*- coding: utf-8 -*-
"""
自动跟单交易系统
Auto-Follow Trading System

根据概率引擎信号自动执行交易

Usage:
    python -m quant_trading.execution.auto_trader

Author: Claude
"""

from __future__ import annotations
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any
import json

# 自动加载 .env
_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

import numpy as np
import pandas as pd

from quant_trading.connectors.gate_sync import GateSync
from quant_trading.signal.probability_engine import ProbabilityEngine, ProbabilitySignal


@dataclass
class TradeConfig:
    """交易配置"""
    symbol: str = "ETH/USDT"
    leverage: int = 10           # 杠杆倍数
    position_size: float = 1.0   # 仓位大小 (ETH) - Gate.io最小1ETH

    # 信号阈值
    strong_buy_threshold: float = 0.60  # 强买信号阈值
    strong_sell_threshold: float = 0.60 # 强卖信号阈值

    # 止损止盈 (按保证金百分比)
    stop_loss_margin_pct: float = 0.10   # 止损 10%保证金
    take_profit_margin_pct: float = 0.20 # 止盈 20%保证金

    # 风控
    max_position: int = 1         # 最大持仓数
    max_daily_trades: int = 5    # 每日最大交易次数


@dataclass
class TradeState:
    """交易状态"""
    has_position: bool = False
    position_side: str = ""       # "long" or "short"
    entry_price: float = 0.0
    position_size: float = 0.0
    entry_time: datetime = None

    # 保证金相关
    margin: float = 0.0          # 保证金
    stop_loss_price: float = 0.0  # 止损价格
    take_profit_price: float = 0.0 # 止盈价格

    # 盈亏
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # 统计
    trade_count: int = 0
    last_trade_time: datetime = None

    # 信号
    last_signal: str = ""
    last_confidence: float = 0.0


class AutoTrader:
    """
    自动跟单交易器

    根据概率引擎信号自动执行交易
    """

    def __init__(
        self,
        config: Optional[TradeConfig] = None,
        symbol: str = "ETH",
        leverage: int = 10
    ):
        normalized_symbol = symbol if "/" in symbol else f"{symbol}/USDT"
        self.config = config or TradeConfig(symbol=normalized_symbol, leverage=leverage)
        self.symbol = self.config.symbol.replace("/", "_").upper()

        # 初始化组件
        self.gate = GateSync(require_auth=True)
        self.engine = ProbabilityEngine(symbol=self.config.symbol)

        # 交易状态
        self.state = TradeState()

        # 运行状态
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # 统计
        self.stats = {
            'start_time': datetime.now(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
        }

        # 交易记录
        self.trade_log: List[Dict] = []

    def start(self, interval: int = 60):
        """
        启动自动交易

        Args:
            interval: 检查间隔（秒）
        """
        if self.running:
            print("自动交易器已在运行中")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, args=(interval,), daemon=True)
        self.thread.start()

        print(f"自动交易器已启动 | 交易对: {self.config.symbol} | 杠杆: {self.config.leverage}x")
        print(
            f"检查间隔: {interval}秒 | 止损: {self.config.stop_loss_margin_pct*100}% | "
            f"止盈: {self.config.take_profit_margin_pct*100}%"
        )

    def stop(self):
        """停止自动交易"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("自动交易器已停止")

    def _run_loop(self, interval: int):
        """运行循环"""
        while self.running:
            try:
                self._check_and_trade()
            except Exception as e:
                print(f"交易循环错误: {e}")

            time.sleep(interval)

    def _format_position_snapshot(self) -> str:
        """格式化后台可见的持仓快照"""
        if not self.state.has_position:
            return "无持仓"

        return (
            f"持仓: {self.state.position_side}"
            f" | 数量: {self.state.position_size:.4f}"
            f" | 开仓价: ${self.state.entry_price:.2f}"
            f" | 保证金: ${self.state.margin:.2f}"
            f" | 盈亏: ${self.state.unrealized_pnl:.2f}"
            f" | 止损: ${self.state.stop_loss_price:.2f}"
            f" | 止盈: ${self.state.take_profit_price:.2f}"
        )

    def _check_and_trade(self):
        """检查信号并交易"""
        # 获取当前价格
        current_price = self.gate.price(self.symbol.replace("_USDT", ""))

        # 获取信号
        result = self._get_signal()
        signal = result.signal
        confidence = result.confidence

        # 更新状态
        self.state.last_signal = signal.value
        self.state.last_confidence = confidence

        # 检查持仓状态
        self._update_position_state(current_price)

        pos_info = ""
        if self.state.has_position:
            pos_info = f" | {self._format_position_snapshot()}"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 信号: {signal.value} ({confidence*100:.1f}%) | 价格: ${current_price:.2f}{pos_info}")

        # 决策逻辑
        if not self.state.has_position:
            # 无持仓，检查是否该入场
            self._check_entry(signal, confidence, current_price)
        else:
            # 有持仓，检查是否该出场
            self._check_exit(current_price)

    def _get_signal(self):
        """获取交易信号"""
        ohlcv_data = self.gate.ohlcv(
            self.symbol.replace("_USDT", ""),
            timeframe='1h',
            limit=100
        )

        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return self.engine.analyze(df)

    def _check_entry(
        self,
        signal: ProbabilitySignal,
        confidence: float,
        current_price: float
    ):
        """检查是否该入场"""
        # 买入条件
        if signal == ProbabilitySignal.STRONG_BUY and confidence >= self.config.strong_buy_threshold:
            self._open_long(current_price)

        # 卖出条件
        elif signal == ProbabilitySignal.STRONG_SELL and confidence >= self.config.strong_sell_threshold:
            self._open_short(current_price)

    def _check_exit(self, current_price: float):
        """检查是否该出场 - 基于价格判断"""
        if not self.state.has_position:
            return

        side = self.state.position_side
        entry = self.state.entry_price
        stop = self.state.stop_loss_price
        tp = self.state.take_profit_price
        size = self.state.position_size

        # 计算未实现盈亏
        if side == "long":
            pnl = (current_price - entry) * size
            # 止损: 价格跌到止损价
            if current_price <= stop:
                self._close_position(f"止损 @ ${current_price:.2f} (亏${abs(pnl):.2f})")
                return
            # 止盈: 价格涨到止盈价
            if current_price >= tp:
                self._close_position(f"止盈 @ ${current_price:.2f} (赚${pnl:.2f})")
                return
        else:  # short
            pnl = (entry - current_price) * size
            # 止损: 价格涨到止损价
            if current_price >= stop:
                self._close_position(f"止损 @ ${current_price:.2f} (亏${abs(pnl):.2f})")
                return
            # 止盈: 价格跌到止盈价
            if current_price <= tp:
                self._close_position(f"止盈 @ ${current_price:.2f} (赚${pnl:.2f})")
                return


    def _safe_float(self, value: Any) -> Optional[float]:
        """安全转换为浮点数"""
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if result <= 0:
            return None
        return result

    def _resolve_entry_price(self, order: Dict[str, Any], fallback_price: float) -> float:
        """解析成交价，优先使用真实成交均价"""
        entry_price = self._safe_float(order.get('average'))
        if entry_price is not None:
            return entry_price

        entry_price = self._safe_float(order.get('price'))
        if entry_price is not None:
            return entry_price

        return fallback_price

    def _resolve_filled_amount(self, order: Dict[str, Any], fallback_amount: float) -> float:
        """解析实际成交数量，缺少可信成交量时返回 0 触发后续实盘对账"""
        filled = self._safe_float(order.get('filled'))
        if filled is not None:
            return filled
        return 0.0

    def _get_live_position_snapshot(self, symbol: str) -> Optional[Dict[str, float]]:
        """获取交易所真实仓位快照（按方向聚合）"""
        try:
            positions = self.gate.positions(symbol)
        except Exception as position_error:
            print(f"查询真实仓位失败: {position_error}")
            return None

        long_size = sum(position.size for position in positions if position.size > 0)
        short_size = sum(-position.size for position in positions if position.size < 0)
        return {
            'long': long_size,
            'short': short_size,
        }

    def _get_live_position_size(self, symbol: str, position_side: Optional[str] = None) -> Optional[float]:
        """获取与目标方向一致的真实持仓数量"""
        target_side = position_side or self.state.position_side
        if target_side not in ("long", "short"):
            return None

        snapshot = self._get_live_position_snapshot(symbol)
        if snapshot is None:
            return None
        return snapshot[target_side]

    def _has_live_position(self, symbol: str, position_side: Optional[str] = None) -> bool:
        """检查交易所是否已有目标方向的真实仓位"""
        live_size = self._get_live_position_size(symbol, position_side=position_side)
        return live_size is not None and live_size > 0

    def _sync_position_closed(self, symbol: str) -> bool:
        """确认真实仓位是否已平，并同步剩余仓位数量"""
        side = self.state.position_side
        if side not in ("long", "short"):
            print("平仓后缺少本地方向信息，保留本地持仓状态")
            return False

        snapshot = self._get_live_position_snapshot(symbol)
        if snapshot is None:
            print("平仓后无法确认真实仓位，保留本地持仓状态")
            return False

        same_side_live_size = snapshot[side]
        opposite_side = "short" if side == "long" else "long"
        opposite_side_live_size = snapshot[opposite_side]

        if same_side_live_size > 0:
            if abs(same_side_live_size - self.state.position_size) > 1e-9:
                self.state.position_size = same_side_live_size
                self.state.margin = self.state.entry_price * same_side_live_size / self.config.leverage
                print(f"平仓后检测到交易所仍有剩余仓位，已同步本地仓位数量: {same_side_live_size}")
            else:
                print("平仓后检测到交易所仍有仓位，保留本地持仓状态")
            return False

        if opposite_side_live_size > 0:
            print(f"平仓后检测到交易所仍有反向真实仓位，保留本地持仓状态: {opposite_side} {opposite_side_live_size}")
            return False

        return True

    def _reset_local_position_state(self):
        """重置本地持仓状态"""
        self.state.has_position = False
        self.state.position_side = ""
        self.state.entry_price = 0.0
        self.state.position_size = 0.0
        self.state.margin = 0.0
        self.state.stop_loss_price = 0.0
        self.state.take_profit_price = 0.0
        self.state.unrealized_pnl = 0.0

    def _open_long(self, price: float):
        """开多"""
        try:
            symbol = self.symbol.replace("_USDT", "")
            amount = self.config.position_size

            print(f"开多 | 价格: ${price:.2f} | 数量: {amount} | 杠杆: {self.config.leverage}x")

            pre_live_size = self._get_live_position_size(symbol, position_side="long")
            order = self.gate.buy_market(symbol, amount, self.config.leverage)
            filled_amount = self._resolve_filled_amount(order, amount)
            if filled_amount <= 0:
                post_live_size = self._get_live_position_size(symbol, position_side="long")
                if pre_live_size is None or post_live_size is None:
                    raise RuntimeError(f"开多后未确认成交: {order}")
                filled_amount = max(0.0, post_live_size - pre_live_size)
                if filled_amount <= 0:
                    raise RuntimeError(f"开多后未确认成交: {order}")

            entry_price = self._resolve_entry_price(order, price)

            # 计算保证金和止损止盈价格
            position_value = entry_price * filled_amount
            margin = position_value / self.config.leverage
            # 做多止损: 价格下跌 (跌到 保证金损失stop_loss_margin_p% 时触发)
            stop_loss = entry_price * (1 - self.config.stop_loss_margin_pct / self.config.leverage)
            # 做多止盈: 价格上涨
            take_profit = entry_price * (1 + self.config.take_profit_margin_pct / self.config.leverage)

            # 先更新本地状态，避免保护单失败时与真实仓位分叉
            self.state.has_position = True
            self.state.position_side = "long"
            self.state.entry_price = entry_price
            self.state.position_size = filled_amount
            self.state.margin = margin
            self.state.stop_loss_price = stop_loss
            self.state.take_profit_price = take_profit
            self.state.entry_time = datetime.now()
            self.state.trade_count += 1

            protection_ok = True
            try:
                self.gate.set_stop_loss(symbol, stop_loss)
                self.gate.set_take_profit(symbol, take_profit)
            except Exception as protection_error:
                protection_ok = False
                print(f"开多保护单设置失败: {protection_error}")

            self._log_trade("OPEN_LONG", entry_price, filled_amount, order.get('id'))

            if protection_ok:
                print(f"开多成功 | 保证金: ${margin:.2f} | 止损: ${stop_loss:.2f} | 止盈: ${take_profit:.2f}")
            else:
                print(f"开多成功但保护单失败 | 保证金: ${margin:.2f} | 入场: ${entry_price:.2f}")

        except Exception as e:
            print(f"开多失败: {e}")

    def _open_short(self, price: float):
        """开空"""
        try:
            symbol = self.symbol.replace("_USDT", "")
            amount = self.config.position_size

            print(f"开空 | 价格: ${price:.2f} | 数量: {amount} | 杠杆: {self.config.leverage}x")

            pre_live_size = self._get_live_position_size(symbol, position_side="short")
            order = self.gate.sell_market(symbol, amount, self.config.leverage)
            filled_amount = self._resolve_filled_amount(order, amount)
            if filled_amount <= 0:
                post_live_size = self._get_live_position_size(symbol, position_side="short")
                if pre_live_size is None or post_live_size is None:
                    raise RuntimeError(f"开空后未确认成交: {order}")
                filled_amount = max(0.0, post_live_size - pre_live_size)
                if filled_amount <= 0:
                    raise RuntimeError(f"开空后未确认成交: {order}")

            entry_price = self._resolve_entry_price(order, price)

            # 计算保证金和止损止盈价格
            position_value = entry_price * filled_amount
            margin = position_value / self.config.leverage
            # 做空止损: 价格上涨 (涨到 保证金损失stop_loss_margin_p% 时触发)
            stop_loss = entry_price * (1 + self.config.stop_loss_margin_pct / self.config.leverage)
            # 做空止盈: 价格下跌
            take_profit = entry_price * (1 - self.config.take_profit_margin_pct / self.config.leverage)

            # 先更新本地状态，避免保护单失败时与真实仓位分叉
            self.state.has_position = True
            self.state.position_side = "short"
            self.state.entry_price = entry_price
            self.state.position_size = filled_amount
            self.state.margin = margin
            self.state.stop_loss_price = stop_loss
            self.state.take_profit_price = take_profit
            self.state.entry_time = datetime.now()
            self.state.trade_count += 1

            protection_ok = True
            try:
                self.gate.set_stop_loss(symbol, stop_loss)
                self.gate.set_take_profit(symbol, take_profit)
            except Exception as protection_error:
                protection_ok = False
                print(f"开空保护单设置失败: {protection_error}")

            self._log_trade("OPEN_SHORT", entry_price, filled_amount, order.get('id'))

            if protection_ok:
                print(f"开空成功 | 保证金: ${margin:.2f} | 止损: ${stop_loss:.2f} | 止盈: ${take_profit:.2f}")
            else:
                print(f"开空成功但保护单失败 | 保证金: ${margin:.2f} | 入场: ${entry_price:.2f}")

        except Exception as e:
            print(f"开空失败: {e}")

    def _close_position(self, reason: str = ""):
        """平仓（支持 dual mode）"""
        if not self.state.has_position:
            return

        try:
            symbol = self.symbol.replace("_USDT", "")
            amount = self.state.position_size
            side = self.state.position_side

            print(f"平仓 | 方向: {side} | 原因: {reason}")

            # Dual mode 平仓规则：
            # - 平多仓(dual_long) → 用 SELL + reduce_only
            # - 平空仓(dual_short) → 用 BUY + reduce_only
            if side == "long":
                self.gate.sell_market(symbol, amount, self.config.leverage, reduce_only=True)
            else:
                self.gate.buy_market(symbol, amount, self.config.leverage, reduce_only=True)

            if not self._sync_position_closed(symbol):
                return

            # 清理该仓位对应的旧保护单，避免残留触发单污染下一笔仓位
            try:
                self.gate.cancel_close_trigger_orders(symbol, position_side=side)
            except Exception as cleanup_error:
                print(f"清理保护单失败: {cleanup_error}")

            # 计算盈亏
            current_price = self.gate.price(symbol)
            pnl = (current_price - self.state.entry_price) * amount
            if side == "short":
                pnl = -pnl

            # 更新统计
            self.stats['total_trades'] += 1
            if pnl > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            self.stats['total_pnl'] += pnl

            self._log_trade("CLOSE", current_price, amount, reason=reason)

            self._reset_local_position_state()

            print(f"平仓完成 | 盈亏: ${pnl:.2f}")

        except Exception as e:
            print(f"平仓失败: {e}")

    def _update_position_state(self, current_price: float):
        """更新持仓状态"""
        if not self.state.has_position:
            return

        entry = self.state.entry_price
        size = self.state.position_size
        side = self.state.position_side

        if side == "long":
            pnl = (current_price - entry) * size
        else:
            pnl = (entry - current_price) * size

        self.state.unrealized_pnl = pnl

    def _log_trade(
        self,
        action: str,
        price: float,
        amount: float,
        order_id: str = None,
        reason: str = None
    ):
        """记录交易"""
        self.trade_log.append({
            'time': datetime.now().isoformat(),
            'action': action,
            'symbol': self.config.symbol,
            'price': price,
            'amount': amount,
            'order_id': order_id,
            'reason': reason,
            'stats': self.stats.copy()
        })

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'running': self.running,
            'has_position': self.state.has_position,
            'position_side': self.state.position_side,
            'entry_price': self.state.entry_price,
            'position_size': self.state.position_size,
            'margin': self.state.margin,
            'stop_loss_price': self.state.stop_loss_price,
            'take_profit_price': self.state.take_profit_price,
            'unrealized_pnl': self.state.unrealized_pnl,
            'realized_pnl': self.state.realized_pnl,
            'trade_count': self.state.trade_count,
            'last_signal': self.state.last_signal,
            'last_confidence': self.state.last_confidence,
            'stats': self.stats
        }

    def run_once(self) -> Dict[str, Any]:
        """单次执行（用于测试）"""
        self._check_and_trade()
        return self.get_status()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='自动跟单交易系统')
    parser.add_argument('--symbol', default='ETH', help='交易对')
    parser.add_argument('--leverage', type=int, default=10, help='杠杆倍数')
    parser.add_argument('--position-size', type=float, default=0.1, help='仓位大小')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔（秒）')
    parser.add_argument('--stop-loss', type=float, default=0.03, help='止损百分比')
    parser.add_argument('--take-profit', type=float, default=0.06, help='止盈百分比')
    parser.add_argument('--once', action='store_true', help='单次执行')

    args = parser.parse_args()

    # 创建配置
    config = TradeConfig(
        symbol=args.symbol if '/' in args.symbol else f"{args.symbol}/USDT",
        leverage=args.leverage,
        position_size=args.position_size,
        stop_loss_margin_pct=args.stop_loss,
        take_profit_margin_pct=args.take_profit,
    )

    # 创建交易器
    trader = AutoTrader(config=config)

    if args.once:
        # 单次执行
        print("执行单次信号检查...")
        status = trader.run_once()
        print(f"\n状态: {json.dumps(status, indent=2, default=str)}")
    else:
        # 启动自动交易
        print("启动自动交易...")
        try:
            trader.start(interval=args.interval)
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n停止中...")
            trader.stop()


if __name__ == '__main__':
    main()
