"""
Risk-Controlled Trading Wrapper - Unified Trading with Risk Management

风险控制交易包装器 - 统一交易与风险管理

Wraps BinanceTradingAdapter with UnifiedRiskManager for:
- Pre-trade risk checks before order placement
- Position size validation
- Daily loss limits
- Post-trade risk monitoring and updates
- Automatic trading halt on risk breaches

Usage:
    >>> from quant_trading.execution.risk_controlled_trading import RiskControlledTrading
    >>> from quant_trading.risk.manager import RiskManager, RiskConfig
    >>> risk_manager = RiskManager(RiskConfig(max_position_size=1000))
    >>> trading = RiskControlledTrading(risk_manager=risk_manager)
    >>> order = await trading.place_order("BTCUSDT", "BUY", 0.001, 50000)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from quant_trading.connectors.binance_trading import (
    BinanceTradingAdapter,
    OrderSide,
    OrderType,
    TradingOrder,
    TradingMode,
    AccountBalance,
)
from quant_trading.execution.binance_order_manager import (
    BinanceOrderManager,
    TrackedOrder,
    OrderState,
)
from quant_trading.risk.manager import RiskManager, RiskConfig, RiskMetrics, RiskLevel


class RiskBreachError(Exception):
    """风险违规"""
    pass


class TradingHaltedError(Exception):
    """交易暂停"""
    pass


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: float
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    pnl: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW


class RiskControlledTrading:
    """
    风险控制交易包装器

    将交易适配器与风控管理器集成，提供：
    - 下单前风控检查
    - 仓位大小验证
    - 日亏损限制
    - 交易后风控更新
    - 风险违规自动暂停交易

    Args:
        adapter: BinanceTradingAdapter实例
        risk_manager: RiskManager实例
        order_manager: BinanceOrderManager实例（可选，将自动创建）
        enable_auto_halt: 风险违规时自动暂停交易
    """

    def __init__(
        self,
        adapter: Optional[BinanceTradingAdapter] = None,
        risk_manager: Optional[RiskManager] = None,
        order_manager: Optional[BinanceOrderManager] = None,
        enable_auto_halt: bool = True,
        max_slippage_pct: float = 0.01,  # 1% max slippage
        default_timeout: float = 60.0,
    ):
        # Adapter
        self._adapter = adapter or BinanceTradingAdapter()

        # Risk manager
        self._risk_manager = risk_manager or RiskManager()

        # Order manager
        self._order_manager = order_manager

        # Settings
        self._enable_auto_halt = enable_auto_halt
        self._max_slippage_pct = max_slippage_pct
        self._default_timeout = default_timeout

        # State
        self._halted = False
        self._halt_reason: Optional[str] = None
        self._trade_history: List[TradeRecord] = []

        # Logger
        self._logger = logging.getLogger("RiskControlledTrading")

        # Callbacks
        self._on_risk_breach_callback: Optional[Callable[[str, Dict], None]] = None
        self._on_trade_callback: Optional[Callable[[TradeRecord], None]] = None

    @property
    def adapter(self) -> BinanceTradingAdapter:
        """交易适配器"""
        return self._adapter

    @property
    def risk_manager(self) -> RiskManager:
        """风控管理器"""
        return self._risk_manager

    @property
    def order_manager(self) -> BinanceOrderManager:
        """订单管理器"""
        if self._order_manager is None:
            self._order_manager = BinanceOrderManager(self._adapter)
        return self._order_manager

    @property
    def is_halted(self) -> bool:
        """是否暂停交易"""
        return self._halted

    @property
    def halt_reason(self) -> Optional[str]:
        """暂停原因"""
        return self._halt_reason

    @property
    def mode(self) -> TradingMode:
        """交易模式"""
        return self._adapter.mode

    # ====================
    # Lifecycle / 生命周期
    # ====================

    async def connect(self) -> None:
        """连接"""
        await self._adapter.connect()

        if self._order_manager:
            await self._order_manager.start()

    async def disconnect(self) -> None:
        """断开连接"""
        if self._order_manager:
            await self._order_manager.stop()

        await self._adapter.disconnect()

    async def start(self) -> None:
        """启动"""
        await self.connect()

    async def stop(self) -> None:
        """停止"""
        await self.disconnect()

    # ====================
    # Pre-Trade Risk Checks / 下单前风控检查
    # ====================

    async def _pre_trade_check(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        下单前风控检查

        Returns:
            Dict with 'allowed' (bool) and 'reason' (str)
        """
        # Check if halted
        if self._halted:
            return {
                "allowed": False,
                "reason": f"Trading halted: {self._halt_reason}",
            }

        # Get current price if not provided
        if price is None:
            try:
                price = await self._adapter.get_ticker_price(symbol)
            except Exception as e:
                return {
                    "allowed": False,
                    "reason": f"Failed to get price: {e}",
                }

        order_value = quantity * price

        # Check position size
        size_check = self._risk_manager.check_position_size(symbol, quantity, price)
        if not size_check["allowed"]:
            self._log_risk_breach("position_size", symbol, size_check)
            return size_check

        # Check trade allowed
        trade_check = self._risk_manager.check_trade_allowed(symbol, order_value * 0.02)  # Assume 2% risk
        if not trade_check["allowed"]:
            self._log_risk_breach("trade_allowed", symbol, trade_check)
            return trade_check

        # Check balance for BUY orders
        if side == OrderSide.BUY:
            quote_asset = symbol.replace("USDT", "").replace("BTC", "").replace("ETH", "")
            balance = await self._adapter.get_balance(quote_asset or "USDT")
            required = order_value * 1.001  # 0.1% buffer

            if balance and balance.free < required:
                result = {
                    "allowed": False,
                    "reason": f"Insufficient {quote_asset or 'USDT'} balance: "
                             f"required {required:.2f}, available {balance.free if balance else 0:.2f}",
                }
                self._log_risk_breach("insufficient_balance", symbol, result)
                return result

        # Check daily loss limit impact
        if self._risk_manager.metrics.daily_pnl < -self._risk_manager.config.max_daily_loss * 0.8:
            result = {
                "allowed": False,
                "reason": f"Daily loss approaching limit "
                         f"(current: {abs(self._risk_manager.metrics.daily_pnl):.2f}, "
                         f"limit: {self._risk_manager.config.max_daily_loss:.2f})",
            }
            self._log_risk_breach("daily_loss_approaching", symbol, result)
            return result

        return {"allowed": True}

    def _log_risk_breach(self, breach_type: str, symbol: str, check_result: Dict) -> None:
        """记录风险违规"""
        self._logger.warning(
            f"Risk breach [{breach_type}] on {symbol}: {check_result['reason']}"
        )

        if self._on_risk_breach_callback:
            self._on_risk_breach_callback(breach_type, check_result)

        if self._enable_auto_halt:
            self._halted = True
            self._halt_reason = f"{breach_type}: {check_result['reason']}"
            self._logger.critical(f"Trading halted: {self._halt_reason}")

    # ====================
    # Post-Trade Updates / 交易后风控更新
    # ====================

    async def _post_trade_update(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> None:
        """更新风控状态"""
        # Calculate P&L
        pnl = 0.0
        if side == OrderSide.SELL:
            # For sells, calculate realized P&L
            pnl = quantity * price * 0.001  # Rough estimate
        else:
            # For buys, update exposure
            self._risk_manager.metrics.current_exposure += quantity * price

        # Update metrics
        self._risk_manager.metrics.daily_pnl += pnl
        self._risk_manager.metrics.daily_trades += 1
        self._risk_manager._last_trade_time = time.time() * 1000

        # Log trade
        record = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            risk_level=self._risk_manager.metrics.current_risk_level,
        )
        self._trade_history.append(record)

        # Log to risk manager
        self._risk_manager.trade_log.append(record.__dict__)

        self._logger.info(
            f"Trade recorded: {side.value} {quantity} {symbol} @ {price}, PnL: {pnl:.2f}"
        )

        # Trigger callback
        if self._on_trade_callback:
            self._on_trade_callback(record)

    # ====================
    # Order Placement / 下单
    # ====================

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        skip_risk_check: bool = False,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """
        风控下单

        Args:
            symbol: 交易对
            side: 方向
            order_type: 订单类型
            quantity: 数量
            price: 价格
            stop_price: 止损价格
            client_order_id: 客户端订单ID
            skip_risk_check: 跳过风控检查（不推荐）
            wait_for_fill: 等待成交
            timeout: 超时时间

        Returns:
            TrackedOrder对象

        Raises:
            RiskBreachError: 风控检查失败
            TradingHaltedError: 交易已暂停
        """
        # Pre-trade risk check
        if not skip_risk_check:
            check_result = await self._pre_trade_check(symbol, side, quantity, price)
            if not check_result["allowed"]:
                raise RiskBreachError(check_result["reason"])

        # Get price for risk update if not provided
        current_price = price
        if current_price is None and order_type != OrderType.MARKET:
            current_price = await self._adapter.get_ticker_price(symbol)

        # Place order
        tracked = await self.order_manager.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            client_order_id=client_order_id,
            wait_for_fill=wait_for_fill,
            timeout=timeout or self._default_timeout,
        )

        # Post-trade update (on successful submission)
        if tracked.state in [OrderState.SUBMITTED, OrderState.PARTIAL, OrderState.FILLED]:
            fill_price = tracked.avg_fill_price if tracked.avg_fill_price > 0 else (current_price or 0)
            await self._post_trade_update(symbol, side, quantity, fill_price)

        return tracked

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        skip_risk_check: bool = False,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """市价单"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            skip_risk_check=skip_risk_check,
            wait_for_fill=wait_for_fill,
            timeout=timeout,
        )

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        skip_risk_check: bool = False,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """限价单"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            skip_risk_check=skip_risk_check,
            wait_for_fill=wait_for_fill,
            timeout=timeout,
        )

    # ====================
    # Cancel Operations / 取消操作
    # ====================

    async def cancel_order(self, client_order_id: str) -> bool:
        """取消订单"""
        return await self.order_manager.cancel_order(client_order_id)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """取消所有订单"""
        return await self.order_manager.cancel_all_orders(symbol)

    # ====================
    # Account Info / 账户信息
    # ====================

    async def get_balance(self, asset: str) -> Optional[AccountBalance]:
        """获取余额"""
        return await self._adapter.get_balance(asset)

    async def get_all_balances(self) -> List[AccountBalance]:
        """获取所有余额"""
        return await self._adapter.get_account_balance()

    # ====================
    # Risk Management / 风控管理
    # ====================

    def reset_halt(self, reason: Optional[str] = None) -> None:
        """
        重置交易暂停状态

        Args:
            reason: 重置原因（可选）
        """
        if reason:
            self._logger.info(f"Trading halt reset: {reason}")
        else:
            self._logger.info("Trading halt reset")
        self._halted = False
        self._halt_reason = None

    def update_risk_config(self, config: RiskConfig) -> None:
        """更新风控配置"""
        self._risk_manager.config = config
        self._logger.info(f"Risk config updated: {config}")

    def get_risk_metrics(self) -> RiskMetrics:
        """获取风控指标"""
        return self._risk_manager.metrics

    def get_trade_history(self, limit: int = 100) -> List[TradeRecord]:
        """获取交易历史"""
        return self._trade_history[-limit:]

    def set_risk_breach_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """设置风险违规回调"""
        self._on_risk_breach_callback = callback

    def set_trade_callback(self, callback: Callable[[TradeRecord], None]) -> None:
        """设置交易回调"""
        self._on_trade_callback = callback

    # ====================
    # Status / 状态
    # ====================

    def get_status(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "mode": self.mode.value,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "risk_level": self._risk_manager.metrics.current_risk_level.value,
            "daily_pnl": self._risk_manager.metrics.daily_pnl,
            "daily_trades": self._risk_manager.metrics.daily_trades,
            "current_exposure": self._risk_manager.metrics.current_exposure,
            "total_trades": len(self._trade_history),
        }


# ====================
# Factory Functions / 工厂函数
# ====================

def create_live_trading(
    api_key: str,
    api_secret: str,
    risk_config: Optional[RiskConfig] = None,
) -> RiskControlledTrading:
    """
    创建实盘风控交易实例

    Args:
        api_key: Binance API密钥
        api_secret: Binance API密钥密码
        risk_config: 风控配置

    Returns:
        RiskControlledTrading实例
    """
    adapter = BinanceTradingAdapter(api_key=api_key, api_secret=api_secret, testnet=False)
    risk_manager = RiskManager(risk_config or RiskConfig())

    return RiskControlledTrading(
        adapter=adapter,
        risk_manager=risk_manager,
        enable_auto_halt=True,
    )


def create_paper_trading(
    risk_config: Optional[RiskConfig] = None,
) -> RiskControlledTrading:
    """
    创建模拟风控交易实例

    Args:
        risk_config: 风控配置

    Returns:
        RiskControlledTrading实例
    """
    adapter = BinanceTradingAdapter(testnet=True)
    risk_manager = RiskManager(risk_config or RiskConfig())

    return RiskControlledTrading(
        adapter=adapter,
        risk_manager=risk_manager,
        enable_auto_halt=True,
    )


__all__ = [
    "RiskControlledTrading",
    "RiskBreachError",
    "TradingHaltedError",
    "TradeRecord",
    "create_live_trading",
    "create_paper_trading",
]
