"""
Paper-to-Live Bridge - Seamless Trading Mode Switching

模拟盘到实盘桥接 - 无缝交易模式切换

Provides unified interface for switching between paper and live trading:
- Same API for both modes
- Position synchronization on startup
- Automatic mode detection
- Connection state management

Usage:
    >>> from quant_trading.execution.paper_to_live import PaperToLiveBridge
    >>> bridge = PaperToLiveBridge()
    >>>
    >>> # Initialize in paper mode
    >>> await bridge.init_paper()
    >>> order = await bridge.place_order("BTCUSDT", "BUY", 0.001, 50000)
    >>>
    >>> # Switch to live mode
    >>> await bridge.switch_to_live("your_api_key", "your_api_secret")
    >>> order = await bridge.place_order("BTCUSDT", "BUY", 0.001, 50000)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from quant_trading.connectors.binance_trading import (
    BinanceTradingAdapter,
    BinanceTradingAdapter,
    TradingMode,
    OrderSide,
    OrderType,
    AccountBalance,
)
from quant_trading.execution.binance_order_manager import (
    BinanceOrderManager,
    TrackedOrder,
    OrderState,
)
from quant_trading.execution.risk_controlled_trading import (
    RiskControlledTrading,
    RiskConfig,
)


class BridgeError(Exception):
    """桥接器错误"""
    pass


class ModeSwitchError(BridgeError):
    """模式切换错误"""
    pass


@dataclass
class PositionSnapshot:
    """仓位快照"""
    symbol: str
    quantity: float
    avg_price: float
    timestamp: float


class PaperToLiveBridge:
    """
    模拟盘到实盘桥接器

    提供统一的交易接口，支持无缝切换模拟盘和实盘：

    Features:
    - 统一的下单接口
    - 自动仓位同步
    - 模式切换平滑过渡
    - 状态监控

    Args:
        risk_config: 风控配置（默认使用宽松配置）
        default_timeout: 默认订单超时时间
    """

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        default_timeout: float = 60.0,
    ):
        # Core components (lazy initialized)
        self._adapter: Optional[BinanceTradingAdapter] = None
        self._order_manager: Optional[BinanceOrderManager] = None
        self._trading: Optional[RiskControlledTrading] = None

        # Configuration
        self._risk_config = risk_config or RiskConfig(
            max_position_size=10000.0,
            max_single_loss=1000.0,
            max_daily_loss=5000.0,
            max_daily_trades=100,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        self._default_timeout = default_timeout

        # State
        self._mode: TradingMode = TradingMode.DISCONNECTED
        self._live_api_key: Optional[str] = None
        self._live_api_secret: Optional[str] = None

        # Position sync
        self._synced_positions: Dict[str, PositionSnapshot] = {}
        self._position_sync_enabled = True

        # Logger
        self._logger = logging.getLogger("PaperToLiveBridge")

        # Callbacks
        self._on_mode_change: Optional[callable] = None
        self._on_position_sync: Optional[callable] = None

    @property
    def mode(self) -> TradingMode:
        """当前模式"""
        return self._mode

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._mode != TradingMode.DISCONNECTED

    @property
    def is_live(self) -> bool:
        """是否实盘模式"""
        return self._mode == TradingMode.LIVE

    @property
    def is_paper(self) -> bool:
        """是否模拟盘模式"""
        return self._mode == TradingMode.PAPER

    # ====================
    # Initialization / 初始化
    # ====================

    async def init_paper(self) -> None:
        """
        初始化模拟盘模式

        Example:
            >>> bridge = PaperToLiveBridge()
            >>> await bridge.init_paper()
        """
        if self._mode != TradingMode.DISCONNECTED:
            raise BridgeError(f"Already initialized in {self._mode.value} mode")

        self._logger.info("Initializing paper trading mode...")

        # Create adapter (no API keys needed for paper)
        self._adapter = BinanceTradingAdapter(
            api_key="",
            api_secret="",
            testnet=True,
        )

        # Create order manager
        self._order_manager = BinanceOrderManager(
            self._adapter,
            poll_interval=1.0,
            order_timeout=self._default_timeout,
        )

        # Create risk-controlled trading wrapper
        self._trading = RiskControlledTrading(
            adapter=self._adapter,
            risk_manager=None,  # Use default for paper
            order_manager=self._order_manager,
            enable_auto_halt=False,  # Disable for paper trading
        )

        # Connect
        await self._adapter.connect()
        await self._order_manager.start()

        self._mode = TradingMode.PAPER
        self._logger.info("Paper trading mode initialized")

        if self._on_mode_change:
            self._on_mode_change(self._mode)

    async def init_live(
        self,
        api_key: str,
        api_secret: str,
        sync_positions: bool = True,
    ) -> None:
        """
        初始化实盘模式

        Args:
            api_key: Binance API密钥
            api_secret: Binance API密钥密码
            sync_positions: 是否同步当前仓位

        Example:
            >>> bridge = PaperToLiveBridge()
            >>> await bridge.init_live("api_key", "api_secret")
        """
        if self._mode != TradingMode.DISCONNECTED:
            raise BridgeError(f"Already initialized in {self._mode.value} mode")

        self._logger.info("Initializing live trading mode...")

        # Store credentials
        self._live_api_key = api_key
        self._live_api_secret = api_secret

        # Create adapter
        self._adapter = BinanceTradingAdapter(
            api_key=api_key,
            api_secret=api_secret,
            testnet=False,
        )

        # Create order manager
        self._order_manager = BinanceOrderManager(
            self._adapter,
            poll_interval=1.0,
            order_timeout=self._default_timeout,
        )

        # Create risk-controlled trading wrapper
        self._trading = RiskControlledTrading(
            adapter=self._adapter,
            risk_manager=None,  # Will use adapter's risk manager
            order_manager=self._order_manager,
            enable_auto_halt=True,
        )

        # Connect
        await self._adapter.connect()
        await self._order_manager.start()

        # Sync positions if requested
        if sync_positions:
            await self._sync_positions_from_exchange()

        self._mode = TradingMode.LIVE
        self._logger.info("Live trading mode initialized")

        if self._on_mode_change:
            self._on_mode_change(self._mode)

    async def disconnect(self) -> None:
        """断开连接"""
        if self._mode == TradingMode.DISCONNECTED:
            return

        self._logger.info(f"Disconnecting from {self._mode.value} mode...")

        if self._order_manager:
            await self._order_manager.stop()

        if self._adapter:
            await self._adapter.disconnect()

        self._mode = TradingMode.DISCONNECTED
        self._logger.info("Disconnected")

        if self._on_mode_change:
            self._on_mode_change(self._mode)

    # ====================
    # Mode Switching / 模式切换
    # ====================

    async def switch_to_live(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sync_positions: bool = True,
    ) -> None:
        """
        切换到实盘模式

        Args:
            api_key: API密钥（如果已初始化可省略）
            api_secret: API密钥密码（如果已初始化可省略）
            sync_positions: 是否同步仓位

        Example:
            >>> bridge = PaperToLiveBridge()
            >>> await bridge.init_paper()
            >>> # ... trade in paper mode ...
            >>> await bridge.switch_to_live("api_key", "api_secret")
        """
        if self._mode == TradingMode.LIVE:
            self._logger.info("Already in live mode")
            return

        key = api_key or self._live_api_key
        secret = api_secret or self._live_api_secret

        if not key or not secret:
            raise ModeSwitchError("API key and secret required for live mode")

        # Capture paper positions if switching from paper
        if self._mode == TradingMode.PAPER:
            await self._capture_paper_positions()

        # Disconnect current
        await self.disconnect()

        # Initialize live
        await self.init_live(key, secret, sync_positions=sync_positions)

        # Sync with paper positions if any
        if self._synced_positions and sync_positions:
            await self._reconcile_positions()

        self._logger.info("Switched to live trading")

    async def switch_to_paper(self) -> None:
        """
        切换到模拟盘模式

        Example:
            >>> await bridge.switch_to_paper()
        """
        if self._mode == TradingMode.PAPER:
            self._logger.info("Already in paper mode")
            return

        if self._mode == TradingMode.LIVE:
            # Capture live positions before switching
            await self._capture_live_positions()

        # Disconnect
        await self.disconnect()

        # Initialize paper
        await self.init_paper()

        # Restore positions from live mode
        if self._synced_positions:
            await self._restore_positions()

        self._logger.info("Switched to paper trading")

    # ====================
    # Position Sync / 仓位同步
    # ====================

    async def _sync_positions_from_exchange(self) -> None:
        """从交易所同步仓位"""
        if not self._adapter:
            return

        self._logger.info("Syncing positions from exchange...")

        try:
            balances = await self._adapter.get_account_balance()

            for balance in balances:
                if balance.free > 0 or balance.locked > 0:
                    self._synced_positions[balance.asset] = PositionSnapshot(
                        symbol=balance.asset,
                        quantity=balance.free + balance.locked,
                        avg_price=0.0,  # Can't determine avg price from balance alone
                        timestamp=time.time(),
                    )

            if self._on_position_sync:
                self._on_position_sync(self._synced_positions)

            self._logger.info(f"Synced {len(self._synced_positions)} positions")

        except Exception as e:
            self._logger.error(f"Position sync failed: {e}")

    async def _capture_paper_positions(self) -> None:
        """捕获模拟盘仓位"""
        # In paper mode, we track positions through the trading wrapper
        # This is a placeholder for more sophisticated paper position tracking
        self._logger.info("Capturing paper positions for later sync")
        # Paper positions would be tracked internally by the trading system

    async def _capture_live_positions(self) -> None:
        """捕获实盘仓位"""
        if not self._adapter:
            return

        self._logger.info("Capturing live positions before mode switch...")

        try:
            balances = await self._adapter.get_account_balance()

            self._synced_positions.clear()
            for balance in balances:
                if balance.free > 0 or balance.locked > 0:
                    self._synced_positions[balance.asset] = PositionSnapshot(
                        symbol=balance.asset,
                        quantity=balance.free + balance.locked,
                        avg_price=0.0,
                        timestamp=time.time(),
                    )

            self._logger.info(f"Captured {len(self._synced_positions)} positions")

        except Exception as e:
            self._logger.error(f"Failed to capture live positions: {e}")

    async def _reconcile_positions(self) -> None:
        """对账仓位差异"""
        self._logger.info("Reconciling positions...")

        # In a real implementation, this would compare:
        # 1. Synced positions from previous mode
        # 2. Current positions on exchange
        # And alert on any discrepancies
        pass

    async def _restore_positions(self) -> None:
        """恢复仓位到新模式"""
        self._logger.info(f"Restoring {len(self._synced_positions)} positions...")

        # In paper mode, this would restore simulated positions
        # In live mode, positions are already synced from exchange
        pass

    def get_synced_positions(self) -> Dict[str, PositionSnapshot]:
        """获取同步的仓位"""
        return self._synced_positions.copy()

    # ====================
    # Trading Operations / 交易操作
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
        下单（统一接口）

        Args:
            symbol: 交易对（如 "BTCUSDT"）
            side: 订单方向（BUY/SELL）
            order_type: 订单类型
            quantity: 数量
            price: 价格（市价单可省略）
            stop_price: 止损价格
            client_order_id: 客户端订单ID
            skip_risk_check: 跳过风控检查
            wait_for_fill: 等待成交
            timeout: 超时时间

        Returns:
            TrackedOrder对象

        Example:
            >>> # 模拟/实盘使用相同接口
            >>> order = await bridge.place_order(
            ...     "BTCUSDT", OrderSide.BUY, OrderType.MARKET, 0.001
            ... )
        """
        if not self._trading:
            raise BridgeError("Not initialized. Call init_paper() or init_live() first.")

        return await self._trading.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            client_order_id=client_order_id,
            skip_risk_check=skip_risk_check,
            wait_for_fill=wait_for_fill,
            timeout=timeout or self._default_timeout,
        )

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """市价单"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            wait_for_fill=wait_for_fill,
            timeout=timeout,
        )

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
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
            wait_for_fill=wait_for_fill,
            timeout=timeout,
        )

    # ====================
    # Order Management / 订单管理
    # ====================

    async def cancel_order(self, client_order_id: str) -> bool:
        """取消订单"""
        if not self._order_manager:
            raise BridgeError("Not initialized")
        return await self._order_manager.cancel_order(client_order_id)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """取消所有订单"""
        if not self._order_manager:
            raise BridgeError("Not initialized")
        return await self._order_manager.cancel_all_orders(symbol)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[TrackedOrder]:
        """获取未成交订单"""
        if not self._order_manager:
            raise BridgeError("Not initialized")
        return await self._order_manager.get_open_orders(symbol)

    # ====================
    # Account Info / 账户信息
    # ====================

    async def get_balance(self, asset: str) -> Optional[AccountBalance]:
        """获取指定资产余额"""
        if not self._adapter:
            raise BridgeError("Not initialized")
        return await self._adapter.get_balance(asset)

    async def get_all_balances(self) -> List[AccountBalance]:
        """获取所有余额"""
        if not self._adapter:
            raise BridgeError("Not initialized")
        return await self._adapter.get_account_balance()

    # ====================
    # Status / 状态
    # ====================

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = {
            "mode": self._mode.value,
            "is_connected": self.is_connected,
            "positions_synced": len(self._synced_positions),
        }

        if self._trading:
            status.update(self._trading.get_status())

        return status

    def set_mode_change_callback(self, callback: callable) -> None:
        """设置模式切换回调"""
        self._on_mode_change = callback

    def set_position_sync_callback(self, callback: callable) -> None:
        """设置仓位同步回调"""
        self._on_position_sync = callback


# ====================
# Context Manager / 上下文管理器
# ====================

class TradingContext:
    """
    交易上下文管理器

    提供便捷的上下文管理器接口：

    Example:
        >>> bridge = PaperToLiveBridge()
        >>> async with TradingContext(bridge, mode="paper") as ctx:
        ...     order = await ctx.place_order("BTCUSDT", "BUY", 0.001, 50000)
    """

    def __init__(
        self,
        bridge: PaperToLiveBridge,
        mode: str = "paper",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self._bridge = bridge
        self._mode = mode
        self._api_key = api_key
        self._api_secret = api_secret

    async def __aenter__(self) -> "TradingContext":
        if self._mode == "paper":
            await self._bridge.init_paper()
        elif self._mode == "live":
            if not self._api_key or not self._api_secret:
                raise BridgeError("API key and secret required for live mode")
            await self._bridge.init_live(self._api_key, self._api_secret)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._bridge.disconnect()

    async def place_order(self, *args, **kwargs) -> TrackedOrder:
        return await self._bridge.place_order(*args, **kwargs)


__all__ = [
    "PaperToLiveBridge",
    "TradingContext",
    "BridgeError",
    "ModeSwitchError",
    "PositionSnapshot",
]
