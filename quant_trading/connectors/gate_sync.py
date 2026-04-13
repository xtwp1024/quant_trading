# -*- coding: utf-8 -*-
"""
Gate.io Synchronous Wrapper
Gate.io 同步封装

对 quant_trading/core/gate_adapter.py 的同步封装，
方便在同步环境中使用 Gate.io。

Usage:
    from quant_trading.connectors.gate_sync import GateSync

    gate = GateSync()
    price = gate.price('ETH')  # ETH 当前价格
    balance = gate.balance()   # 账户余额
    positions = gate.positions('ETH')  # 持仓
"""

import asyncio
import math
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from quant_trading.core.gate_adapter import GateExchangeAdapter


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    size: float
    entry_price: float
    liq_price: float
    pnl: float
    margin: float


class GateSync:
    """
    Gate.io 同步封装

    将异步的 GateExchangeAdapter 封装为同步调用，
    方便在普通脚本中使用。
    """

    def __init__(self, require_auth: bool = False):
        self.api_key = os.environ.get('GATE_API_KEY', '')
        self.api_secret = os.environ.get('GATE_API_SECRET', '')
        self.require_auth = require_auth

        config = {
            'exchange': {
                'name': 'gate',
                'gate': {
                    'api_key': self.api_key,
                    'secret': self.api_secret,
                }
            }
        }

        self._adapter = GateExchangeAdapter(config)
        self._closed = False
        self._initialized = False
        # 创建持久事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _ensure_init(self):
        """确保适配器已初始化"""
        if not self._initialized:
            initialized = self._loop.run_until_complete(self._adapter.initialize())
            if not initialized:
                raise RuntimeError('GateExchangeAdapter initialization failed')
            if self.require_auth and not self._adapter.is_authenticated:
                raise RuntimeError('Gate authentication required but adapter is not authenticated')
            self._initialized = True

    def _require_authenticated(self, operation: str):
        """确保私有交易操作只能在认证成功后执行"""
        self._ensure_init()
        if not self._adapter.is_authenticated:
            raise RuntimeError(f'Gate authentication required for {operation}')

    def _run(self, coro):
        """运行异步操作"""
        if self._closed:
            raise RuntimeError('GateSync has been closed')
        self._ensure_init()
        return self._loop.run_until_complete(coro)

    def close(self):
        """关闭连接"""
        if not self._closed:
            try:
                self._loop.run_until_complete(self._adapter.close())
            except:
                pass
            self._closed = True

    def price(self, symbol: str = 'ETH') -> float:
        """获取当前价格"""
        ticker = self._run(self._adapter.get_ticker(self._normalize_symbol(symbol)))
        return ticker.get('last', 0)

    def ohlcv(
        self,
        symbol: str = 'ETH',
        timeframe: str = '15m',
        limit: int = 100
    ) -> List[List]:
        """获取K线数据"""
        return self._run(self._adapter.get_ohlcv(self._normalize_symbol(symbol), timeframe, limit))

    def balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        self._require_authenticated('fetch balance')
        return self._run(self._adapter.get_balance())

    def positions(self, symbol: str = 'ETH') -> List[Position]:
        """获取持仓"""
        raw = self._run(self._adapter.get_position(self._normalize_symbol(symbol)))
        return [
            Position(
                symbol=p.get('symbol', symbol),
                size=self._position_contracts(p),
                entry_price=float(p.get('entry_price', 0)),
                liq_price=float(p.get('liq_price', 0)),
                pnl=float(p.get('unrealized_pnl', 0)),
                margin=float(p.get('margin', 0)),
            )
            for p in raw
        ]

    def _position_contracts(self, position: Dict[str, Any]) -> float:
        """获取标准化后的持仓合约数量"""
        contracts = position.get('contracts')
        info = position.get('info', {})
        raw_size = info.get('size')

        normalized_contracts = float(contracts or 0)
        normalized_raw_size = float(raw_size or 0)

        if normalized_raw_size != 0:
            return normalized_raw_size
        return normalized_contracts

    def _normalize_symbol(self, symbol: str) -> str:
        """统一转换为 GateSync 内部使用的交易对格式"""
        if ':' in symbol or '/' in symbol or '_' in symbol:
            return self._adapter._convert_symbol_to_gate(symbol)
        return f'{symbol}_USDT'

    def _validate_trigger_price(self, trigger_price: Optional[float], order_type: str) -> float:
        """校验保护单触发价格"""
        if trigger_price is None:
            raise ValueError(f'{order_type} trigger_price is required')

        normalized_trigger_price = float(trigger_price)
        if not math.isfinite(normalized_trigger_price):
            raise ValueError(f'{order_type} trigger_price must be finite')
        if normalized_trigger_price <= 0:
            raise ValueError(f'{order_type} trigger_price must be positive')

        return normalized_trigger_price

    def _validate_trigger_price_direction(
        self,
        trigger_price: float,
        current_price: float,
        rule: int,
        order_type: str,
    ) -> None:
        """校验保护单触发价格方向与当前价格一致"""
        normalized_current_price = float(current_price)
        if not math.isfinite(normalized_current_price):
            raise RuntimeError(f'Cannot validate {order_type} trigger_price against non-finite current price')
        if normalized_current_price <= 0:
            raise RuntimeError(f'Cannot validate {order_type} trigger_price against non-positive current price')

        if rule == 1 and trigger_price >= normalized_current_price:
            raise ValueError(
                f'{order_type} trigger_price must be below current price {normalized_current_price}'
            )
        if rule == 2 and trigger_price <= normalized_current_price:
            raise ValueError(
                f'{order_type} trigger_price must be above current price {normalized_current_price}'
            )

    def buy_market(
        self,
        symbol: str = 'ETH',
        amount: float = 1,
        leverage: int = 10,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        市价做多

        Args:
            symbol: 交易对
            amount: 数量
            leverage: 杠杆
            reduce_only: 是否为平仓（减少持仓），dual mode 平多仓时设为 True
        """
        normalized_symbol = self._normalize_symbol(symbol)
        self._run(self._adapter.set_leverage(leverage, normalized_symbol))
        params = {'reduce_only': True} if reduce_only else {}
        return self._run(self._adapter.create_order(
            symbol=normalized_symbol,
            side='buy',
            order_type='market',
            amount=amount,
            price=None,
            params=params,
        ))

    def sell_market(
        self,
        symbol: str = 'ETH',
        amount: float = 1,
        leverage: int = 10,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        市价做空

        Args:
            symbol: 交易对
            amount: 数量
            leverage: 杠杆
            reduce_only: 是否为平仓（减少持仓），dual mode 平空仓时设为 True
        """
        normalized_symbol = self._normalize_symbol(symbol)
        self._run(self._adapter.set_leverage(leverage, normalized_symbol))
        params = {'reduce_only': True} if reduce_only else {}
        return self._run(self._adapter.create_order(
            symbol=normalized_symbol,
            side='sell',
            order_type='market',
            amount=amount,
            price=None,
            params=params,
        ))

    def close_all(self, symbol: str = 'ETH') -> List[Dict]:
        """
        平所有仓位（支持 dual mode）

        Dual Mode 平仓规则：
        - dual_long（多仓）→ 用 SELL + reduce_only 平仓
        - dual_short（空仓）→ 用 BUY + reduce_only 平仓

        注意：dual mode 下不能用 close=True，必须用 reduce_only=True
        """
        self._require_authenticated('close positions')
        normalized_symbol = self._normalize_symbol(symbol)
        raw_positions = self._run(self._adapter.get_position(normalized_symbol))
        orders = []
        failures = []

        for p in raw_positions:
            info = p.get('info', {})
            size = self._position_contracts(p)
            mode = info.get('mode', '')

            if size == 0:
                continue

            # Dual mode 平仓：反向操作 + reduce_only
            # dual_long(多仓) → SELL 平仓
            # dual_short(空仓) → BUY 平仓
            if mode == 'dual_long':
                side = 'sell'
            elif mode == 'dual_short':
                side = 'buy'
            elif mode == 'single':
                side = 'sell' if size > 0 else 'buy'
            else:
                failures.append(f"[{mode}] {normalized_symbol}: unsupported position mode for safe close")
                continue

            try:
                order = self._run(self._adapter.create_order(
                    symbol=normalized_symbol,
                    side=side,
                    order_type='market',
                    amount=abs(size),
                    price=None,
                    params={'reduce_only': True},  # 必须用 reduce_only，不能用 close=True
                ))
                orders.append(order)
            except Exception as e:
                failures.append(f"[{mode}] {normalized_symbol}: {e}")

        if failures:
            raise RuntimeError("; ".join(failures))

        return orders

    def set_cross_margin_mode(self, symbol: str = 'ETH') -> Dict[str, Any]:
        """
        设置全仓模式（Cross Margin）

        Gate.io API: POST /futures/usdt/positions/{settle}/dual_mode
        dual_mode = False 表示单边持仓模式（可转换为全仓）

        注意：必须有持仓才能切换，需要先平所有仓位
        """
        self._require_authenticated('set cross margin mode')
        try:
            result = self._run(
                self._adapter.exchange.privateFuturesPostSettleDualMode(
                    {'settle': 'usdt', 'dual_mode': False}
                )
            )
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_position_mode(self, symbol: str = 'ETH') -> Dict[str, Any]:
        """
        获取当前持仓模式

        返回：dual_long, dual_short, 或 single (全仓模式)
        """
        raw_positions = self._run(self._adapter.get_position(self._normalize_symbol(symbol)))
        modes = set()
        unknown_modes = set()
        for p in raw_positions:
            info = p.get('info', {})
            size = self._position_contracts(p)
            mode = info.get('mode', '')
            if size != 0:
                if mode in ('dual_long', 'dual_short', 'single'):
                    modes.add(mode)
                else:
                    unknown_modes.add(mode or '<missing>')

        if unknown_modes:
            raise RuntimeError(
                f'Cannot infer position mode for {symbol}: unsupported mode values {sorted(unknown_modes)}'
            )

        if not modes:
            return {'mode': 'none', 'has_position': False}
        elif 'dual_long' in modes and 'dual_short' in modes:
            return {'mode': 'dual_both', 'has_position': True}
        elif 'dual_long' in modes:
            return {'mode': 'dual_long', 'has_position': True}
        elif 'dual_short' in modes:
            return {'mode': 'dual_short', 'has_position': True}
        elif modes == {'single'}:
            return {'mode': 'single', 'has_position': True}
        raise RuntimeError(f'Cannot infer position mode for {symbol}: inconsistent active modes {sorted(modes)}')

    def _trigger_rule_for_position(self, symbol: str, order_type: str) -> int:
        """根据持仓方向选择触发规则"""
        mode_info = self.get_position_mode(symbol)
        mode = mode_info.get('mode', 'none')

        if mode in ('none', 'dual_both'):
            raise RuntimeError(f'Cannot infer trigger rule from position mode: {mode}')

        is_short = mode == 'dual_short'
        if mode == 'single':
            positions = self.positions(symbol)
            if not positions:
                raise RuntimeError('Cannot infer trigger rule without an active position')
            if len(positions) != 1:
                raise RuntimeError(f'Cannot infer trigger rule from multiple single-mode positions: {len(positions)}')
            is_short = positions[0].size < 0

        if order_type == 'stop_loss':
            return 2 if is_short else 1
        if order_type == 'take_profit':
            return 1 if is_short else 2
        raise ValueError(f'Unsupported trigger order type: {order_type}')

    def _trigger_order_target(self, symbol: str) -> Dict[str, Any]:
        """根据当前持仓推导 Gate 保护单平仓语义"""
        mode_info = self.get_position_mode(symbol)
        mode = mode_info.get('mode', 'none')

        if mode == 'dual_long':
            return {'order_type': 'close-long-position', 'auto_size': 'close_long'}
        if mode == 'dual_short':
            return {'order_type': 'close-short-position', 'auto_size': 'close_short'}
        if mode == 'single':
            positions = self.positions(symbol)
            if not positions:
                raise RuntimeError('Cannot infer trigger order target without an active position')
            if len(positions) != 1:
                raise RuntimeError(f'Cannot infer trigger order target from multiple single-mode positions: {len(positions)}')
            if positions[0].size < 0:
                return {'order_type': 'close-short-position', 'auto_size': None}
            return {'order_type': 'close-long-position', 'auto_size': None}
        raise RuntimeError(f'Cannot infer trigger order target from position mode: {mode}')

    def _is_matching_trigger_order(
        self,
        order: Dict[str, Any],
        symbol: str,
        rule: int,
        expected_order_type: str,
        expected_auto_size: Optional[str],
    ) -> bool:
        """判断旧保护单是否与当前准备创建的保护单属于同一语义目标"""
        trigger = order.get('trigger') or {}
        initial = order.get('initial') or {}
        existing_rule = trigger.get('rule')
        strategy_type = trigger.get('strategy_type')
        order_id = order.get('id')
        order_symbol = initial.get('contract')
        existing_order_type = order.get('order_type')
        is_close = initial.get('close')
        auto_size = initial.get('auto_size')
        reduce_only = initial.get('reduce_only')

        try:
            normalized_rule = int(existing_rule)
            normalized_strategy_type = int(strategy_type)
        except (TypeError, ValueError):
            raise RuntimeError('found active trigger orders with unknown trigger metadata')

        if order_symbol and order_symbol != symbol:
            return False

        if normalized_strategy_type != 0:
            raise RuntimeError('found active trigger orders with unsupported strategy_type')

        is_supported_close_order = existing_order_type in ('close-long-position', 'close-short-position')
        is_single_close = is_close is True and auto_size is None
        is_dual_close = is_close in (None, False) and auto_size in ('close_long', 'close_short')

        if not is_supported_close_order:
            return False

        if not (is_single_close or is_dual_close):
            return False

        if reduce_only not in (None, True):
            raise RuntimeError('found active trigger orders with unsupported reduce_only flag')

        if existing_order_type != expected_order_type:
            return False

        if auto_size != expected_auto_size:
            return False

        if normalized_rule != rule:
            return False

        if not order_id:
            raise RuntimeError('found active trigger orders without id')

        return True

    def _replace_trigger_orders(self, symbol: str, rule: int, order_type: str) -> None:
        """创建新保护单前，仅替换本封装创建的同类旧保护单"""
        existing_orders = self._run(self._adapter.fetch_trigger_orders(symbol))
        orders_to_cancel = []
        target = self._trigger_order_target(symbol)

        for order in existing_orders:
            try:
                matches_target = self._is_matching_trigger_order(
                    order=order,
                    symbol=symbol,
                    rule=rule,
                    expected_order_type=target['order_type'],
                    expected_auto_size=target['auto_size'],
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f'Cannot safely replace existing {order_type} order(s) for {symbol}: {exc}'
                ) from exc

            if matches_target:
                orders_to_cancel.append(str(order['id']))

        for order_id in orders_to_cancel:
            cancelled = self._run(self._adapter.cancel_trigger_order(order_id))
            if not cancelled:
                raise RuntimeError(f'Failed to cancel existing {order_type} order {order_id} for {symbol}')

    def set_stop_loss(
        self,
        symbol: str = 'ETH',
        trigger_price: float = None,
        callback_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """设置止损单"""
        self._require_authenticated('set stop loss')
        normalized_symbol = self._normalize_symbol(symbol)
        validated_trigger_price = self._validate_trigger_price(trigger_price, 'stop_loss')
        rule = self._trigger_rule_for_position(normalized_symbol, 'stop_loss')
        current_price = float(self.price(normalized_symbol))
        self._validate_trigger_price_direction(validated_trigger_price, current_price, rule, 'stop_loss')
        self._replace_trigger_orders(normalized_symbol, rule, 'stop_loss')
        target = self._trigger_order_target(normalized_symbol)
        return self._run(self._adapter.create_trigger_order(
            symbol=normalized_symbol,
            trigger_price=validated_trigger_price,
            rule=rule,
            order_type='stop_loss',
            close_order_type=target['order_type'],
            auto_size=target['auto_size'],
        ))

    def set_take_profit(
        self,
        symbol: str = 'ETH',
        trigger_price: float = None,
        callback_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """设置止盈单"""
        self._require_authenticated('set take profit')
        normalized_symbol = self._normalize_symbol(symbol)
        validated_trigger_price = self._validate_trigger_price(trigger_price, 'take_profit')
        rule = self._trigger_rule_for_position(normalized_symbol, 'take_profit')
        current_price = float(self.price(normalized_symbol))
        self._validate_trigger_price_direction(validated_trigger_price, current_price, rule, 'take_profit')
        self._replace_trigger_orders(normalized_symbol, rule, 'take_profit')
        target = self._trigger_order_target(normalized_symbol)
        return self._run(self._adapter.create_trigger_order(
            symbol=normalized_symbol,
            trigger_price=validated_trigger_price,
            rule=rule,
            order_type='take_profit',
            close_order_type=target['order_type'],
            auto_size=target['auto_size'],
        ))

    def cancel_all_trigger_orders(self, symbol: str = 'ETH') -> bool:
        """取消指定交易对的所有保护单"""
        self._require_authenticated('cancel trigger orders')
        return self._run(self._adapter.cancel_all_trigger_orders(self._normalize_symbol(symbol)))

    def _trigger_order_targets_for_side(self, position_side: str) -> List[Dict[str, Optional[str]]]:
        """返回指定方向可能存在的平仓保护单语义（兼容 single/dual mode 遗留单）"""
        if position_side == 'long':
            return [
                {'order_type': 'close-long-position', 'auto_size': 'close_long'},
                {'order_type': 'close-long-position', 'auto_size': None},
            ]
        if position_side == 'short':
            return [
                {'order_type': 'close-short-position', 'auto_size': 'close_short'},
                {'order_type': 'close-short-position', 'auto_size': None},
            ]
        raise ValueError(f'Unsupported position side: {position_side}')

    def cancel_close_trigger_orders(self, symbol: str = 'ETH', position_side: str = 'long') -> bool:
        """仅取消指定方向对应的平仓保护单"""
        self._require_authenticated('cancel trigger orders')
        normalized_symbol = self._normalize_symbol(symbol)
        existing_orders = self._run(self._adapter.fetch_trigger_orders(normalized_symbol))
        targets = self._trigger_order_targets_for_side(position_side)
        cancelled_any = False
        skipped_orders = []

        for order in existing_orders:
            try:
                matches_target = any(
                    self._is_matching_trigger_order(
                        order=order,
                        symbol=normalized_symbol,
                        rule=rule,
                        expected_order_type=target['order_type'],
                        expected_auto_size=target['auto_size'],
                    )
                    for target in targets
                    for rule in (1, 2)
                )
            except RuntimeError as exc:
                skipped_orders.append(f"{order.get('id', '<missing-id>')}: {exc}")
                continue

            if not matches_target:
                continue

            cancelled = self._run(self._adapter.cancel_trigger_order(str(order['id'])))
            if not cancelled:
                raise RuntimeError(f"Failed to cancel trigger order {order['id']} for {normalized_symbol}")
            cancelled_any = True

        if skipped_orders:
            print(
                f"警告: 跳过 {normalized_symbol} 上 {len(skipped_orders)} 个无法安全识别的保护单: "
                + "; ".join(skipped_orders)
            )

        return cancelled_any

    def cancel_all_orders(self, symbol: str = 'ETH') -> bool:
        """取消所有订单"""
        self._require_authenticated('cancel all orders')
        return self._run(self._adapter.cancel_all_orders(self._normalize_symbol(symbol)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# 快捷函数
_gate_instance = None


def get_gate() -> GateSync:
    """获取 Gate.io 同步实例 (单例)"""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = GateSync()
    return _gate_instance


def price(symbol: str = 'ETH') -> float:
    """获取价格"""
    return get_gate().price(symbol)


def balance() -> Dict[str, Any]:
    """获取余额"""
    return get_gate().balance()


def positions(symbol: str = 'ETH') -> List[Position]:
    """获取持仓"""
    return get_gate().positions(symbol)


if __name__ == '__main__':
    # 测试
    with GateSync() as gate:
        print(f'ETH Price: ${gate.price()}')
        print(f'Balance: {gate.balance()}')
        print(f'Positions: {gate.positions()}')
