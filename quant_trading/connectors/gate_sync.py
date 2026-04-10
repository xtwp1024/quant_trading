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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# 自动加载 .env 文件
_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

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

    def __init__(self):
        self.api_key = os.environ.get('GATE_API_KEY', '')
        self.api_secret = os.environ.get('GATE_API_SECRET', '')

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
            self._loop.run_until_complete(self._adapter.initialize())
            self._initialized = True

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
        ticker = self._run(self._adapter.get_ticker(f'{symbol}_USDT'))
        return ticker.get('last', 0)

    def ohlcv(
        self,
        symbol: str = 'ETH',
        timeframe: str = '15m',
        limit: int = 100
    ) -> List[List]:
        """获取K线数据"""
        return self._run(self._adapter.get_ohlcv(f'{symbol}_USDT', timeframe, limit))

    def balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        return self._run(self._adapter.get_balance())

    def positions(self, symbol: str = 'ETH') -> List[Position]:
        """获取持仓"""
        raw = self._run(self._adapter.get_position(f'{symbol}_USDT'))
        return [
            Position(
                symbol=p.get('symbol', symbol),
                size=float(p.get('size', 0)),
                entry_price=float(p.get('entry_price', 0)),
                liq_price=float(p.get('liq_price', 0)),
                pnl=float(p.get('unrealized_pnl', 0)),
                margin=float(p.get('margin', 0)),
            )
            for p in raw
        ]

    def buy_market(
        self,
        symbol: str = 'ETH',
        amount: float = 1,
        leverage: int = 10
    ) -> Dict[str, Any]:
        """市价做多"""
        self._run(self._adapter.set_leverage(leverage, f'{symbol}_USDT'))
        return self._run(self._adapter.create_order(
            symbol=f'{symbol}_USDT',
            order_side='buy',
            order_type='market',
            amount=amount,
            price=None,
        ))

    def sell_market(
        self,
        symbol: str = 'ETH',
        amount: float = 1,
        leverage: int = 10
    ) -> Dict[str, Any]:
        """市价做空"""
        self._run(self._adapter.set_leverage(leverage, f'{symbol}_USDT'))
        return self._run(self._adapter.create_order(
            symbol=f'{symbol}_USDT',
            order_side='sell',
            order_type='market',
            amount=amount,
            price=None,
        ))

    def close_all(self, symbol: str = 'ETH') -> List[Dict]:
        """平所有仓位"""
        positions = self.positions(symbol)
        orders = []
        for pos in positions:
            if pos.size == 0:
                continue
            side = 'sell' if pos.size > 0 else 'buy'
            order = self._run(self._adapter.create_order(
                symbol=f'{symbol}_USDT',
                order_side=side,
                order_type='market',
                amount=abs(pos.size),
                price=None,
            ))
            orders.append(order)
        return orders

    def set_stop_loss(
        self,
        symbol: str = 'ETH',
        trigger_price: float = None,
        callback_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """设置止损单"""
        return self._run(self._adapter.create_trigger_order(
            symbol=f'{symbol}_USDT',
            trigger_price=trigger_price,
            rule=1,  # <= trigger
            order_type='stop_loss',
        ))

    def set_take_profit(
        self,
        symbol: str = 'ETH',
        trigger_price: float = None,
        callback_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """设置止盈单"""
        return self._run(self._adapter.create_trigger_order(
            symbol=f'{symbol}_USDT',
            trigger_price=trigger_price,
            rule=2,  # >= trigger
            order_type='take_profit',
        ))

    def cancel_all_orders(self, symbol: str = 'ETH') -> bool:
        """取消所有订单"""
        return self._run(self._adapter.cancel_all_orders(f'{symbol}_USDT'))

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
