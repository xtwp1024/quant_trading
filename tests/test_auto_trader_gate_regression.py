# -*- coding: utf-8 -*-

from __future__ import annotations

from quant_trading.connectors.gate_sync import Position
from quant_trading.execution.auto_trader import AutoTrader, TradeConfig, TradeState


class FakeGate:
    def __init__(self, positions_sequence=None):
        self.positions_sequence = list(positions_sequence or [])
        self.buy_market_calls = []
        self.sell_market_calls = []
        self.stop_loss_calls = []
        self.take_profit_calls = []
        self.cancel_close_trigger_orders_calls = []
        self.price_value = 100.0

    def positions(self, symbol):
        if self.positions_sequence:
            return self.positions_sequence.pop(0)
        return []

    def buy_market(self, symbol, amount, leverage, reduce_only=False):
        self.buy_market_calls.append(
            {
                "symbol": symbol,
                "amount": amount,
                "leverage": leverage,
                "reduce_only": reduce_only,
            }
        )
        return {"id": "buy-order"}

    def sell_market(self, symbol, amount, leverage, reduce_only=False):
        self.sell_market_calls.append(
            {
                "symbol": symbol,
                "amount": amount,
                "leverage": leverage,
                "reduce_only": reduce_only,
            }
        )
        return {"id": "sell-order"}

    def set_stop_loss(self, symbol, trigger_price):
        self.stop_loss_calls.append((symbol, trigger_price))

    def set_take_profit(self, symbol, trigger_price):
        self.take_profit_calls.append((symbol, trigger_price))

    def cancel_close_trigger_orders(self, symbol, position_side):
        self.cancel_close_trigger_orders_calls.append((symbol, position_side))

    def price(self, symbol):
        return self.price_value


def make_trader(gate: FakeGate, *, position_size: float = 1.0, leverage: int = 10) -> AutoTrader:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = TradeConfig(
        symbol="ETH/USDT",
        leverage=leverage,
        position_size=position_size,
        stop_loss_margin_pct=0.10,
        take_profit_margin_pct=0.20,
    )
    trader.symbol = "ETH_USDT"
    trader.gate = gate
    trader.state = TradeState()
    trader.running = False
    trader.thread = None
    trader.stats = {
        "start_time": None,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
    }
    trader.trade_log = []
    return trader


def make_position(size: float) -> Position:
    return Position(
        symbol="ETH_USDT",
        size=size,
        entry_price=100.0,
        liq_price=0.0,
        pnl=0.0,
        margin=0.0,
    )


def test_open_long_uses_live_position_delta_when_filled_missing():
    gate = FakeGate(
        positions_sequence=[
            [make_position(2.0)],
            [make_position(5.0)],
        ]
    )
    trader = make_trader(gate, position_size=9.0)

    trader._open_long(100.0)

    assert trader.state.has_position is True
    assert trader.state.position_side == "long"
    assert trader.state.position_size == 3.0
    assert trader.state.entry_price == 100.0
    assert trader.state.trade_count == 1
    assert gate.buy_market_calls == [
        {
            "symbol": "ETH",
            "amount": 9.0,
            "leverage": 10,
            "reduce_only": False,
        }
    ]
    assert len(gate.stop_loss_calls) == 1
    assert len(gate.take_profit_calls) == 1


def test_open_short_uses_live_position_delta_when_filled_missing():
    gate = FakeGate(
        positions_sequence=[
            [make_position(-1.0)],
            [make_position(-3.5)],
        ]
    )
    trader = make_trader(gate, position_size=8.0)

    trader._open_short(100.0)

    assert trader.state.has_position is True
    assert trader.state.position_side == "short"
    assert trader.state.position_size == 2.5
    assert trader.state.entry_price == 100.0
    assert trader.state.trade_count == 1
    assert gate.sell_market_calls == [
        {
            "symbol": "ETH",
            "amount": 8.0,
            "leverage": 10,
            "reduce_only": False,
        }
    ]
    assert len(gate.stop_loss_calls) == 1
    assert len(gate.take_profit_calls) == 1


def test_sync_position_closed_keeps_state_when_opposite_side_residual_exists():
    gate = FakeGate(positions_sequence=[[make_position(-1.25)]])
    trader = make_trader(gate)
    trader.state.has_position = True
    trader.state.position_side = "long"
    trader.state.entry_price = 100.0
    trader.state.position_size = 2.0
    trader.state.margin = 20.0

    closed = trader._sync_position_closed("ETH")

    assert closed is False
    assert trader.state.has_position is True
    assert trader.state.position_side == "long"
    assert trader.state.position_size == 2.0
    assert trader.state.margin == 20.0


def test_sync_position_closed_updates_same_side_residual_instead_of_resetting():
    gate = FakeGate(positions_sequence=[[make_position(0.75)]])
    trader = make_trader(gate)
    trader.state.has_position = True
    trader.state.position_side = "long"
    trader.state.entry_price = 100.0
    trader.state.position_size = 2.0
    trader.state.margin = 20.0

    closed = trader._sync_position_closed("ETH")

    assert closed is False
    assert trader.state.position_size == 0.75
    assert trader.state.margin == 7.5
    assert trader.state.has_position is True


def test_close_position_does_not_reset_local_state_when_sync_reports_not_closed(monkeypatch):
    gate = FakeGate()
    trader = make_trader(gate)
    trader.state.has_position = True
    trader.state.position_side = "long"
    trader.state.entry_price = 100.0
    trader.state.position_size = 2.0
    trader.state.margin = 20.0

    monkeypatch.setattr(trader, "_sync_position_closed", lambda symbol: False)

    trader._close_position("regression")

    assert gate.sell_market_calls == [
        {
            "symbol": "ETH",
            "amount": 2.0,
            "leverage": 10,
            "reduce_only": True,
        }
    ]
    assert trader.state.has_position is True
    assert trader.state.position_side == "long"
    assert gate.cancel_close_trigger_orders_calls == []


def test_get_status_includes_visible_position_fields():
    gate = FakeGate()
    trader = make_trader(gate)
    trader.running = True
    trader.state.has_position = True
    trader.state.position_side = "long"
    trader.state.entry_price = 101.5
    trader.state.position_size = 3.25
    trader.state.margin = 12.5
    trader.state.stop_loss_price = 99.0
    trader.state.take_profit_price = 105.0
    trader.state.unrealized_pnl = 8.75
    trader.state.realized_pnl = -1.5
    trader.state.trade_count = 2
    trader.state.last_signal = "STRONG_BUY"
    trader.state.last_confidence = 0.88

    status = trader.get_status()

    assert status["running"] is True
    assert status["has_position"] is True
    assert status["position_side"] == "long"
    assert status["entry_price"] == 101.5
    assert status["position_size"] == 3.25
    assert status["margin"] == 12.5
    assert status["stop_loss_price"] == 99.0
    assert status["take_profit_price"] == 105.0
    assert status["unrealized_pnl"] == 8.75
    assert status["realized_pnl"] == -1.5
    assert status["trade_count"] == 2
    assert status["last_signal"] == "STRONG_BUY"
    assert status["last_confidence"] == 0.88


def test_get_status_clears_position_fields_after_local_reset():
    gate = FakeGate()
    trader = make_trader(gate)
    trader.state.has_position = True
    trader.state.position_side = "short"
    trader.state.entry_price = 100.0
    trader.state.position_size = 2.0
    trader.state.margin = 20.0
    trader.state.stop_loss_price = 101.0
    trader.state.take_profit_price = 98.0
    trader.state.unrealized_pnl = 3.0

    trader._reset_local_position_state()
    status = trader.get_status()

    assert status["has_position"] is False
    assert status["position_side"] == ""
    assert status["entry_price"] == 0.0
    assert status["position_size"] == 0.0
    assert status["margin"] == 0.0
    assert status["stop_loss_price"] == 0.0
    assert status["take_profit_price"] == 0.0
    assert status["unrealized_pnl"] == 0.0


def test_check_and_trade_prints_visible_position_snapshot(monkeypatch, capsys):
    gate = FakeGate()
    trader = make_trader(gate)
    trader.state.has_position = True
    trader.state.position_side = "long"
    trader.state.entry_price = 100.0
    trader.state.position_size = 2.5
    trader.state.margin = 25.0
    trader.state.stop_loss_price = 99.0
    trader.state.take_profit_price = 102.0

    class FakeResult:
        signal = type("Signal", (), {"value": "STRONG_BUY"})()
        confidence = 0.77

    monkeypatch.setattr(trader, "_get_signal", lambda: FakeResult())
    monkeypatch.setattr(trader, "_check_exit", lambda current_price: None)

    trader._check_and_trade()

    captured = capsys.readouterr()
    assert "持仓: long" in captured.out
    assert "数量: 2.5000" in captured.out
    assert "开仓价: $100.00" in captured.out
    assert "保证金: $25.00" in captured.out
    assert "止损: $99.00" in captured.out
    assert "止盈: $102.00" in captured.out
