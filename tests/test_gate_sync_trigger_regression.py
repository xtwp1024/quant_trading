# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from quant_trading.connectors.gate_sync import GateSync


class FakeAdapter:
    def __init__(self, orders, *, cancel_results=None):
        self.orders = list(orders)
        self.cancel_results = list(cancel_results or [])
        self.cancelled_ids = []

    def fetch_trigger_orders(self, symbol):
        return self.orders

    def cancel_trigger_order(self, order_id):
        self.cancelled_ids.append(order_id)
        if self.cancel_results:
            return self.cancel_results.pop(0)
        return True


def make_gate_sync(orders, *, cancel_results=None):
    gate = GateSync.__new__(GateSync)
    gate._adapter = FakeAdapter(orders, cancel_results=cancel_results)
    gate._run = lambda value: value
    gate._require_authenticated = lambda operation: None
    gate._normalize_symbol = lambda symbol: "ETH_USDT"
    return gate


def make_trigger_order(
    order_id: str,
    *,
    order_type: str,
    rule: int,
    auto_size,
    contract: str = "ETH_USDT",
    close=None,
    strategy_type: int = 0,
    reduce_only=True,
):
    return {
        "id": order_id,
        "order_type": order_type,
        "trigger": {
            "rule": rule,
            "strategy_type": strategy_type,
        },
        "initial": {
            "contract": contract,
            "close": close,
            "auto_size": auto_size,
            "reduce_only": reduce_only,
        },
    }


def test_cancel_close_trigger_orders_only_cancels_requested_side():
    long_order = make_trigger_order(
        "long-1",
        order_type="close-long-position",
        rule=1,
        auto_size="close_long",
        close=False,
    )
    short_order = make_trigger_order(
        "short-1",
        order_type="close-short-position",
        rule=1,
        auto_size="close_short",
        close=False,
    )
    gate = make_gate_sync([long_order, short_order])

    cancelled = gate.cancel_close_trigger_orders("ETH", position_side="long")

    assert cancelled is True
    assert gate._adapter.cancelled_ids == ["long-1"]


def test_cancel_close_trigger_orders_matches_legacy_single_mode_orders_for_long_side():
    legacy_long_order = make_trigger_order(
        "legacy-long-1",
        order_type="close-long-position",
        rule=2,
        auto_size=None,
        close=True,
    )
    legacy_short_order = make_trigger_order(
        "legacy-short-1",
        order_type="close-short-position",
        rule=2,
        auto_size=None,
        close=True,
    )
    gate = make_gate_sync([legacy_long_order, legacy_short_order])

    cancelled = gate.cancel_close_trigger_orders("ETH", position_side="long")

    assert cancelled is True
    assert gate._adapter.cancelled_ids == ["legacy-long-1"]


def test_cancel_close_trigger_orders_only_cancels_requested_short_side():
    long_order = make_trigger_order(
        "long-1",
        order_type="close-long-position",
        rule=1,
        auto_size="close_long",
        close=False,
    )
    short_order = make_trigger_order(
        "short-1",
        order_type="close-short-position",
        rule=1,
        auto_size="close_short",
        close=False,
    )
    gate = make_gate_sync([long_order, short_order])

    cancelled = gate.cancel_close_trigger_orders("ETH", position_side="short")

    assert cancelled is True
    assert gate._adapter.cancelled_ids == ["short-1"]


def test_cancel_close_trigger_orders_skips_unknown_metadata_and_continues_batch(capsys):
    bad_order = {
        "id": "bad-1",
        "order_type": "close-long-position",
        "trigger": {"rule": 1},
        "initial": {
            "contract": "ETH_USDT",
            "close": False,
            "auto_size": "close_long",
            "reduce_only": True,
        },
    }
    good_order = make_trigger_order(
        "good-1",
        order_type="close-long-position",
        rule=2,
        auto_size="close_long",
        close=False,
    )
    gate = make_gate_sync([bad_order, good_order])

    cancelled = gate.cancel_close_trigger_orders("ETH", position_side="long")

    captured = capsys.readouterr()
    assert cancelled is True
    assert gate._adapter.cancelled_ids == ["good-1"]
    assert "跳过 ETH_USDT 上 1 个无法安全识别的保护单" in captured.out
    assert "bad-1" in captured.out


def test_cancel_close_trigger_orders_raises_when_matched_cancel_fails():
    long_order = make_trigger_order(
        "long-1",
        order_type="close-long-position",
        rule=1,
        auto_size="close_long",
        close=False,
    )
    gate = make_gate_sync([long_order], cancel_results=[False])

    with pytest.raises(RuntimeError, match="Failed to cancel trigger order long-1 for ETH_USDT"):
        gate.cancel_close_trigger_orders("ETH", position_side="long")
