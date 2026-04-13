# -*- coding: utf-8 -*-

from __future__ import annotations

from quant_trading.signal.probability_engine import (
    ProbabilityEngine,
    ProbabilityModelResult,
    ProbabilitySignal,
)
from quant_trading.execution.auto_trader import AutoTrader, TradeConfig, TradeState


def make_model(name: str, signal: str, confidence: float, weight: float) -> ProbabilityModelResult:
    return ProbabilityModelResult(
        name=name,
        signal=signal,
        confidence=confidence,
        value=0.0,
        weight=weight,
    )


def test_calculate_weighted_signal_includes_bullish_and_bearish_tokens_in_direction_scores():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    weighted = engine._calculate_weighted_signal(
        [
            make_model("omega", "BULLISH", 0.90, 0.15),
            make_model("bayesian", "BUY", 0.80, 0.20),
            make_model("entropy", "TREND", 0.70, 0.10),
            make_model("black_scholes", "SELL", 0.60, 0.20),
            make_model("other", "BEARISH", 0.95, 0.35),
        ]
    )

    assert round(weighted["confidence"], 4) == 0.8175
    assert round(weighted["bullish"], 4) == 0.33
    assert round(weighted["bearish"], 4) == 0.4525
    assert round(weighted["net_direction"], 4) == -0.1225


def test_calculate_weighted_signal_treats_unknown_tokens_as_confidence_only_without_direction():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    weighted = engine._calculate_weighted_signal(
        [
            make_model("omega", "BUY", 0.80, 0.25),
            make_model("future_model", "RISK_OFF", 0.90, 0.75),
        ]
    )

    assert round(weighted["confidence"], 4) == 0.875
    assert round(weighted["bullish"], 4) == 0.20
    assert round(weighted["bearish"], 4) == 0.0
    assert round(weighted["net_direction"], 4) == 0.20



def test_calculate_weighted_signal_treats_trend_as_half_strength_bullish_and_chaos_as_half_strength_bearish():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    weighted = engine._calculate_weighted_signal(
        [
            make_model("entropy", "TREND", 0.90, 0.10),
            make_model("entropy", "CHAOS", 0.80, 0.10),
        ]
    )

    assert round(weighted["bullish"], 4) == 0.225
    assert round(weighted["bearish"], 4) == 0.20
    assert round(weighted["net_direction"], 4) == 0.025


def test_determine_signal_returns_strong_buy_only_when_both_thresholds_pass():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    signal = engine._determine_signal({
        "net_direction": 0.25,
        "confidence": 0.60,
    })

    assert signal == ProbabilitySignal.STRONG_BUY


def test_determine_signal_returns_neutral_when_confidence_is_high_but_direction_too_small():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    signal = engine._determine_signal({
        "net_direction": 0.14,
        "confidence": 0.99,
    })

    assert signal == ProbabilitySignal.NEUTRAL


def test_determine_signal_returns_neutral_when_direction_is_strong_but_confidence_too_low():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    signal = engine._determine_signal({
        "net_direction": -0.40,
        "confidence": 0.49,
    })

    assert signal == ProbabilitySignal.NEUTRAL


def test_determine_signal_returns_buy_at_exact_buy_threshold():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    signal = engine._determine_signal({
        "net_direction": 0.15,
        "confidence": 0.50,
    })

    assert signal == ProbabilitySignal.BUY




def test_determine_signal_returns_strong_sell_at_exact_strong_sell_threshold():
    engine = ProbabilityEngine.__new__(ProbabilityEngine)

    signal = engine._determine_signal({
        "net_direction": -0.25,
        "confidence": 0.60,
    })

    assert signal == ProbabilitySignal.STRONG_SELL


class FakeGate:
    def __init__(self):
        self.open_long_prices = []
        self.open_short_prices = []

    def price(self, symbol):
        return 100.0

    def ohlcv(self, symbol, timeframe='1h', limit=100):
        raise AssertionError("ohlcv should not be called in this test")


class FakeResult:
    def __init__(self, signal, confidence):
        self.signal = signal
        self.confidence = confidence


def make_trader() -> AutoTrader:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = TradeConfig(symbol="ETH/USDT", leverage=10)
    trader.symbol = "ETH_USDT"
    trader.gate = FakeGate()
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


def test_check_entry_opens_long_only_for_strong_buy_over_threshold(monkeypatch):
    trader = make_trader()
    calls = []
    monkeypatch.setattr(trader, "_open_long", lambda price: calls.append(("long", price)))
    monkeypatch.setattr(trader, "_open_short", lambda price: calls.append(("short", price)))

    threshold = trader.config.strong_buy_threshold
    trader._check_entry(ProbabilitySignal.STRONG_BUY, threshold, 123.4)
    trader._check_entry(ProbabilitySignal.BUY, 0.99, 123.4)
    trader._check_entry(ProbabilitySignal.STRONG_BUY, threshold - 0.01, 123.4)

    assert calls == [("long", 123.4)]


def test_check_entry_opens_short_only_for_strong_sell_over_threshold(monkeypatch):
    trader = make_trader()
    calls = []
    monkeypatch.setattr(trader, "_open_long", lambda price: calls.append(("long", price)))
    monkeypatch.setattr(trader, "_open_short", lambda price: calls.append(("short", price)))

    threshold = trader.config.strong_sell_threshold
    trader._check_entry(ProbabilitySignal.STRONG_SELL, threshold, 88.8)
    trader._check_entry(ProbabilitySignal.SELL, 0.99, 88.8)
    trader._check_entry(ProbabilitySignal.STRONG_SELL, threshold - 0.01, 88.8)

    assert calls == [("short", 88.8)]
