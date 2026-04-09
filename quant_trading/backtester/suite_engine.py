"""SuiteTrading v2 — dual-runner backtesting engine.

Adapts the SuiteTrading engine (Simple + FSM runners) for integration
with the existing ``quant_trading.backtester`` namespace.

Double-runner architecture
--------------------------
- **SimpleRunner** (``run_simple_backtest``): lightweight bar loop,
  no pyramiding / no partial TP.  Throughput ~63+ bt/sec.  Suitable for
  high-throughput archetype A/B screening of vectorisable strategies.
- **FSMRunner** (``run_fsm_backtest``): full state-machine bar loop with
  position sizing, trailing, partial TP, break-even and pyramiding.
  Handles the full 6-shift × 6-stop × 6-archetype risk prototype space.

Both paths are deterministic: identical inputs produce identical outputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd


# ── Dataclasses (aligned with SuiteTrading _internal/schemas.py) ────────────

@dataclass
class TradeRecord:
    """Record of a single completed trade."""

    entry_bar: int
    exit_bar: int
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float  # GROSS PnL (price-only, no commission)
    exit_reason: str
    commission: float = 0.0  # Total commission (entry + exit)


@dataclass
class BacktestResult:
    """Complete output of a single backtest run."""

    equity_curve: np.ndarray
    trades: list[TradeRecord] = field(default_factory=list)
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    mode: str = "fsm"


# ── Enums (aligned with suitetrading.risk.contracts) ────────────────────────

class PositionState:
    """Position lifecycle states (mirrors suitetrading.risk.contracts)."""

    FLAT = "flat"
    OPEN_INITIAL = "open_initial"
    OPEN_BREAKEVEN = "open_breakeven"
    OPEN_TRAILING = "open_trailing"
    OPEN_PYRAMIDED = "open_pyramided"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"


# ── Position snapshot ────────────────────────────────────────────────────────

@dataclass
class PositionSnapshot:
    """Self-contained snapshot of a position at a given bar."""

    state: str = PositionState.FLAT
    direction: str = "flat"  # "long" | "short" | "flat"
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    stop_price: float | None = None
    break_even_price: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    pyramid_level: int = 0
    tp1_hit: bool = False
    tp1_bar_index: int | None = None
    entry_bar_index: int | None = None
    last_order_bar_index: int | None = None
    bars_in_position: int = 0


# ── RiskConfig (simplified Pydantic-free version for standalone use) ────────

@dataclass
class SizingConfig:
    model: str = "fixed_fractional"
    risk_pct: float = 1.0
    atr_multiple: float = 2.0
    max_risk_per_trade: float = 5.0
    max_position_size: float = 1e12
    max_leverage: float = 1.0


@dataclass
class StopConfig:
    model: str = "atr"  # "atr" | "fixed_pct"
    atr_multiple: float = 2.0
    fixed_pct: float = 2.0


@dataclass
class TrailingConfig:
    trailing_mode: str = "signal"  # "signal" | "policy"
    atr_multiple: float = 2.5


@dataclass
class PartialTPConfig:
    enabled: bool = True
    close_pct: float = 35.0
    trigger: str = "signal"  # "signal" | "r_multiple"
    r_multiple: float = 1.0
    profit_distance_factor: float = 1.01


@dataclass
class BreakEvenConfig:
    enabled: bool = True
    buffer: float = 1.0007
    activation: str = "after_tp1"  # "after_tp1" | "r_multiple"


@dataclass
class PyramidConfig:
    enabled: bool = False
    max_adds: int = 0
    block_bars: int = 15
    threshold_factor: float = 1.01


@dataclass
class TimeExitConfig:
    enabled: bool = False
    max_bars: int = 100


@dataclass
class SuiteRiskConfig:
    """Simplified risk config compatible with SuiteTrading engine."""

    archetype: str = "mixed"
    initial_capital: float = 4000.0
    commission_pct: float = 0.10
    slippage_pct: float = 0.0
    sizing: SizingConfig = field(default_factory=SizingConfig)
    stop: StopConfig = field(default_factory=StopConfig)
    trailing: TrailingConfig = field(default_factory=TrailingConfig)
    partial_tp: PartialTPConfig = field(default_factory=PartialTPConfig)
    break_even: BreakEvenConfig = field(default_factory=BreakEvenConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    time_exit: TimeExitConfig = field(default_factory=TimeExitConfig)


# ── StrategySignals ──────────────────────────────────────────────────────────

@dataclass
class StrategySignals:
    """Pre-computed boolean signal arrays aligned to the base timeframe."""

    entry_long: np.ndarray
    entry_short: np.ndarray | None = None
    exit_long: np.ndarray | None = None
    exit_short: np.ndarray | None = None
    trailing_long: np.ndarray | None = None
    trailing_short: np.ndarray | None = None
    indicators_payload: dict[str, Any] = field(default_factory=dict)


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class BacktestDataset:
    """Self-contained bundle of OHLCV data for a single run."""

    symbol: str
    base_timeframe: str
    ohlcv: pd.DataFrame  # requires: open, high, low, close


# ── FSM Runner ────────────────────────────────────────────────────────────────

def run_fsm_backtest(
    *,
    dataset: BacktestDataset,
    signals: StrategySignals,
    risk_config: SuiteRiskConfig,
    direction: str = "long",
) -> BacktestResult:
    """Full FSM bar-loop backtest using PositionStateMachine.

    Processes every bar through the state machine, handling entries,
    exits, stop-loss, partial TP, break-even, trailing and pyramiding.
    """
    ohlcv = dataset.ohlcv
    n = len(ohlcv)
    if n == 0:
        return BacktestResult(equity_curve=np.array([]), final_equity=0.0)

    snapshot = PositionSnapshot()
    equity = risk_config.initial_capital
    equity_curve = np.full(n, equity)
    trades: list[TradeRecord] = []
    commission_pct = risk_config.commission_pct

    # Pre-extract arrays for speed
    opens = ohlcv["open"].values
    highs = ohlcv["high"].values
    lows = ohlcv["low"].values
    closes = ohlcv["close"].values

    entry_long = signals.entry_long if signals.entry_long is not None else np.zeros(n, dtype=bool)
    entry_short = signals.entry_short if signals.entry_short is not None else np.zeros(n, dtype=bool)
    exit_long = signals.exit_long if signals.exit_long is not None else np.zeros(n, dtype=bool)
    exit_short = signals.exit_short if signals.exit_short is not None else np.zeros(n, dtype=bool)
    trailing_long = signals.trailing_long if signals.trailing_long is not None else np.zeros(n, dtype=bool)
    trailing_short = signals.trailing_short if signals.trailing_short is not None else np.zeros(n, dtype=bool)

    # Pre-compute ATR
    atr_values = _compute_atr(highs, lows, closes, period=14)

    current_entry_bar = 0
    current_entry_price = 0.0
    current_entry_qty = 0.0
    trade_commission = 0.0

    for i in range(n):
        bar = {
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
        }

        # Direction-aware signals
        if direction in ("long", "both"):
            entry_sig = bool(entry_long[i])
            exit_sig = bool(exit_long[i])
            trail_sig = bool(trailing_long[i])
            entry_dir = "long"
        else:
            entry_sig = bool(entry_short[i])
            exit_sig = bool(exit_short[i])
            trail_sig = bool(trailing_short[i])
            entry_dir = "short"

        if direction == "both":
            if snapshot.direction == "short":
                entry_sig = bool(entry_short[i])
                exit_sig = bool(exit_short[i])
                trail_sig = bool(trailing_short[i])
                entry_dir = "short"
            elif snapshot.state == PositionState.FLAT and bool(entry_short[i]) and not bool(entry_long[i]):
                entry_sig = True
                entry_dir = "short"

        # Position sizing
        entry_size = 0.0
        stop_override = None
        if entry_sig and snapshot.state in (PositionState.FLAT,):
            atr_val = float(atr_values[i]) if atr_values[i] > 0 else None
            stop_model = risk_config.stop.model

            if stop_model == "fixed_pct":
                stop_dist = closes[i] * risk_config.stop.fixed_pct / 100.0
                if entry_dir == "long":
                    stop_override = closes[i] - stop_dist
                else:
                    stop_override = closes[i] + stop_dist
            else:
                # Default: ATR-based stop
                stop_dist = (
                    atr_val * risk_config.stop.atr_multiple
                    if atr_val
                    else closes[i] * risk_config.stop.fixed_pct / 100.0
                )
                if entry_dir == "long":
                    stop_override = closes[i] - stop_dist
                else:
                    stop_override = closes[i] + stop_dist

            risk_amount = equity * risk_config.sizing.risk_pct / 100.0
            entry_size = risk_amount / stop_dist if stop_dist > 0 else 0.0
            entry_size = min(entry_size, risk_config.sizing.max_position_size)

        # Evaluate FSM
        result = _evaluate_bar(
            snapshot,
            bar,
            i,
            entry_sig=entry_sig,
            entry_direction=entry_dir,
            exit_sig=exit_sig,
            trailing_sig=trail_sig,
            entry_size=entry_size,
            atr_value=float(atr_values[i]) if i < len(atr_values) else None,
            stop_override=stop_override,
            risk_config=risk_config,
        )

        prev_state = snapshot.state
        snapshot = result["snapshot"]

        # Process orders
        for order in result["orders"]:
            action = order.get("action", "")
            filled_qty = order.get("filled_qty", 0.0)
            price = order.get("price", 0.0)

            if action == "entry":
                current_entry_bar = i
                current_entry_price = price
                current_entry_qty = filled_qty
                comm = abs(filled_qty * price) * commission_pct / 100.0
                equity -= comm
                trade_commission = comm

            elif action in ("close_all", "close_partial"):
                comm = abs(filled_qty * price) * commission_pct / 100.0
                equity -= comm
                trade_commission += comm

            elif action == "pyramid_add":
                comm = abs(filled_qty * price) * commission_pct / 100.0
                equity -= comm
                trade_commission += comm
                current_entry_qty += filled_qty

        # Record trade when position closes
        if prev_state not in (PositionState.FLAT,) and snapshot.state == PositionState.CLOSED:
            trade_pnl = snapshot.realized_pnl
            equity += trade_pnl

            trades.append(TradeRecord(
                entry_bar=current_entry_bar,
                exit_bar=i,
                direction=snapshot.direction,
                entry_price=current_entry_price,
                exit_price=closes[i],
                quantity=current_entry_qty,
                pnl=trade_pnl,
                exit_reason=result.get("reason") or "",
                commission=trade_commission,
            ))
            trade_commission = 0.0
            current_entry_qty = 0.0
            snapshot = _reset_snapshot(snapshot)

        # Mark-to-market
        if snapshot.state not in (PositionState.FLAT, PositionState.CLOSED):
            equity_curve[i] = equity + snapshot.unrealized_pnl
        else:
            equity_curve[i] = equity

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        final_equity=equity,
        total_return_pct=(equity / risk_config.initial_capital - 1.0) * 100.0,
        mode="fsm",
    )


# ── Simple Runner ─────────────────────────────────────────────────────────────

def run_simple_backtest(
    *,
    dataset: BacktestDataset,
    signals: StrategySignals,
    risk_config: SuiteRiskConfig,
) -> BacktestResult:
    """Lightweight single-position backtest (no pyramiding, no partial TP).

    Faster than FSM for high-throughput screening of archetypes A/B.
    Uses a thin bar loop with gap-aware stop-loss tracking.
    """
    ohlcv = dataset.ohlcv
    n = len(ohlcv)
    if n == 0:
        return BacktestResult(equity_curve=np.array([]), final_equity=0.0)

    opens = ohlcv["open"].values
    closes = ohlcv["close"].values
    highs = ohlcv["high"].values
    lows = ohlcv["low"].values
    entries = signals.entry_long if signals.entry_long is not None else np.zeros(n, dtype=bool)
    exits = signals.exit_long if signals.exit_long is not None else np.zeros(n, dtype=bool)

    atr = _compute_atr(highs, lows, closes, period=14)
    slip = risk_config.slippage_pct
    risk_pct = risk_config.sizing.risk_pct
    atr_mult = risk_config.stop.atr_multiple
    fixed_pct = risk_config.stop.fixed_pct
    commission = risk_config.commission_pct

    equity = risk_config.initial_capital
    equity_curve = np.full(n, equity)
    trades: list[TradeRecord] = []

    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    qty = 0.0
    entry_bar = 0
    entry_comm = 0.0
    prev_close = 0.0  # Track previous close for gap-aware stop

    for i in range(n):
        equity_curve[i] = equity

        if in_position:
            # Gap-aware stop: check overnight gap first
            gap_triggered = (i > 0) and (prev_close <= stop_price < opens[i])
            bar_triggered = lows[i] <= stop_price
            if gap_triggered or bar_triggered:
                fill = opens[i]  # Gap down: fill at open
                if slip:
                    fill *= (1 - slip / 100.0)
                pnl = (fill - entry_price) * qty
                equity += pnl
                exit_comm = abs(qty * fill) * commission / 100.0
                equity -= exit_comm
                trades.append(TradeRecord(
                    entry_bar=entry_bar, exit_bar=i, direction="long",
                    entry_price=entry_price, exit_price=fill,
                    quantity=qty, pnl=pnl, exit_reason="SL",
                    commission=entry_comm + exit_comm,
                ))
                in_position = False
            elif exits[i]:
                # Use next bar open for realistic execution
                if i + 1 < n:
                    fill = opens[i + 1]
                else:
                    fill = closes[i]
                if slip:
                    fill *= (1 - slip / 100.0)
                pnl = (fill - entry_price) * qty
                equity += pnl
                exit_comm = abs(qty * fill) * commission / 100.0
                equity -= exit_comm
                trades.append(TradeRecord(
                    entry_bar=entry_bar, exit_bar=i, direction="long",
                    entry_price=entry_price, exit_price=fill,
                    quantity=qty, pnl=pnl, exit_reason="signal",
                    commission=entry_comm + exit_comm,
                ))
                in_position = False

            equity_curve[i] = equity

        # Track previous close for gap detection
        if i > 0:
            prev_close = closes[i - 1]

        if not in_position and entries[i]:
            # Use next bar open for entry to avoid look-ahead bias
            if i + 1 < n:
                entry_price = opens[i + 1]
            else:
                entry_price = closes[i]
            if atr[i] > 0:
                stop_dist = atr[i] * atr_mult
            else:
                stop_dist = entry_price * fixed_pct / 100.0
            stop_price = entry_price - stop_dist
            risk_amount = equity * risk_pct / 100.0
            qty = risk_amount / stop_dist if stop_dist > 0 else 0.0
            if qty > 0:
                in_position = True
                entry_bar = i
                entry_comm = abs(qty * entry_price) * commission / 100.0
                equity -= entry_comm
                equity_curve[i] = equity

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        final_equity=equity,
        total_return_pct=(equity / risk_config.initial_capital - 1.0) * 100.0,
        mode="simple",
    )


# ── Core FSM evaluation (inline, no external deps) ───────────────────────────

def _evaluate_bar(
    snapshot: PositionSnapshot,
    bar: dict[str, float],
    bar_index: int,
    *,
    entry_signal: bool = False,
    entry_direction: str = "long",
    exit_signal: bool = False,
    trailing_signal: bool = False,
    entry_size: float = 0.0,
    atr_value: float | None = None,
    stop_override: float | None = None,
    risk_config: SuiteRiskConfig,
) -> dict[str, Any]:
    """Evaluate one bar against the position snapshot (fixed evaluation order).

    Priority order (immutable contract):
        1. Stop-loss
        2. Partial TP (TP1)
        3. Break-even
        4. Trailing exit
        5. Time exit
        6. New entry / pyramid add
    """
    snap = deepcopy(snapshot)
    orders: list[dict[str, Any]] = []
    reason: str | None = None

    if snap.state not in (PositionState.FLAT, PositionState.CLOSED):
        snap.bars_in_position += 1
        snap.unrealized_pnl = _calc_unrealized(snap, bar["close"])

    # Priority 1: Stop-loss
    if _should_stop_loss(snap, bar):
        snap, reason, fill, qty = _apply_stop_loss(snap, bar, risk_config)
        orders.append(_close_order(reason, fill, qty))
        return {"snapshot": snap, "reason": reason, "orders": orders}

    # Priority 2: Partial TP1
    if _should_take_profit_1(snap, bar, exit_signal, risk_config):
        snap, reason, tp_order = _apply_take_profit_1(snap, bar, bar_index, risk_config)
        orders.append(tp_order)

    # Priority 3: Break-even
    be_result = _should_break_even(snap, bar, risk_config)
    if be_result is not None:
        snap, be_reason, be_fill, be_qty = be_result
        orders.append(_close_order(be_reason, be_fill, be_qty))
        return {"snapshot": snap, "reason": be_reason, "orders": orders}

    # Priority 4: Trailing exit
    if _should_trailing_exit(snap, bar, trailing_signal, bar_index):
        snap, reason, fill, qty = _apply_trailing_exit(snap, bar, risk_config)
        orders.append(_close_order(reason, fill, qty))
        return {"snapshot": snap, "reason": reason, "orders": orders}

    # Priority 5: Time exit
    if _should_time_exit(snap, risk_config):
        snap, reason, fill, qty = _apply_time_exit(snap, bar)
        orders.append(_close_order(reason, fill, qty))
        return {"snapshot": snap, "reason": reason, "orders": orders}

    # Priority 6: Entry / Pyramid
    if entry_signal and _can_enter(snap, bar, bar_index, entry_direction, risk_config):
        snap, entry_order = _apply_entry(
            snap, bar, bar_index, entry_direction, entry_size, stop_override, risk_config,
        )
        orders.append(entry_order)

    return {"snapshot": snap, "reason": reason, "orders": orders}


def _should_stop_loss(snap: PositionSnapshot, bar: dict[str, float]) -> bool:
    if snap.state in (PositionState.FLAT, PositionState.CLOSED):
        return False
    if snap.tp1_hit:
        return False
    if snap.stop_price is None:
        return False
    if snap.direction == "long":
        return bar["low"] <= snap.stop_price
    return bar["high"] >= snap.stop_price


def _apply_stop_loss(
    snap: PositionSnapshot,
    bar: dict[str, float],
    cfg: SuiteRiskConfig,
) -> tuple[PositionSnapshot, str, float, float]:
    if snap.stop_price is not None:
        fill = min(snap.stop_price, bar["open"]) if snap.direction == "long" else max(snap.stop_price, bar["open"])
    else:
        fill = bar["close"]
    fill = _slippage_adjust(fill, snap.direction, cfg.slippage_pct)
    original_qty = snap.quantity
    pnl = _fill_pnl(snap, fill, original_qty)
    snap = replace(
        snap,
        state=PositionState.CLOSED,
        realized_pnl=snap.realized_pnl + pnl,
        unrealized_pnl=0.0,
        quantity=0.0,
    )
    direction_label = "L" if snap.direction == "long" else "S"
    return snap, f"SL {direction_label}", fill, original_qty


def _should_take_profit_1(
    snap: PositionSnapshot,
    bar: dict[str, float],
    exit_signal: bool,
    cfg: SuiteRiskConfig,
) -> bool:
    if snap.state in (PositionState.FLAT, PositionState.CLOSED):
        return False
    if snap.tp1_hit:
        return False
    if not cfg.partial_tp.enabled:
        return False
    trigger = cfg.partial_tp.trigger
    if trigger == "signal":
        if not exit_signal:
            return False
        return _is_in_profit(snap, bar["close"], cfg.partial_tp.profit_distance_factor)
    if trigger == "r_multiple":
        return _check_r_multiple_tp1(snap, bar, cfg)
    return _is_in_profit(snap, bar["close"], cfg.partial_tp.profit_distance_factor)


def _check_r_multiple_tp1(snap: PositionSnapshot, bar: dict[str, float], cfg: SuiteRiskConfig) -> bool:
    if snap.stop_price is None or snap.quantity == 0:
        return False
    stop_dist = abs(snap.avg_entry_price - snap.stop_price)
    if stop_dist == 0:
        return False
    tp1_dist = stop_dist * cfg.partial_tp.r_multiple
    if snap.direction == "long":
        return bar["high"] >= snap.avg_entry_price + tp1_dist
    return bar["low"] <= snap.avg_entry_price - tp1_dist


def _apply_take_profit_1(
    snap: PositionSnapshot,
    bar: dict[str, float],
    bar_index: int,
    cfg: SuiteRiskConfig,
) -> tuple[PositionSnapshot, str, dict[str, Any]]:
    close_qty = snap.quantity * cfg.partial_tp.close_pct / 100.0
    remaining = snap.quantity - close_qty

    if cfg.partial_tp.trigger == "r_multiple" and snap.stop_price is not None:
        stop_dist = abs(snap.avg_entry_price - snap.stop_price)
        tp1_dist = stop_dist * cfg.partial_tp.r_multiple
        if snap.direction == "long":
            tp1_target = snap.avg_entry_price + tp1_dist
        else:
            tp1_target = snap.avg_entry_price - tp1_dist
        fill = _slippage_adjust(tp1_target, snap.direction, cfg.slippage_pct)
    else:
        fill = _slippage_adjust(bar["close"], snap.direction, cfg.slippage_pct)

    pnl = _fill_pnl(snap, fill, close_qty)

    if snap.direction == "long":
        be_price = snap.avg_entry_price * cfg.break_even.buffer
    else:
        be_price = snap.avg_entry_price / cfg.break_even.buffer

    snap = replace(
        snap,
        state=PositionState.PARTIALLY_CLOSED,
        quantity=remaining,
        realized_pnl=snap.realized_pnl + pnl,
        tp1_hit=True,
        tp1_bar_index=bar_index,
        stop_price=be_price,
        break_even_price=be_price,
    )
    direction_label = "L" if snap.direction == "long" else "S"
    order = {
        "action": "close_partial",
        "quantity": close_qty,
        "filled_qty": close_qty,
        "price": fill,
        "reason": f"TP1 {direction_label}",
    }
    return snap, f"TP1 {direction_label}", order


def _should_break_even(
    snap: PositionSnapshot,
    bar: dict[str, float],
    cfg: SuiteRiskConfig,
) -> tuple[PositionSnapshot, str, float, float] | None:
    if snap.state in (PositionState.FLAT, PositionState.CLOSED):
        return None
    if not cfg.break_even.enabled:
        return None

    activation = cfg.break_even.activation

    if activation == "after_tp1":
        if not snap.tp1_hit:
            return None
        if snap.break_even_price is None:
            return None
    elif activation == "r_multiple":
        if snap.break_even_price is None:
            if snap.stop_price is None or snap.quantity <= 0:
                return None
            r_unit = abs(snap.avg_entry_price - snap.stop_price)
            if r_unit == 0:
                return None
            unrealized_r = snap.unrealized_pnl / (r_unit * snap.quantity)
            if unrealized_r < cfg.break_even.r_multiple:
                return None
        # fall through to hit-check

    if snap.break_even_price is None:
        if snap.direction == "long":
            snap = replace(snap, break_even_price=snap.avg_entry_price * cfg.break_even.buffer)
        else:
            snap = replace(snap, break_even_price=snap.avg_entry_price / cfg.break_even.buffer)

    hit = False
    if snap.direction == "long" and bar["low"] <= snap.break_even_price:
        hit = True
    elif snap.direction == "short" and bar["high"] >= snap.break_even_price:
        hit = True

    if hit:
        original_qty = snap.quantity
        fill = _slippage_adjust(snap.break_even_price, snap.direction, cfg.slippage_pct)
        pnl = _fill_pnl(snap, fill, original_qty)
        snap = replace(
            snap,
            state=PositionState.CLOSED,
            realized_pnl=snap.realized_pnl + pnl,
            unrealized_pnl=0.0,
            quantity=0.0,
        )
        direction_label = "L" if snap.direction == "long" else "S"
        return snap, f"BE {direction_label}", fill, original_qty

    if snap.state != PositionState.OPEN_BREAKEVEN:
        snap = replace(snap, state=PositionState.OPEN_BREAKEVEN)
    return None


def _should_trailing_exit(
    snap: PositionSnapshot,
    bar: dict[str, float],
    trailing_signal: bool,
    bar_index: int,
) -> bool:
    if snap.state in (PositionState.FLAT, PositionState.CLOSED):
        return False
    if not trailing_signal:
        return False
    if snap.tp1_hit:
        if snap.tp1_bar_index is not None and bar_index <= snap.tp1_bar_index:
            return False
        return _is_in_profit_simple(snap, bar["close"])
    return True


def _apply_trailing_exit(
    snap: PositionSnapshot,
    bar: dict[str, float],
    cfg: SuiteRiskConfig,
) -> tuple[PositionSnapshot, str, float, float]:
    original_qty = snap.quantity
    fill = _slippage_adjust(bar["close"], snap.direction, cfg.slippage_pct)
    pnl = _fill_pnl(snap, fill, original_qty)
    snap = replace(
        snap,
        state=PositionState.CLOSED,
        realized_pnl=snap.realized_pnl + pnl,
        unrealized_pnl=0.0,
        quantity=0.0,
    )
    direction_label = "L" if snap.direction == "long" else "S"
    return snap, f"Trail {direction_label}", fill, original_qty


def _should_time_exit(snap: PositionSnapshot, cfg: SuiteRiskConfig) -> bool:
    if not cfg.time_exit.enabled:
        return False
    if snap.state in (PositionState.FLAT, PositionState.CLOSED):
        return False
    return snap.bars_in_position >= cfg.time_exit.max_bars


def _apply_time_exit(
    snap: PositionSnapshot,
    bar: dict[str, float],
) -> tuple[PositionSnapshot, str, float, float]:
    original_qty = snap.quantity
    fill = bar["close"]
    pnl = _fill_pnl(snap, fill, original_qty)
    snap = replace(
        snap,
        state=PositionState.CLOSED,
        realized_pnl=snap.realized_pnl + pnl,
        unrealized_pnl=0.0,
        quantity=0.0,
    )
    return snap, "Time exit", fill, original_qty


def _can_enter(
    snap: PositionSnapshot,
    bar: dict[str, float],
    bar_index: int,
    direction: str,
    cfg: SuiteRiskConfig,
) -> bool:
    if snap.last_order_bar_index is not None:
        if bar_index - snap.last_order_bar_index <= cfg.pyramid.block_bars:
            return False

    if snap.state == PositionState.FLAT:
        return True

    if snap.state == PositionState.CLOSED:
        return False

    if not cfg.pyramid.enabled:
        return False
    if snap.direction != direction:
        return False
    if snap.pyramid_level >= cfg.pyramid.max_adds:
        return False
    remaining = cfg.pyramid.max_adds - snap.pyramid_level
    if remaining <= 0:
        return False
    if snap.stop_price is not None:
        threshold_dist = (
            abs(snap.stop_price - snap.avg_entry_price)
            / remaining
            * cfg.pyramid.threshold_factor
        )
        if direction == "long":
            return bar["close"] <= snap.avg_entry_price - threshold_dist
        return bar["close"] >= snap.avg_entry_price + threshold_dist
    return False


def _apply_entry(
    snap: PositionSnapshot,
    bar: dict[str, float],
    bar_index: int,
    direction: str,
    size: float,
    stop_override: float | None,
    cfg: SuiteRiskConfig,
) -> tuple[PositionSnapshot, dict[str, Any]]:
    is_pyramid = snap.state not in (PositionState.FLAT, PositionState.CLOSED)
    price = bar["close"]

    if is_pyramid:
        new_qty = snap.quantity + size
        new_avg = (snap.avg_entry_price * snap.quantity + price * size) / new_qty
        snap = replace(
            snap,
            state=PositionState.OPEN_PYRAMIDED,
            quantity=new_qty,
            avg_entry_price=new_avg,
            pyramid_level=snap.pyramid_level + 1,
            last_order_bar_index=bar_index,
        )
        reason = f"Pyramid L{snap.pyramid_level}"
    else:
        snap = replace(
            snap,
            state=PositionState.OPEN_INITIAL,
            direction=direction,
            quantity=size,
            avg_entry_price=price,
            stop_price=stop_override,
            break_even_price=None,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            pyramid_level=0,
            tp1_hit=False,
            tp1_bar_index=None,
            entry_bar_index=bar_index,
            last_order_bar_index=bar_index,
            bars_in_position=0,
        )
        reason = f"Entry {direction}"

    order = {
        "action": "entry" if not is_pyramid else "pyramid_add",
        "direction": direction,
        "quantity": size,
        "filled_qty": size,
        "price": price,
        "reason": reason,
    }
    return snap, order


def _reset_snapshot(snap: PositionSnapshot) -> PositionSnapshot:
    return replace(
        snap,
        state=PositionState.FLAT,
        direction="flat",
        quantity=0.0,
        avg_entry_price=0.0,
        stop_price=None,
        break_even_price=None,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        pyramid_level=0,
        tp1_hit=False,
        tp1_bar_index=None,
        entry_bar_index=None,
        last_order_bar_index=None,
        bars_in_position=0,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slippage_adjust(price: float, direction: str, slippage_pct: float) -> float:
    if slippage_pct == 0.0:
        return price
    if direction == "long":
        return price * (1 - slippage_pct / 100.0)
    return price * (1 + slippage_pct / 100.0)


def _calc_unrealized(snap: PositionSnapshot, current_price: float) -> float:
    if snap.quantity == 0:
        return 0.0
    if snap.direction == "long":
        return (current_price - snap.avg_entry_price) * snap.quantity
    return (snap.avg_entry_price - current_price) * snap.quantity


def _fill_pnl(snap: PositionSnapshot, fill_price: float, qty: float) -> float:
    if snap.direction == "long":
        return (fill_price - snap.avg_entry_price) * qty
    return (snap.avg_entry_price - fill_price) * qty


def _is_in_profit(snap: PositionSnapshot, price: float, distance_factor: float) -> bool:
    if snap.quantity == 0:
        return False
    if snap.direction == "long":
        return price > snap.avg_entry_price and (
            abs(price - snap.avg_entry_price) >= abs(snap.avg_entry_price) * (distance_factor - 1)
        )
    return price < snap.avg_entry_price and (
        abs(snap.avg_entry_price - price) >= abs(snap.avg_entry_price) * (distance_factor - 1)
    )


def _is_in_profit_simple(snap: PositionSnapshot, price: float) -> bool:
    if snap.direction == "long":
        return price > snap.avg_entry_price
    return price < snap.avg_entry_price


def _close_order(reason: str | None, fill_price: float = 0.0, original_qty: float = 0.0) -> dict[str, Any]:
    return {
        "action": "close_all",
        "quantity": original_qty,
        "filled_qty": original_qty,
        "price": fill_price,
        "reason": reason,
    }


def _compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute ATR using Wilder's smoothing (no TA-Lib dependency)."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.empty(n)
    atr[:period] = 0.0
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ── Public Engine class ───────────────────────────────────────────────────────

class SuiteEngine:
    """Orchestrates single and batch backtesting runs using SuiteTrading double-runner.

    Stateless — each ``run`` call is independent and deterministic.

    Parameters
    ----------
    mode
        ``"auto"`` selects based on archetype vectorizability.
        ``"fsm"`` forces full state-machine loop.
        ``"simple"`` forces lightweight bar loop.
    """

    def __init__(self, mode: str = "auto") -> None:
        if mode not in ("fsm", "simple", "auto"):
            raise ValueError(f"Invalid mode {mode!r}. Choose from ('fsm', 'simple', 'auto')")
        self._mode = mode

    def run(
        self,
        *,
        dataset: BacktestDataset,
        signals: StrategySignals,
        risk_config: SuiteRiskConfig,
        direction: str = "long",
    ) -> dict[str, Any]:
        """Execute a single backtest and return results dict."""
        effective_mode = self._mode
        if effective_mode == "auto":
            effective_mode = "simple"  # default: prefer simple for speed

        if effective_mode == "simple":
            result = run_simple_backtest(
                dataset=dataset,
                signals=signals,
                risk_config=risk_config,
            )
        else:
            result = run_fsm_backtest(
                dataset=dataset,
                signals=signals,
                risk_config=risk_config,
                direction=direction,
            )

        return _result_to_dict(result, dataset, effective_mode)


def _result_to_dict(result: BacktestResult, dataset: BacktestDataset, mode: str) -> dict[str, Any]:
    """Flatten BacktestResult into a serialisable dict."""
    trades_df = pd.DataFrame([
        {
            "entry_bar": t.entry_bar,
            "exit_bar": t.exit_bar,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl": t.pnl,
            "commission": t.commission,
            "exit_reason": t.exit_reason,
        }
        for t in result.trades
    ]) if result.trades else pd.DataFrame()

    return {
        "symbol": dataset.symbol,
        "timeframe": dataset.base_timeframe,
        "mode": mode,
        "equity_curve": result.equity_curve,
        "trades": trades_df,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "total_trades": len(result.trades),
    }
