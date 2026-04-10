"""Unified Risk Manager — Centralized risk management layer for all strategies.

Provides a single interface for:
- Entry/exit risk checks
- Position sizing (Kelly, fractional, ATR-based)
- Stop loss management (fixed, trailing, ATR, time-based)
- Drawdown controls (account-level and position-level)
- Correlation risk (prevent correlated positions)
- Sector limits
- Volatility targeting
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

# Import existing risk components
from .manager import RiskManager, RiskConfig, RiskLevel
from .volatility import (
    VolatilityRegime,
    detect_regime,
    regime_multipliers,
    volatility_adjusted_position,
)
from .advanced_metrics import AdvancedRiskMetrics
from .position_sizer import (
    PositionSizingMethod,
    KellySizer,
    FractionalSizer,
    ATRSizer,
    VolatilitySizer,
    FixedRatioSizer,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StopLossConfig:
    """Stop loss configuration."""
    method: str = "fixed"  # fixed, trailing, atr, time
    fixed_pct: float = 0.02  # 2% fixed stop
    trailing_pct: float = 0.03  # 3% trailing stop
    atr_multiplier: float = 2.0  # ATR multiplier for ATR stops
    time_bars: int = 20  # Bars to hold before time-based exit


@dataclass
class SectorLimit:
    """Sector position limit."""
    sector: str
    max_positions: int = 3
    max_exposure: float = 0.20  # 20% of portfolio


@dataclass
class Position:
    """Represents a current position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: str = "long"  # long or short
    sector: Optional[str] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None
    peak_price: Optional[float] = None
    entry_time: Optional[int] = None


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool
    reason: str
    severity: str = "info"  # info, warning, error, critical


@dataclass
class ExitResult:
    """Result of an exit check."""
    should_exit: bool
    reason: str
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# Unified Risk Manager
# ---------------------------------------------------------------------------

class UnifiedRiskManager:
    """Unified risk management interface for all trading strategies.

    Provides centralized risk controls including:
    - Entry/exit risk checks
    - Multiple position sizing methods
    - Stop loss management (fixed, trailing, ATR, time-based)
    - Account and position drawdown controls
    - Correlation-based position limits
    - Sector exposure limits
    - Volatility-adjusted position sizing
    """

    def __init__(
        self,
        account_balance: float = 100000.0,
        max_portfolio_exposure: float = 0.80,
        max_sector_exposure: float = 0.30,
        max_drawdown_pct: float = 0.15,
        position_drawdown_pct: float = 0.08,
        correlation_threshold: float = 0.70,
        stop_loss_config: Optional[StopLossConfig] = None,
        sector_limits: Optional[List[SectorLimit]] = None,
        position_sizing_method: PositionSizingMethod = PositionSizingMethod.ATR_BASED,
    ):
        """
        Initialize the Unified Risk Manager.

        Args:
            account_balance: Starting account balance
            max_portfolio_exposure: Maximum total portfolio exposure (0.0-1.0)
            max_sector_exposure: Maximum exposure per sector
            max_drawdown_pct: Maximum account drawdown before stopping trading
            position_drawdown_pct: Maximum drawdown per position before exit
            correlation_threshold: Correlation threshold for position limits
            stop_loss_config: Stop loss configuration
            sector_limits: List of sector position limits
            position_sizing_method: Default position sizing method
        """
        self.account_balance = account_balance
        self.peak_balance = account_balance
        self.initial_balance = account_balance

        # Portfolio limits
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown_pct = max_drawdown_pct
        self.position_drawdown_pct = position_drawdown_pct

        # Correlation settings
        self.correlation_threshold = correlation_threshold

        # Stop loss config
        self.stop_loss_config = stop_loss_config or StopLossConfig()

        # Sector limits
        self.sector_limits = sector_limits or []
        self._sector_map: Dict[str, SectorLimit] = {
            s.sector: s for s in self.sector_limits
        }

        # Position sizing
        self.position_sizing_method = position_sizing_method
        self._position_sizers = {
            PositionSizingMethod.KELLY: KellySizer(fractional=0.5),
            PositionSizingMethod.FRACTIONAL: FractionalSizer(risk_pct=0.02),
            PositionSizingMethod.ATR_BASED: ATRSizer(risk_pct=0.02, atr_multiplier=2.0),
            PositionSizingMethod.VOLATILITY_ADJUSTED: VolatilitySizer(
                base_position=10000, target_vol=0.15
            ),
            PositionSizingMethod.FIXED_RATIO: FixedRatioSizer(delta=0.5),
        }

        # Current state
        self.positions: Dict[str, Position] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.max_daily_trades: int = 20

        # Legacy RiskManager for backward compatibility
        self._legacy_risk = RiskManager()

        # Performance tracking
        self.equity_curve: List[float] = [account_balance]
        self.trade_history: List[Dict] = []

    def check_entry(
        self,
        symbol: str,
        signal: Dict[str, Any],
        position_size: float,
    ) -> Tuple[bool, str]:
        """
        Check if a new entry is allowed based on risk rules.

        Args:
            symbol: Trading symbol
            signal: Signal dict with entry_price, side, sector, atr, etc.
            position_size: Proposed position size in currency units

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check account-level drawdown
        if self.account_balance < self.peak_balance * (1 - self.max_drawdown_pct):
            return False, f"Account drawdown exceeded {self.max_drawdown_pct*100}%"

        # Check total portfolio exposure
        total_exposure = sum(
            p.quantity * p.current_price for p in self.positions.values()
        )
        new_exposure = total_exposure + position_size
        max_exposure = self.account_balance * self.max_portfolio_exposure
        if new_exposure > max_exposure:
            return False, f"Portfolio exposure would exceed {self.max_portfolio_exposure*100}% limit"

        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit ({self.max_daily_trades}) reached"

        # Check sector limits
        sector = signal.get("sector")
        if sector:
            sector_exposure = sum(
                p.quantity * p.current_price
                for p in self.positions.values()
                if p.sector == sector
            )
            sector_limit = self._sector_map.get(sector)
            if sector_limit:
                if sector_exposure + position_size > self.account_balance * sector_limit.max_exposure:
                    return False, f"Sector {sector} exposure limit would be exceeded"
                sector_positions = len([p for p in self.positions.values() if p.sector == sector])
                if sector_positions >= sector_limit.max_positions:
                    return False, f"Sector {sector} position limit reached"

        # Check correlation limits
        for existing_symbol, existing_pos in self.positions.items():
            corr_key = tuple(sorted([symbol, existing_symbol]))
            corr = self.correlation_matrix.get(corr_key, 0.0)
            if abs(corr) >= self.correlation_threshold:
                if existing_pos.side == signal.get("side", "long"):
                    return False, f"Correlation with {existing_symbol} ({corr:.2f}) exceeds threshold"

        # Check position drawdown limit
        if symbol in self.positions:
            pos = self.positions[symbol]
            drawdown = (pos.current_price - pos.entry_price) / pos.entry_price
            if pos.side == "long" and drawdown < -self.position_drawdown_pct:
                return False, f"Position {symbol} drawdown {drawdown*100:.1f}% exceeds limit"

        return True, "Entry allowed"

    def check_exit(
        self,
        symbol: str,
        position: Optional[Position] = None,
    ) -> Tuple[bool, str, float]:
        """
        Check if an exit is required based on risk rules.

        Args:
            symbol: Trading symbol
            position: Position object (if None, looks up from self.positions)

        Returns:
            Tuple of (should_exit: bool, reason: str, exit_price: float)
        """
        if position is None:
            position = self.positions.get(symbol)

        if position is None:
            return False, "No position found", 0.0

        current_price = position.current_price
        entry_price = position.entry_price
        side = position.side

        # Calculate current PnL percentage
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Check fixed stop loss
        if self.stop_loss_config.method == "fixed":
            stop_pct = self.stop_loss_config.fixed_pct
            if side == "long" and current_price <= entry_price * (1 - stop_pct):
                return True, f"Fixed stop loss triggered ({stop_pct*100}%)", current_price
            elif side == "short" and current_price >= entry_price * (1 + stop_pct):
                return True, f"Fixed stop loss triggered ({stop_pct*100}%)", current_price

        # Check trailing stop
        if self.stop_loss_config.method == "trailing":
            trailing_pct = self.stop_loss_config.trailing_pct
            peak = position.peak_price or entry_price
            if side == "long":
                if current_price > peak:
                    position.peak_price = current_price
                trailing_stop = peak * (1 - trailing_pct)
                if current_price <= trailing_stop:
                    return True, f"Trailing stop triggered ({trailing_pct*100}%)", current_price
            else:
                if current_price < peak:
                    position.peak_price = current_price
                trailing_stop = peak * (1 + trailing_pct)
                if current_price >= trailing_stop:
                    return True, f"Trailing stop triggered ({trailing_pct*100}%)", current_price

        # Check ATR-based stop
        if self.stop_loss_config.method == "atr":
            atr = getattr(position, "atr", None)
            if atr:
                atr_multiplier = self.stop_loss_config.atr_multiplier
                if side == "long":
                    atr_stop = entry_price - (atr * atr_multiplier)
                    if current_price <= atr_stop:
                        return True, f"ATR stop triggered", current_price
                else:
                    atr_stop = entry_price + (atr * atr_multiplier)
                    if current_price >= atr_stop:
                        return True, f"ATR stop triggered", current_price

        # Check time-based exit
        if self.stop_loss_config.method == "time" and position.entry_time:
            bars_held = (int(time.time() * 1000) - position.entry_time) // (60000 * 5)  # 5-min bars
            if bars_held >= self.stop_loss_config.time_bars:
                return True, f"Time-based exit after {bars_held} bars", current_price

        # Check position drawdown limit
        if pnl_pct <= -self.position_drawdown_pct:
            return True, f"Position drawdown {pnl_pct*100:.1f}% exceeds limit", current_price

        # Check take profit (inverse of stop loss)
        take_profit_pct = self.stop_loss_config.fixed_pct * 2.5  # Default 5% take profit
        if pnl_pct >= take_profit_pct:
            return True, f"Take profit target reached ({pnl_pct*100:.1f}%)", current_price

        return False, "Hold position", current_price

    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        account_balance: Optional[float] = None,
    ) -> float:
        """
        Calculate recommended position size based on selected method.

        Args:
            signal: Signal dict with entry_price, atr, volatility, etc.
            account_balance: Account balance (uses self.account_balance if None)

        Returns:
            Recommended position size in currency units
        """
        if account_balance is None:
            account_balance = self.account_balance

        entry_price = signal.get("entry_price", 1.0)
        atr = signal.get("atr")
        volatility = signal.get("volatility", 0.15)
        regime = signal.get("regime")

        method = self.position_sizing_method

        if method == PositionSizingMethod.ATR_BASED and atr:
            sizer = self._position_sizers[PositionSizingMethod.ATR_BASED]
            return sizer.calculate(account_balance, atr, entry_price)

        elif method == PositionSizingMethod.KELLY:
            sizer = self._position_sizers[PositionSizingMethod.KELLY]
            return sizer.calculate(
                account_balance,
                win_rate=signal.get("win_rate", 0.55),
                avg_win=signal.get("avg_win", 100),
                avg_loss=signal.get("avg_loss", 80),
            )

        elif method == PositionSizingMethod.FRACTIONAL:
            sizer = self._position_sizers[PositionSizingMethod.FRACTIONAL]
            stop_price = signal.get("stop_loss_price", entry_price * 0.98)
            return sizer.calculate(account_balance, entry_price, stop_price)

        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            sizer = self._position_sizers[PositionSizingMethod.VOLATILITY_ADJUSTED]
            regime_mult = regime_multipliers(regime) if regime else 1.0
            return sizer.calculate(volatility, regime_mult)

        elif method == PositionSizingMethod.FIXED_RATIO:
            sizer = self._position_sizers[PositionSizingMethod.FIXED_RATIO]
            return sizer.calculate(account_balance) * 1000  # Unit value

        # Default: fixed 2% risk
        return account_balance * 0.02 / 0.02  # 2% of balance with 2% stop

    def get_portfolio_risk(self) -> Dict[str, Any]:
        """
        Get current portfolio risk metrics.

        Returns:
            Dict with portfolio risk metrics
        """
        total_value = sum(
            p.quantity * p.current_price for p in self.positions.values()
        )
        exposure_pct = total_value / self.account_balance if self.account_balance > 0 else 0

        # Calculate sector exposures
        sector_exposures: Dict[str, float] = {}
        for pos in self.positions.values():
            if pos.sector:
                sector_exposures[pos.sector] = sector_exposures.get(pos.sector, 0) + (
                    pos.quantity * pos.current_price
                )

        # Calculate current drawdown
        current_drawdown = (
            (self.peak_balance - self.account_balance) / self.peak_balance
            if self.peak_balance > 0 else 0.0
        )

        # Position-level drawdowns
        position_drawdowns = {}
        for symbol, pos in self.positions.items():
            if pos.side == "long":
                dd = (pos.current_price - pos.entry_price) / pos.entry_price
            else:
                dd = (pos.entry_price - pos.current_price) / pos.entry_price
            position_drawdowns[symbol] = dd

        return {
            "total_exposure": total_value,
            "exposure_pct": exposure_pct,
            "cash": self.account_balance - total_value,
            "cash_pct": 1.0 - exposure_pct,
            "peak_balance": self.peak_balance,
            "current_drawdown": current_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sector_exposures": sector_exposures,
            "position_count": len(self.positions),
            "position_drawdowns": position_drawdowns,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "max_daily_trades": self.max_daily_trades,
        }

    def check_drawdown(
        self,
        account_balance: Optional[float] = None,
        peak_balance: Optional[float] = None,
    ) -> bool:
        """
        Check if account drawdown is within limits.

        Args:
            account_balance: Current balance (uses self.account_balance if None)
            peak_balance: Peak balance (uses self.peak_balance if None)

        Returns:
            True if drawdown is within limits, False otherwise
        """
        if account_balance is None:
            account_balance = self.account_balance
        if peak_balance is None:
            peak_balance = self.peak_balance

        if peak_balance <= 0:
            return True

        drawdown = (peak_balance - account_balance) / peak_balance
        return drawdown <= self.max_drawdown_pct

    def update_position(
        self,
        symbol: str,
        current_price: float,
        atr: Optional[float] = None,
    ) -> None:
        """Update an existing position with current price."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        pos.current_price = current_price
        if atr:
            pos.atr = atr

        # Update peak price for trailing stops
        if pos.side == "long":
            if pos.peak_price is None or current_price > pos.peak_price:
                pos.peak_price = current_price
        else:
            if pos.peak_price is None or current_price < pos.peak_price:
                pos.peak_price = current_price

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        sector: Optional[str] = None,
        atr: Optional[float] = None,
    ) -> bool:
        """Open a new position."""
        signal = {
            "side": side,
            "sector": sector,
            "atr": atr,
            "entry_price": entry_price,
        }
        position_value = quantity * entry_price

        allowed, reason = self.check_entry(symbol, signal, position_value)
        if not allowed:
            return False

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            side=side,
            sector=sector,
            atr=atr,
            peak_price=entry_price,
            entry_time=int(time.time() * 1000),
        )

        self.daily_trades += 1
        return True

    def close_position(self, symbol: str, exit_price: Optional[float] = None) -> Optional[float]:
        """Close an existing position and return PnL."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        if exit_price is None:
            exit_price = pos.current_price

        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Update account balance
        self.account_balance += pnl
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

        # Record trade
        self.trade_history.append({
            "symbol": symbol,
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": exit_price,
            "quantity": pos.quantity,
            "pnl": pnl,
            "timestamp": int(time.time() * 1000),
        })

        self.daily_pnl += pnl
        self.equity_curve.append(self.account_balance)

        # Remove position
        del self.positions[symbol]

        return pnl

    def set_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Set correlation between two symbols."""
        key = tuple(sorted([symbol1, symbol2]))
        self.correlation_matrix[key] = correlation

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        key = tuple(sorted([symbol1, symbol2]))
        return self.correlation_matrix.get(key, 0.0)

    def set_sector_limit(self, sector: str, max_positions: int = 3, max_exposure: float = 0.20) -> None:
        """Set or update a sector limit."""
        limit = SectorLimit(sector=sector, max_positions=max_positions, max_exposure=max_exposure)
        self._sector_map[sector] = limit
        # Also update in the list
        for i, s in enumerate(self.sector_limits):
            if s.sector == sector:
                self.sector_limits[i] = limit
                return
        self.sector_limits.append(limit)

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_trades = 0
        self.daily_pnl = 0.0

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        portfolio_risk = self.get_portfolio_risk()

        # Count positions by sector
        sector_counts: Dict[str, int] = {}
        for pos in self.positions.values():
            if pos.sector:
                sector_counts[pos.sector] = sector_counts.get(pos.sector, 0) + 1

        return {
            "account_balance": self.account_balance,
            "peak_balance": self.peak_balance,
            "total_exposure": portfolio_risk["total_exposure"],
            "exposure_pct": portfolio_risk["exposure_pct"],
            "current_drawdown": portfolio_risk["current_drawdown"],
            "max_drawdown_pct": self.max_drawdown_pct,
            "drawdown_safe": self.check_drawdown(),
            "position_count": len(self.positions),
            "sector_counts": sector_counts,
            "sector_limits": {s.sector: {"max_positions": s.max_positions, "max_exposure": s.max_exposure}
                            for s in self.sector_limits},
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "daily_pnl": self.daily_pnl,
            "correlation_threshold": self.correlation_threshold,
            "stop_loss_method": self.stop_loss_config.method,
            "position_sizing_method": self.position_sizing_method.value,
        }


__all__ = [
    "UnifiedRiskManager",
    "StopLossConfig",
    "SectorLimit",
    "Position",
    "RiskCheckResult",
    "ExitResult",
]
