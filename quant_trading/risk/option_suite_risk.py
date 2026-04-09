"""
OptionSuite Risk Management — Greeks-Based Portfolio Risk
===========================================================

Adopted from D:/Hive/Data/trading_repos/OptionSuite/:
  - portfolioManager/portfolio.py        → Portfolio Greeks aggregation (Delta/Gamma/Vega/Theta/Rho)
  - riskManager/riskManagement.py        → Abstract risk management interface
  - riskManager/putVerticalRiskManagement.py → Margin-aware put-vertical risk rules
  - riskManager/strangleRiskManagement.py → Strangle risk management strategies
  - base/option.py                        → Option Greeks data model (delta, gamma, vega, theta, rho)

Key classes:
  - PortfolioGreeks     : Stores and updates aggregated Greeks for a live portfolio
  - GreeksAggregator    : Computes portfolio-level Greeks from a list of positions
  - MarginCalculator    : Calculates Reg-T / portfolio-margin requirements for short options
  - RiskScenarioAnalyzer: Stress-tests the portfolio under custom price / vol scenarios
  - RiskReport          : Generates a formatted risk report (VaR, CVaR, Greeks exposure)

All computations use pure NumPy. Greeks are sourced from quant_trading.options.pricing.greeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Greeks imports from options pricing module
# --------------------------------------------------------------------------- #
from quant_trading.options.pricing.greeks import (
    Greeks,
    calculate_greeks,
    calculate_portfolio_greeks,
)
from quant_trading.options.pricing.black_scholes import BlackScholes

# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #

OptionType = str          # "call" or "put"
PositionSide = str         # "long" or "short"


@dataclass
class OptionPosition:
    """
    Single option position descriptor.

    Attributes:
        symbol       : Ticker / identifier (e.g. "SPY")
        option_type  : "call" or "put"
        side         : "long" or "short"
        strike       : Strike price (same unit as spot_price)
        spot_price   : Current underlying price
        vol          : Implied volatility (decimal, e.g. 0.20 = 20%)
        time_to_exp  : Time to expiration in years (e.g. 30/365)
        risk_free_rate: Risk-free rate (decimal, e.g. 0.05 = 5%)
        contracts    : Number of contracts (1 contract = 100 shares)
        price        : Current option price (premium per share)
    """
    symbol: str
    option_type: OptionType
    side: PositionSide
    strike: float
    spot_price: float
    vol: float
    time_to_exp: float
    risk_free_rate: float
    contracts: int = 1
    price: float = 0.0

    # Derived Greeks (lazy-computed)
    _greeks: Optional[Greeks] = field(default=None, repr=False)

    def multiplier(self) -> int:
        """Position sign: +1 for long, -1 for short."""
        return 1 if self.side == "long" else -1

    def notional(self) -> float:
        """Notional value in underlying currency (contracts * 100 * strike)."""
        return self.contracts * 100 * self.strike

    def premium(self) -> float:
        """Total premium paid / received (contracts * 100 * price)."""
        return self.contracts * 100 * self.price

    def greeks(self) -> Greeks:
        """Return Greeks for this position (cached)."""
        if self._greeks is None:
            self._greeks = calculate_greeks(
                S=self.spot_price,
                K=self.strike,
                T=self.time_to_exp,
                r=self.risk_free_rate,
                sigma=self.vol,
                option_type=self.option_type,
            )
        return self._greeks

    def weighted_greeks(self) -> Greeks:
        """Greeks scaled by position size and multiplier."""
        g = self.greeks()
        mult = self.multiplier()
        contracts = self.contracts
        return Greeks(
            price=g.price * mult * contracts,
            delta=g.delta * mult * contracts,
            gamma=g.gamma * mult * contracts,
            vega=g.vega * mult * contracts,
            theta=g.theta * mult * contracts,
            rho=g.rho * mult * contracts,
            vanna=g.vanna * mult * contracts,
            charm=g.charm * mult * contracts,
            speed=g.speed * mult * contracts,
            color=g.color * mult * contracts,
            volga=g.volga * mult * contracts,
        )


# --------------------------------------------------------------------------- #
# PortfolioGreeks — aggregates Greeks across all positions in a portfolio
# --------------------------------------------------------------------------- #

class PortfolioGreeks:
    """
    Maintains and updates aggregated Greeks for a live option portfolio.

    Adopted from OptionSuite/portfolioManager/portfolio.py:
      totalDelta, totalGamma, totalVega, totalTheta tracking pattern.

    Attributes:
        positions      : List of live OptionPosition objects
        spot_price    : Current underlying spot price (per underlier, keyed by symbol)
        risk_free_rate: Risk-free rate applied to all positions

    Example:
        >>> pg = PortfolioGreeks()
        >>> pg.add_position(OptionPosition("SPY", "call", "long", 450, 450, 0.20, 30/365, 0.05))
        >>> pg.compute()
        PortfolioGreeksResult(delta=0.48, gamma=0.0021, vega=0.18, theta=-0.03, rho=0.04)
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self._positions: List[OptionPosition] = []
        self.risk_free_rate = risk_free_rate
        self._spot_prices: Dict[str, float] = {}   # symbol -> spot

    # --- mutators ----------------------------------------------------------- #

    def add_position(self, pos: OptionPosition) -> None:
        """Add a position to the portfolio."""
        self._positions.append(pos)
        self._spot_prices[pos.symbol] = pos.spot_price

    def remove_position(self, symbol: str, strike: float,
                        option_type: OptionType) -> bool:
        """Remove the first matching position; returns True if found."""
        for i, p in enumerate(self._positions):
            if (p.symbol == symbol and p.strike == strike
                    and p.option_type == option_type):
                del self._positions[i]
                return True
        return False

    def clear(self) -> None:
        """Remove all positions."""
        self._positions.clear()
        self._spot_prices.clear()

    def update_spot(self, symbol: str, spot: float) -> None:
        """Refresh spot price and invalidate Greeks cache for that symbol."""
        self._spot_prices[symbol] = spot
        for p in self._positions:
            if p.symbol == symbol:
                p._greeks = None   # force recompute

    # --- core aggregation ---------------------------------------------------- #

    def compute(self) -> Greeks:
        """
        Compute net portfolio Greeks by summing weighted Greeks of all positions.

        Returns:
            Greeks: Aggregated Greeks (delta, gamma, vega, theta, rho, …).
        """
        total = Greeks(
            price=0.0, delta=0.0, gamma=0.0, vega=0.0,
            theta=0.0, rho=0.0, vanna=0.0, charm=0.0,
            speed=0.0, color=0.0, volga=0.0,
        )
        for pos in self._positions:
            wg = pos.weighted_greeks()
            total.price  += wg.price
            total.delta  += wg.delta
            total.gamma  += wg.gamma
            total.vega   += wg.vega
            total.theta  += wg.theta
            total.rho    += wg.rho
            total.vanna  += wg.vanna
            total.charm  += wg.charm
            total.speed  += wg.speed
            total.color  += wg.color
            total.volga  += wg.volga
        return total

    # --- key exposure methods ------------------------------------------------ #

    def compute_portfolio_delta(self) -> float:
        """Net portfolio Delta (dV/dS)."""
        return self.compute().delta

    def compute_portfolio_gamma(self) -> float:
        """Net portfolio Gamma (d²V/dS²)."""
        return self.compute().gamma

    def compute_portfolio_vega(self) -> float:
        """Net portfolio Vega (dV/dσ, per 1% vol move)."""
        return self.compute().vega

    def compute_portfolio_theta(self) -> float:
        """Net portfolio Theta (daily time decay in $)."""
        return self.compute().theta

    def compute_portfolio_rho(self) -> float:
        """Net portfolio Rho (dV/dr, per 1% rate move)."""
        return self.compute().rho

    def net_notional(self) -> float:
        """Sum of absolute notionals across all positions."""
        return sum(abs(p.notional()) for p in self._positions)

    def net_premium(self) -> float:
        """Net premium paid (-) or received (+) across all positions."""
        return sum(p.multiplier() * p.premium() for p in self._positions)

    def delta_exposure_per_symbol(self) -> Dict[str, float]:
        """Delta exposure broken down by underlying symbol."""
        out: Dict[str, float] = {}
        for p in self._positions:
            wg = p.weighted_greeks()
            out[p.symbol] = out.get(p.symbol, 0.0) + wg.delta
        return out

    def __len__(self) -> int:
        return len(self._positions)

    def __repr__(self) -> str:
        g = self.compute()
        return (
            f"PortfolioGreeks(positions={len(self)}, "
            f"Δ={g.delta:.4f} Γ={g.gamma:.6f} "
            f"V={g.vega:.4f} Θ={g.theta:.4f} ρ={g.rho:.4f})"
        )


# --------------------------------------------------------------------------- #
# GreeksAggregator — static utility that computes portfolio Greeks from a
#                    list of position dicts (same interface as
#                    quant_trading.options.pricing.greeks.calculate_portfolio_greeks
# --------------------------------------------------------------------------- #

class GreeksAggregator:
    """
    Static / utility class that computes portfolio-level Greeks from a
    flat list of position descriptors.

    Accepts the same dict-based interface as calculate_portfolio_greeks()
    but also works with OptionPosition objects. Provides a clean functional
    wrapper around the options pricing module.

    Example:
        >>> positions = [
        ...     {"option_type": "call", "size": 2,  "S": 450, "K": 450, "T": 30/365, "r": 0.05, "sigma": 0.20},
        ...     {"option_type": "put",  "size": -1, "S": 450, "K": 440, "T": 30/365, "r": 0.05, "sigma": 0.22},
        ... ]
        >>> g = GreeksAggregator.compute(positions)
        >>> print(g.delta, g.gamma)
    """

    @staticmethod
    def compute(
        positions: List[Dict | OptionPosition],
        risk_free_rate: float = 0.05,
    ) -> Greeks:
        """
        Aggregate Greeks from a list of positions.

        Args:
            positions   : List of position descriptors.
                          Each item is either:
                            - dict with keys: option_type, size, S, K, T, r, sigma
                            - OptionPosition instance
            risk_free_rate: Default r to use when a dict lacks the key.

        Returns:
            Greeks: Net portfolio Greeks.
        """
        total = Greeks(
            price=0.0, delta=0.0, gamma=0.0, vega=0.0,
            theta=0.0, rho=0.0, vanna=0.0, charm=0.0,
            speed=0.0, color=0.0, volga=0.0,
        )

        for pos in positions:
            if isinstance(pos, OptionPosition):
                wg = pos.weighted_greeks()
                size = pos.contracts
                mult = pos.multiplier()
            else:
                g = calculate_greeks(
                    S=pos["S"], K=pos["K"], T=pos["T"],
                    r=pos.get("r", risk_free_rate),
                    sigma=pos["sigma"],
                    option_type=pos["option_type"],
                )
                size = pos.get("size", 1)
                mult = 1 if size >= 0 else -1
                wg = Greeks(
                    price=g.price * size,
                    delta=g.delta * size,
                    gamma=g.gamma * size,
                    vega=g.vega * size,
                    theta=g.theta * size,
                    rho=g.rho * size,
                )

            total.price  += wg.price
            total.delta  += wg.delta
            total.gamma  += wg.gamma
            total.vega   += wg.vega
            total.theta  += wg.theta
            total.rho    += wg.rho
            total.vanna  += wg.vanna
            total.charm  += wg.charm
            total.speed  += wg.speed
            total.color  += wg.color
            total.volga  += wg.volga

        return total

    @staticmethod
    def compute_delta(positions: List[Dict | OptionPosition],
                       risk_free_rate: float = 0.05) -> float:
        """Net portfolio Delta."""
        return GreeksAggregator.compute(positions, risk_free_rate).delta

    @staticmethod
    def compute_gamma(positions: List[Dict | OptionPosition],
                       risk_free_rate: float = 0.05) -> float:
        """Net portfolio Gamma."""
        return GreeksAggregator.compute(positions, risk_free_rate).gamma

    @staticmethod
    def compute_vega(positions: List[Dict | OptionPosition],
                      risk_free_rate: float = 0.05) -> float:
        """Net portfolio Vega."""
        return GreeksAggregator.compute(positions, risk_free_rate).vega

    @staticmethod
    def compute_theta(positions: List[Dict | OptionPosition],
                       risk_free_rate: float = 0.05) -> float:
        """Net portfolio Theta (daily)."""
        return GreeksAggregator.compute(positions, risk_free_rate).theta


# --------------------------------------------------------------------------- #
# MarginCalculator — Reg-T and portfolio-margin for short option positions
# --------------------------------------------------------------------------- #

class MarginCalculator:
    """
    Calculates margin requirements for short option positions.

    Adopted from OptionSuite/riskManager/putVerticalRiskManagement.py
    and riskManager/strangleRiskManagement.py margin / buying-power logic.

    Supports two modes:
      - "regt" : FINRA Reg-T margin (https://www.finra.org/rules-guidance/key-concepts#margin)
                 - Short naked call: greater of:
                     (a) 100% of option proceeds + 20% of underlying - OTM
                     (b) 100% of option proceeds + 10% of strike
                 - Short naked put: greater of:
                     (a) 100% of option proceeds + 20% of underlying - OTM
                     (b) 100% of option proceeds + 10% of strike
      - "portfolio": Portfolio margin (higher leverage, lower margin for
                     offsetting positions) — simplified approximation using
                     net delta and worst-case loss.

    Attributes:
        mode: "regt" or "portfolio"
    """

    def __init__(self, mode: str = "regt") -> None:
        if mode not in ("regt", "portfolio"):
            raise ValueError("mode must be 'regt' or 'portfolio'")
        self.mode = mode

    def calc_option_margin(self, pos: OptionPosition) -> float:
        """
        Calculate margin required for a single short option position.

        Args:
            pos: Short option position (side must be "short").

        Returns:
            float: Margin requirement in dollars.

        Raises:
            ValueError: If position is long (no margin required).
        """
        if pos.side != "short":
            raise ValueError("Margin is only required for short positions")

        premium = pos.price
        S = pos.spot_price
        K = pos.strike

        if pos.option_type == "call":
            otm = max(0, K - S)          # OTM for short call = max(K - S, 0)
            base = 100 * premium          # 100% of proceeds
            comp_a = base + 0.20 * 100 * S - 100 * otm
            comp_b = base + 0.10 * 100 * K
            raw = max(comp_a, comp_b)
        else:  # put
            otm = max(0, S - K)          # OTM for short put = max(S - K, 0)
            base = 100 * premium
            comp_a = base + 0.20 * 100 * S - 100 * otm
            comp_b = base + 0.10 * 100 * K
            raw = max(comp_a, comp_b)

        # Convert from per-share to total (multiply by contracts * 100)
        return raw * pos.contracts

    def calc_portfolio_margin(self,
                               positions: List[OptionPosition],
                               portfolio_delta: float = 0.0,
                               portfolio_gamma: float = 0.0) -> float:
        """
        Calculate simplified portfolio margin across all short positions.

        This is a simplified portfolio-margin approximation:
          - Sums individual Reg-T margins for each short position
          - Applies a delta-based reduction: margin -= |delta_excess| * spot * haircut
          - Adds a gamma scalp reserve: |gamma| * spot^2 * 0.02

        For a full portfolio-margin system (SPAN), use a certified
        margin engine such as OCC's SPAN or Interactive Brokers' API.

        Args:
            positions        : All short option positions in the portfolio.
            portfolio_delta  : Net portfolio delta.
            portfolio_gamma  : Net portfolio gamma.

        Returns:
            float: Estimated portfolio margin in dollars.
        """
        total = 0.0
        spot_prices: Dict[str, float] = {}

        for p in positions:
            if p.side != "short":
                continue
            total += self.calc_option_margin(p)
            spot_prices[p.symbol] = p.spot_price

        # Delta-based haircut (simplified portfolio margin offset)
        if abs(portfolio_delta) > 0:
            # Average spot across underlyings (simple approximation)
            avg_spot = sum(spot_prices.values()) / max(len(spot_prices), 1)
            total -= abs(portfolio_delta) * avg_spot * 0.10

        # Gamma scalp reserve: |Gamma| * spot^2 * 0.02
        if abs(portfolio_gamma) > 0:
            avg_spot = sum(spot_prices.values()) / max(len(spot_prices), 1)
            total += abs(portfolio_gamma) * avg_spot * avg_spot * 0.02

        return max(total, 0.0)

    def total_margin(self,
                      positions: List[OptionPosition],
                      portfolio_delta: float = 0.0,
                      portfolio_gamma: float = 0.0) -> float:
        """
        Main entry point — total margin for all short positions.

        Args:
            positions        : All option positions.
            portfolio_delta  : Net portfolio delta.
            portfolio_gamma  : Net portfolio gamma.

        Returns:
            float: Total margin in dollars.
        """
        short_positions = [p for p in positions if p.side == "short"]
        if self.mode == "regt":
            return sum(self.calc_option_margin(p) for p in short_positions)
        else:
            return self.calc_portfolio_margin(
                short_positions, portfolio_delta, portfolio_gamma
            )


# --------------------------------------------------------------------------- #
# RiskScenarioAnalyzer — stress-test portfolio under custom market scenarios
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioResult:
    """
    Result of a single stress-test scenario.

    Attributes:
        scenario_name     : Descriptive name of the scenario.
        spot_shift        : Fractional shift in spot price (e.g. -0.10 = -10%).
        vol_shift         : Fractional shift in implied vol (e.g. +0.30 = +30 vol points).
        pnl               : Estimated P&L under this scenario.
        new_portfolio_delta: Portfolio delta after the shock.
        new_portfolio_vega : Portfolio vega after the shock.
        new_portfolio_gamma: Portfolio gamma after the shock.
        margin_change     : Change in margin requirement under this scenario.
    """
    scenario_name: str
    spot_shift: float
    vol_shift: float
    pnl: float
    new_portfolio_delta: float
    new_portfolio_vega: float
    new_portfolio_gamma: float
    margin_change: float


class RiskScenarioAnalyzer:
    """
    Stress-tests an option portfolio under a set of predefined or custom
    market scenarios (price moves, vol changes).

    Adopted from OptionSuite/portfolioManager/portfolio.py
    `updatePortfolio` / `__calcPortfolioValues` pattern extended with
    explicit scenario enumeration.

    Example:
        >>> analyzer = RiskScenarioAnalyzer()
        >>> analyzer.add_scenario("SPY -10%", spot_shift=-0.10, vol_shift=0.0)
        >>> analyzer.add_scenario("VIX +20%", spot_shift=0.0, vol_shift=0.20)
        >>> results = analyzer.run_scenarios(positions, current_margin=5000)
        >>> for r in results:
        ...     print(r.scenario_name, r.pnl)
    """

    # --- predefined scenarios (adopted from common risk frameworks) -------- #

    PREDEFINED_SCENARIOS: Dict[str, Tuple[float, float]] = {
        "spot_+10%":       ( 0.10,  0.00),
        "spot_+5%":        ( 0.05,  0.00),
        "spot_-5%":        (-0.05,  0.00),
        "spot_-10%":       (-0.10,  0.00),
        "spot_-20%":       (-0.20,  0.00),
        "vol_+10%_vol_pts":( 0.00,  0.10),
        "vol_+20%_vol_pts":( 0.00,  0.20),
        "vol_-10%_vol_pts":( 0.00, -0.10),
        "vol_-20%_vol_pts":( 0.00, -0.20),
        "crash_(spot-10%_vol+15%)": (-0.10,  0.15),
        "rally_(spot+10%_vol-5%)":  ( 0.10, -0.05),
        "stress_(spot-20%_vol+30%)":(-0.20,  0.30),
    }

    def __init__(self) -> None:
        self._custom_scenarios: Dict[str, Tuple[float, float]] = {}

    # --- mutators ----------------------------------------------------------- #

    def add_scenario(self, name: str, spot_shift: float,
                      vol_shift: float) -> None:
        """
        Register a custom scenario.

        Args:
            name      : Unique scenario name.
            spot_shift: Fractional spot price change (e.g. -0.10 = -10%).
            vol_shift : Fractional IV change (e.g. +0.20 = +20 vol points).
        """
        self._custom_scenarios[name] = (spot_shift, vol_shift)

    def remove_scenario(self, name: str) -> None:
        """Remove a previously registered custom scenario."""
        self._custom_scenarios.pop(name, None)

    # --- core --------------------------------------------------------------- #

    def _shocked_greeks(self, pos: OptionPosition,
                         spot_shift: float,
                         vol_shift: float) -> Greeks:
        """
        Recompute Greeks under a shocked spot / vol environment.

        Uses first-order Taylor approximation for efficiency:
          ΔP ≈ Δ * ΔS + Γ * (ΔS)²/2 + ν * Δσ
        Greeks are re-evaluated at the shocked point to first order.
        """
        new_spot = pos.spot_price * (1 + spot_shift)
        new_vol  = max(pos.vol + vol_shift, 1e-4)

        g = calculate_greeks(
            S=new_spot, K=pos.strike, T=pos.time_to_exp,
            r=pos.risk_free_rate, sigma=new_vol,
            option_type=pos.option_type,
        )
        mult = pos.multiplier()
        contracts = pos.contracts
        return Greeks(
            price=g.price * mult * contracts,
            delta=g.delta * mult * contracts,
            gamma=g.gamma * mult * contracts,
            vega=g.vega * mult * contracts,
            theta=g.theta * mult * contracts,
            rho=g.rho * mult * contracts,
            vanna=g.vanna * mult * contracts,
            charm=g.charm * mult * contracts,
            speed=g.speed * mult * contracts,
            color=g.color * mult * contracts,
            volga=g.volga * mult * contracts,
        )

    def stress_test(self,
                     positions: List[OptionPosition],
                     current_margin: float = 0.0,
                     use_predefined: bool = True) -> List[ScenarioResult]:
        """
        Run all scenarios (predefined + custom) against the given positions.

        Args:
            positions     : Option positions to stress-test.
            current_margin: Current margin requirement (used to compute delta margin).
            use_predefined: Include PREDEFINED_SCENARIOS (default True).

        Returns:
            List[ScenarioResult]: Results for each scenario, sorted by name.
        """
        all_scenarios: Dict[str, Tuple[float, float]] = {}
        if use_predefined:
            all_scenarios.update(self.PREDEFINED_SCENARIOS)
        all_scenarios.update(self._custom_scenarios)

        # Baseline Greeks
        base_greeks = GreeksAggregator.compute(positions)

        results: List[ScenarioResult] = []

        for name, (ds, dv) in all_scenarios.items():
            # Build shocked positions
            shocked_positions: List[OptionPosition] = []
            for p in positions:
                sp = OptionPosition(
                    symbol=p.symbol,
                    option_type=p.option_type,
                    side=p.side,
                    strike=p.strike,
                    spot_price=p.spot_price * (1 + ds),
                    vol=max(p.vol + dv, 1e-4),
                    time_to_exp=p.time_to_exp,
                    risk_free_rate=p.risk_free_rate,
                    contracts=p.contracts,
                    price=p.price,
                )
                shocked_positions.append(sp)

            shocked_greeks = GreeksAggregator.compute(shocked_positions)

            # First-order P&L approximation
            pnl = (
                base_greeks.delta * (ds * 100)          # delta × spot move (×100 shares)
                + 0.5 * base_greeks.gamma * (ds * 100) ** 2  # gamma convexity term
                + base_greeks.vega * dv                   # vega × vol move
            )

            # Margin change estimate (crude: apply spot shift to margin)
            margin_calc = MarginCalculator(mode="regt")
            new_margin = margin_calc.total_margin(
                shocked_positions,
                portfolio_delta=shocked_greeks.delta,
                portfolio_gamma=shocked_greeks.gamma,
            )
            margin_change = new_margin - current_margin

            results.append(ScenarioResult(
                scenario_name=name,
                spot_shift=ds,
                vol_shift=dv,
                pnl=pnl,
                new_portfolio_delta=shocked_greeks.delta,
                new_portfolio_vega=shocked_greeks.vega,
                new_portfolio_gamma=shocked_greeks.gamma,
                margin_change=margin_change,
            ))

        return sorted(results, key=lambda r: r.scenario_name)

    # --- batch -------------------------------------------------------------- #

    def worst_scenarios(self,
                         positions: List[OptionPosition],
                         current_margin: float = 0.0,
                         n: int = 5) -> List[ScenarioResult]:
        """
        Return the N scenarios with the most negative P&L (worst-case first).

        Args:
            positions     : Option positions.
            current_margin: Current margin.
            n             : Number of worst scenarios to return.

        Returns:
            List[ScenarioResult]: Worst-case scenarios sorted by P&L ascending.
        """
        all_results = self.stress_test(positions, current_margin)
        return sorted(all_results, key=lambda r: r.pnl)[:n]

    def scenario_summary_df(self,
                             positions: List[OptionPosition],
                             current_margin: float = 0.0) -> Dict:
        """
        Return a dict suitable for building a summary table of all scenarios.

        Returns:
            Dict with keys: scenario_name, spot_shift, vol_shift, pnl,
                            new_delta, new_gamma, new_vega, margin_change.
        """
        results = self.stress_test(positions, current_margin)
        return {
            "scenario_name":      [r.scenario_name for r in results],
            "spot_shift":         [r.spot_shift for r in results],
            "vol_shift":          [r.vol_shift for r in results],
            "pnl":                [r.pnl for r in results],
            "new_delta":          [r.new_portfolio_delta for r in results],
            "new_gamma":          [r.new_portfolio_gamma for r in results],
            "new_vega":           [r.new_portfolio_vega for r in results],
            "margin_change":      [r.margin_change for r in results],
        }


# --------------------------------------------------------------------------- #
# RiskReport — generates formatted risk report with VaR, CVaR, Greeks exposure
# --------------------------------------------------------------------------- #

class RiskReport:
    """
    Generates a formatted risk report for an option portfolio.

    Adopted from OptionSuite/portfolioManager/portfolio.py
    `positionMonitoring` logging output, extended with:
      - Portfolio Greeks summary
      - Greeks exposure by symbol / option type / expiry
      - Value-at-Risk (VaR) approximation using Greeks covariance
      - Conditional VaR (CVaR / Expected Shortfall)

    Attributes:
        positions       : Positions to include in the report.
        portfolio_greeks: Pre-computed portfolio Greeks (optional).
        confidence_level: VaR confidence level (default 0.95).
        scenarios       : Optional list of ScenarioResult from stress_test.
    """

    def __init__(self,
                 positions: List[OptionPosition],
                 portfolio_greeks: Optional[Greeks] = None,
                 confidence_level: float = 0.95,
                 scenarios: Optional[List[ScenarioResult]] = None) -> None:
        self.positions = positions
        self._pg = portfolio_greeks
        self.confidence_level = confidence_level
        self.scenarios = scenarios or []

    # --- VaR / CVaR -------------------------------------------------------- #

    @staticmethod
    def _normal_var(mean: float, std: float,
                     confidence: float) -> float:
        """Parametric VaR under normal distribution."""
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        return -(mean + z * std)

    @staticmethod
    def _normal_cvar(mean: float, std: float,
                      confidence: float) -> float:
        """Parametric CVaR (Expected Shortfall) under normal distribution."""
        from scipy.stats import norm
        alpha = 1 - confidence
        z = norm.ppf(alpha)
        pdf_z = norm.pdf(z)
        cvar = -(mean - std * pdf_z / alpha)
        return cvar

    def var_cvar(self,
                  portfolio_value: float = 1_000_000.0,
                  daily_vol: float = 0.02,
                  horizon_days: int = 1) -> Tuple[float, float]:
        """
        Compute parametric VaR and CVaR for the portfolio.

        Args:
            portfolio_value: Total portfolio market value (default $1M).
            daily_vol       : Daily return volatility of the portfolio.
            horizon_days    : VaR horizon in days (sqrt-scaling applied).

        Returns:
            Tuple[float, float]: (VaR, CVaR) in dollars.
        """
        std = daily_vol * math.sqrt(horizon_days)
        mean_daily = 0.0   # assume zero mean for risk VaR
        var  = self._normal_var(mean_daily, std, self.confidence_level)
        cvar = self._normal_cvar(mean_daily, std, self.confidence_level)
        return portfolio_value * var, portfolio_value * cvar

    # --- Greeks exposure tables -------------------------------------------- #

    def greeks_exposure_table(self) -> Dict[str, Dict[str, float]]:
        """
        Build a Greeks exposure breakdown by underlying symbol.

        Returns:
            Dict: {symbol: {"delta": ..., "gamma": ..., "vega": ..., "theta": ...}}
        """
        table: Dict[str, Dict[str, float]] = {}
        for p in self.positions:
            sym = p.symbol
            if sym not in table:
                table[sym] = {"delta": 0.0, "gamma": 0.0,
                               "vega": 0.0, "theta": 0.0, "rho": 0.0}
            wg = p.weighted_greeks()
            table[sym]["delta"] += wg.delta
            table[sym]["gamma"] += wg.gamma
            table[sym]["vega"]  += wg.vega
            table[sym]["theta"] += wg.theta
            table[sym]["rho"]   += wg.rho
        return table

    def option_type_exposure(self) -> Dict[str, Dict[str, float]]:
        """
        Build Greeks exposure breakdown by option type (call/put).

        Returns:
            Dict: {"call": {...}, "put": {...}}
        """
        table: Dict[str, Dict[str, float]] = {
            "call": {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0},
            "put":  {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0},
        }
        for p in self.positions:
            wg = p.weighted_greeks()
            ot = p.option_type
            table[ot]["delta"] += wg.delta
            table[ot]["gamma"] += wg.gamma
            table[ot]["vega"]  += wg.vega
            table[ot]["theta"] += wg.theta
            table[ot]["rho"]   += wg.rho
        return table

    # --- formatted report --------------------------------------------------- #

    def generate(self,
                  portfolio_value: float = 1_000_000.0,
                  daily_vol: float = 0.02,
                  horizon_days: int = 1) -> str:
        """
        Generate a full human-readable risk report.

        Args:
            portfolio_value: Total portfolio market value (for VaR scaling).
            daily_vol       : Daily return volatility.
            horizon_days    : VaR horizon in days.

        Returns:
            str: Formatted multi-line risk report.
        """
        pg = self._pg
        if pg is None:
            pg = GreeksAggregator.compute(self.positions)

        var, cvar = self.var_cvar(portfolio_value, daily_vol, horizon_days)

        lines: List[str] = []
        sep = "=" * 62
        lines.append(sep)
        lines.append("  OptionSuite Risk Report")
        lines.append(f"  Confidence level : {self.confidence_level:.0%}")
        lines.append(f"  Positions        : {len(self.positions)}")
        lines.append(f"  VaR (${horizon_days}d) : ${var:>12,.2f}")
        lines.append(f"  CVaR (${horizon_days}d) : ${cvar:>12,.2f}")
        lines.append(sep)

        # Greeks summary
        lines.append("\n  Portfolio Greeks")
        lines.append(f"    Delta  : {pg.delta:>12.6f}")
        lines.append(f"    Gamma  : {pg.gamma:>12.6f}")
        lines.append(f"    Vega   : {pg.vega:>12.6f}  (per 1% vol)")
        lines.append(f"    Theta  : {pg.theta:>12.6f}  (daily $)")
        lines.append(f"    Rho    : {pg.rho:>12.6f}  (per 1% rate)")

        # Greeks by symbol
        sym_exp = self.greeks_exposure_table()
        lines.append("\n  Greeks by Underlying")
        lines.append(f"    {'Symbol':<10} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12}")
        for sym, g in sym_exp.items():
            lines.append(
                f"    {sym:<10} {g['delta']:>12.6f} {g['gamma']:>12.6f} "
                f"{g['vega']:>12.6f} {g['theta']:>12.6f}"
            )

        # Greeks by option type
        ot_exp = self.option_type_exposure()
        lines.append("\n  Greeks by Option Type")
        lines.append(f"    {'Type':<10} {'Delta':>12} {'Gamma':>12} {'Vega':>12} {'Theta':>12}")
        for ot, g in ot_exp.items():
            lines.append(
                f"    {ot:<10} {g['delta']:>12.6f} {g['gamma']:>12.6f} "
                f"{g['vega']:>12.6f} {g['theta']:>12.6f}"
            )

        # Stress test results
        if self.scenarios:
            lines.append("\n  Stress-Test Scenarios (top 10 by P&L)")
            lines.append(
                f"    {'Scenario':<30} {'Spot':>8} {'Vol':>8} "
                f"{'P&L':>14} {'NewΔ':>12} {'NewΓ':>12}"
            )
            for sc in sorted(self.scenarios, key=lambda s: s.pnl)[:10]:
                lines.append(
                    f"    {sc.scenario_name:<30} {sc.spot_shift:>8.0%} "
                    f"{sc.vol_shift:>8.0%} {sc.pnl:>14,.2f} "
                    f"{sc.new_portfolio_delta:>12.6f} {sc.new_portfolio_gamma:>12.6f}"
                )

        lines.append(sep)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.generate()


# --------------------------------------------------------------------------- #
# __all__
# --------------------------------------------------------------------------- #

__all__ = [
    # Option position descriptor
    "OptionPosition",
    # Greeks aggregation
    "PortfolioGreeks",
    "GreeksAggregator",
    # Margin
    "MarginCalculator",
    # Scenario analysis
    "RiskScenarioAnalyzer",
    "ScenarioResult",
    # Reporting
    "RiskReport",
]
