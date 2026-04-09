"""Portfolio Manager — Final Investment Decision Authority.

组合经理 — 最终投资决策权威模块.

Provides PortfolioManager that serves as the final decision authority,
integrating outputs from all analysts and researchers into executable
trading decisions. Designed to work alongside the Bull vs Bear debate
architecture defined in invest_debate.py.

Classes:
    PortfolioManager: 最终决策权威，整合所有Analyst和Researcher的输出
    PositionState: 当前持仓状态
    PortfolioRisk: 组合风险评估

Functions:
    compute_allowed_actions: 计算每只股票允许的交易行为和最大数量
    compute_position_limits: 基于波动率和相关性计算仓位限制

Bilingual docstrings (English primary, Chinese comments).
REST-only LLM client — no heavy SDK dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

__all__ = [
    "PortfolioManager",
    "PositionState",
    "PortfolioRisk",
    "compute_allowed_actions",
    "compute_position_limits",
    "calculate_volatility_metrics",
    "calculate_volatility_adjusted_limit",
    "calculate_correlation_multiplier",
]


# ---------------------------------------------------------------------------
# Position & Risk data classes
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    """持仓状态 / Current position state.

    Attributes:
        symbol: 股票代码
        long_shares: 多头持仓股数
        long_cost_basis: 多头持仓成本
        short_shares: 空头持仓股数
        short_cost_basis: 空头持仓成本
        current_price: 当前市场价格
    """

    symbol: str
    long_shares: int = 0
    long_cost_basis: float = 0.0
    short_shares: int = 0
    short_cost_basis: float = 0.0
    current_price: float = 0.0

    @property
    def total_exposure(self) -> float:
        """总敞口 (long - short) * price."""
        return abs(self.long_shares - self.short_shares) * self.current_price

    @property
    def market_value(self) -> float:
        """持仓市值 — long positive, short negative."""
        return (self.long_shares - self.short_shares) * self.current_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "long_shares": self.long_shares,
            "long_cost_basis": self.long_cost_basis,
            "short_shares": self.short_shares,
            "short_cost_basis": self.short_cost_basis,
            "current_price": self.current_price,
            "total_exposure": self.total_exposure,
            "market_value": self.market_value,
        }


@dataclass
class PortfolioRisk:
    """组合风险评估 / Portfolio risk assessment.

    Attributes:
        volatility_metrics: 波动率指标 (annualized/daily vol, percentile)
        correlation_metrics: 相关性指标 (与活跃仓位的平均/最大相关性)
        position_limit: 该标的的仓位限制 (美元金额)
        remaining_limit: 剩余可用仓位限制
        base_limit_pct: 基础仓位限制百分比
        correlation_multiplier: 相关性调整倍数
        combined_limit_pct: 组合后仓位限制百分比
        reasoning: 风险计算推理过程
    """

    volatility_metrics: dict[str, float] = field(default_factory=dict)
    correlation_metrics: dict[str, Any] = field(default_factory=dict)
    position_limit: float = 0.0
    remaining_limit: float = 0.0
    base_limit_pct: float = 0.20
    correlation_multiplier: float = 1.0
    combined_limit_pct: float = 0.20
    reasoning: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "volatility_metrics": self.volatility_metrics,
            "correlation_metrics": self.correlation_metrics,
            "position_limit": self.position_limit,
            "remaining_limit": self.remaining_limit,
            "base_limit_pct": self.base_limit_pct,
            "correlation_multiplier": self.correlation_multiplier,
            "combined_limit_pct": self.combined_limit_pct,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Volatility & Correlation utilities
# ---------------------------------------------------------------------------


def calculate_volatility_metrics(
    prices: list[float],
    lookback_days: int = 60,
) -> dict[str, float]:
    """Calculate comprehensive volatility metrics from price series.

    计算波动率指标 (年化波动率, 日波动率, 波动率百分位).

    Args:
        prices: List of closing prices (oldest to newest).
        lookback_days: Number of days for lookback (default 60).

    Returns:
        Dict with keys: daily_volatility, annualized_volatility,
        volatility_percentile, data_points.
    """
    if len(prices) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * math.sqrt(252),
            "volatility_percentile": 50.0,
            "data_points": len(prices),
        }

    import numpy as np

    prices_arr = np.array(prices)
    returns = np.diff(prices_arr) / prices_arr[:-1]

    if len(returns) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * math.sqrt(252),
            "volatility_percentile": 50.0,
            "data_points": len(returns),
        }

    # Use recent lookback for vol calculation
    recent = returns[-lookback_days:] if len(returns) > lookback_days else returns
    daily_vol = float(np.std(recent))
    annualized_vol = daily_vol * math.sqrt(252)

    # Percentile: compare current vol against rolling vols
    if len(returns) >= 30:
        rolling_vols = []
        for i in range(30, len(returns) + 1):
            rolling_vols.append(float(np.std(returns[i - 30 : i])))
        if rolling_vols and max(rolling_vols) > 0:
            vol_percentile = float(np.mean(np.array(rolling_vols) <= daily_vol)) * 100
        else:
            vol_percentile = 50.0
    else:
        vol_percentile = 50.0

    return {
        "daily_volatility": daily_vol,
        "annualized_volatility": annualized_vol,
        "volatility_percentile": vol_percentile,
        "data_points": len(recent),
    }


def calculate_volatility_adjusted_limit(annualized_volatility: float) -> float:
    """Calculate position size limit percentage based on volatility.

    基于波动率调整仓位限制:
    - 低波动 (<15%): 最高25%仓位
    - 中波动 (15-30%): 12.5-20%仓位
    - 高波动 (30-50%): 5-15%仓位
    - 极高波动 (>50%): 最高10%仓位

    Args:
        annualized_volatility: 年化波动率 (e.g. 0.25 = 25%)

    Returns:
        Position limit as a fraction of portfolio (e.g. 0.20 = 20%)
    """
    base_limit = 0.20

    if annualized_volatility < 0.15:
        vol_multiplier = 1.25  # Up to 25%
    elif annualized_volatility < 0.30:
        vol_multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5  # 20% -> 12.5%
    elif annualized_volatility < 0.50:
        vol_multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5  # 15% -> 5%
    else:
        vol_multiplier = 0.50  # Max 10%

    return max(0.05, min(0.25, base_limit * vol_multiplier))


def calculate_correlation_multiplier(avg_correlation: float) -> float:
    """Map average correlation to position size adjustment multiplier.

    基于相关性调整仓位倍数:
    - 极高相关 (>=0.8): 0.70x (大幅降低)
    - 高相关 (0.6-0.8): 0.85x
    - 中相关 (0.4-0.6): 1.00x (中性)
    - 低相关 (0.2-0.4): 1.05x (略增加)
    - 极低相关 (<0.2): 1.10x

    Args:
        avg_correlation: 平均相关系数 [0, 1]

    Returns:
        Position size multiplier.
    """
    if avg_correlation >= 0.80:
        return 0.70
    if avg_correlation >= 0.60:
        return 0.85
    if avg_correlation >= 0.40:
        return 1.00
    if avg_correlation >= 0.20:
        return 1.05
    return 1.10


def compute_position_limits(
    symbols: list[str],
    current_prices: dict[str, float],
    portfolio_value: float,
    volatility_data: dict[str, dict[str, float]],
    correlation_matrix: Optional[dict[str, dict[str, float]]] = None,
    active_positions: Optional[set[str]] = None,
) -> dict[str, PortfolioRisk]:
    """Compute volatility- and correlation-adjusted position limits.

    计算基于波动率和相关性的仓位限制.

    Args:
        symbols: List of ticker symbols to analyze.
        current_prices: Dict mapping symbol -> current price.
        portfolio_value: Total portfolio net liquidation value.
        volatility_data: Dict mapping symbol -> volatility metrics dict.
        correlation_matrix: Optional dict of dicts (symbol -> {symbol -> correlation}).
        active_positions: Optional set of symbols with non-zero exposure.

    Returns:
        Dict mapping symbol -> PortfolioRisk instance.
    """
    active = active_positions or set()
    results = {}

    for symbol in symbols:
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            results[symbol] = PortfolioRisk(
                reasoning={"error": "Missing price data for risk calculation"}
            )
            continue

        vol_data = volatility_data.get(symbol, {})
        ann_vol = vol_data.get("annualized_volatility", 0.25)

        # Base volatility-adjusted limit
        base_limit_pct = calculate_volatility_adjusted_limit(ann_vol)

        # Correlation adjustment
        corr_multiplier = 1.0
        corr_metrics: dict[str, Any] = {"avg_correlation_with_active": None}
        if correlation_matrix is not None and symbol in correlation_matrix:
            comparable = [t for t in active if t in correlation_matrix[symbol] and t != symbol]
            if not comparable:
                comparable = [t for t in correlation_matrix[symbol] if t != symbol]
            if comparable:
                corrs = [correlation_matrix[symbol][t] for t in comparable]
                avg_corr = sum(corrs) / len(corrs)
                max_corr = max(corrs)
                corr_metrics = {
                    "avg_correlation_with_active": avg_corr,
                    "max_correlation_with_active": max_corr,
                    "top_correlated": sorted(
                        [(t, correlation_matrix[symbol][t]) for t in comparable],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3],
                }
                corr_multiplier = calculate_correlation_multiplier(avg_corr)

        # Combined limit
        combined_limit_pct = base_limit_pct * corr_multiplier
        position_limit = portfolio_value * combined_limit_pct

        # Current position value
        # (placeholder — would come from current_positions in make_decision)
        current_pos_value = 0.0
        remaining = position_limit - current_pos_value

        results[symbol] = PortfolioRisk(
            volatility_metrics={
                "daily_volatility": vol_data.get("daily_volatility", 0.05),
                "annualized_volatility": ann_vol,
                "volatility_percentile": vol_data.get("volatility_percentile", 50.0),
                "data_points": int(vol_data.get("data_points", 0)),
            },
            correlation_metrics=corr_metrics,
            position_limit=position_limit,
            remaining_limit=remaining,
            base_limit_pct=base_limit_pct,
            correlation_multiplier=corr_multiplier,
            combined_limit_pct=combined_limit_pct,
            reasoning={
                "portfolio_value": portfolio_value,
                "current_position_value": current_pos_value,
                "vol_adjusted_limit_pct": base_limit_pct,
                "risk_adjustment": f"Vol x Corr adjusted: {combined_limit_pct:.1%} (base {base_limit_pct:.1%})",
            },
        )

    return results


def compute_allowed_actions(
    symbols: list[str],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Compute allowed actions and max quantities per symbol.

    计算每只股票允许的交易行为和最大数量.

    Args:
        symbols: List of ticker symbols.
        current_prices: Dict mapping symbol -> current price.
        max_shares: Dict mapping symbol -> max shares allowed by risk.
        portfolio: Portfolio dict with keys 'cash', 'positions', 'margin_requirement'.

    Returns:
        Dict mapping symbol -> {action: max_qty} (pruned to non-zero actions).
    """
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {}) or {}
    margin_req = float(portfolio.get("margin_requirement", 0.5))
    margin_used = float(portfolio.get("margin_used", 0.0))
    equity = float(portfolio.get("equity", cash))

    allowed = {}
    for symbol in symbols:
        price = float(current_prices.get(symbol, 0.0))
        pos = positions.get(
            symbol,
            {"long": 0, "long_cost_basis": 0.0, "short": 0, "short_cost_basis": 0.0},
        )
        long_shares = int(pos.get("long", 0) or 0)
        short_shares = int(pos.get("short", 0) or 0)
        max_qty = int(max_shares.get(symbol, 0) or 0)

        actions: dict[str, int] = {"buy": 0, "sell": 0, "short": 0, "cover": 0, "hold": 0}

        # Long side
        if long_shares > 0:
            actions["sell"] = long_shares
        if cash > 0 and price > 0:
            max_buy_cash = int(cash // price)
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy

        # Short side
        if short_shares > 0:
            actions["cover"] = short_shares
        if price > 0 and max_qty > 0:
            if margin_req <= 0.0:
                max_short = max_qty
            else:
                available_margin = max(0.0, (equity / margin_req) - margin_used)
                max_short_margin = int(available_margin // price)
                max_short = max(0, min(max_qty, max_short_margin))
            if max_short > 0:
                actions["short"] = max_short

        # Prune zero-capacity actions, keep hold
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v

        allowed[symbol] = pruned

    return allowed


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------


class PortfolioManager:
    """组合经理 — 最终决策权威 / Portfolio Manager — Final Decision Authority.

    整合所有Analyst和Researcher的输出, 生成最终交易决策.
    Works with the AnalystTeam from invest_debate.py and the existing
    DebateEngine from quant_trading.agent.debate_engine.

    Attributes:
        analysts: List of AnalystAgent instances.
        bull_researcher: BullResearcher from invest_debate.py.
        bear_researcher: BearResearcher from invest_debate.py.
        llm: Optional LLM call function (REST-only).

    Decision workflow:
        1. Collect AnalysisReports from all analysts
        2. Run BullResearcher + BearResearcher debate (configurable rounds)
        3. Compute risk-adjusted position limits
        4. Generate final TradingDecision per symbol
    """

    def __init__(
        self,
        analysts: list,  # list of AnalystAgent
        bull_researcher: Optional[Any] = None,
        bear_researcher: Optional[Any] = None,
        llm: Optional[Callable[[str], str]] = None,
        debate_rounds: int = 1,
    ) -> None:
        self.analysts = analysts
        self.bull_researcher = bull_researcher
        self.bear_researcher = bear_researcher
        self._llm = llm
        self.debate_rounds = debate_rounds

        # Import here to avoid circular import at module level
        # (invest_debate.py will be imported by the caller)
        self._analysis_report_cls = None
        self._trading_decision_cls = None

    def make_decision(
        self,
        symbol: str,
        market_data: dict[str, Any],
        current_positions: dict[str, Any],
    ) -> dict[str, Any]:
        """生成最终交易决策 / Generate final trading decision.

        Args:
            symbol: Ticker symbol.
            market_data: Market data for analysts:
                - prices (list[float]): price series for volatility calc
                - insider_trades, news_sentiments, headlines, financials, etc.
            current_positions: Portfolio positions:
                {'cash': float, 'positions': {symbol: {'long': int, 'short': int}}}

        Returns:
            Dict with keys: action, quantity, entry_price, stop_loss,
            take_profit, confidence, reasoning, risk_assessment.
        """
        # Lazy-load trading decision class
        if self._trading_decision_cls is None:
            from quant_trading.agent.invest_debate import TradingDecision, AnalysisReport
            self._analysis_report_cls = AnalysisReport
            self._trading_decision_cls = TradingDecision

        # Run analysts
        reports = []
        for analyst in self.analysts:
            analyst_data = self._route_data(analyst.agent_id, market_data)
            try:
                report = analyst.analyze(symbol, analyst_data)
                reports.append(report)
            except Exception:
                pass

        # Bull/Bear debate
        bull_ctx = None
        bear_ctx = None
        bull_result = {"claims": [], "overall_confidence": 0.5}
        bear_result = {"claims": [], "overall_confidence": 0.5}

        if self.bull_researcher is not None and self.bear_researcher is not None:
            for _ in range(self.debate_rounds):
                bull_result = self.bull_researcher.research(symbol, reports, bear_ctx)
                bear_result = self.bear_researcher.research(symbol, reports, bull_ctx)
                bull_ctx = "\n".join(bull_result.get("claims", []))
                bear_ctx = "\n".join(bear_result.get("claims", []))

        # Compute position limits from volatility
        prices_list = market_data.get("prices", [])
        vol_metrics = {}
        if prices_list:
            vol_metrics[symbol] = calculate_volatility_metrics(prices_list)

        current_prices = {symbol: prices_list[-1] if prices_list else 0.0}
        portfolio_value = self._calc_portfolio_value(current_positions, current_prices)
        risk_limits = compute_position_limits(
            symbols=[symbol],
            current_prices=current_prices,
            portfolio_value=portfolio_value,
            volatility_data=vol_metrics,
        )

        # Synthesize
        return self._synthesize(
            symbol=symbol,
            reports=reports,
            bull_result=bull_result,
            bear_result=bear_result,
            current_positions=current_positions,
            risk_limits=risk_limits,
        )

    def assess_risk(
        self,
        decision: dict[str, Any],
        portfolio: dict[str, Any],
    ) -> dict[str, Any]:
        """评估决策风险 / Assess the risk of a trading decision.

        Args:
            decision: Decision dict from make_decision.
            portfolio: Current portfolio dict.

        Returns:
            Dict with risk_level ('low'|'medium'|'high'), reason,
            max_loss_pct (if applicable), position_cost, available_cash.
        """
        action = decision.get("action", "hold")
        quantity = float(decision.get("quantity", 0))
        entry = float(decision.get("entry_price") or 0)
        stop = float(decision.get("stop_loss") or 0)

        cash = float(portfolio.get("cash", 0))
        positions = portfolio.get("positions", {})
        current_shares = 0
        symbol = decision.get("symbol", "")
        if symbol in positions:
            pos = positions[symbol]
            current_shares = int(pos.get("long", 0)) if isinstance(pos, dict) else 0

        if action == "hold":
            return {"risk_level": "none", "reason": "No position change"}
        if action == "buy":
            cost = quantity * entry
            if cost > cash:
                return {"risk_level": "high", "reason": "Insufficient cash"}
            max_loss_pct = (entry - stop) / entry if stop > 0 else 0.10
            return {
                "risk_level": "medium" if max_loss_pct < 0.10 else "high",
                "max_loss_pct": round(max_loss_pct, 4),
                "position_cost": round(cost, 2),
                "available_cash": round(cash, 2),
            }
        if action == "sell":
            if current_shares < quantity:
                return {"risk_level": "high", "reason": "Insufficient shares to sell"}
            return {"risk_level": "low", "reason": "Selling existing position"}
        return {"risk_level": "unknown"}

    def _route_data(self, agent_id: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Route relevant data fields to each analyst type."""
        routing = {
            "market_analyst": ["prices_df", "prices"],
            "sentiment_analyst": ["insider_trades", "news_sentiments", "weights"],
            "news_analyst": ["headlines"],
            "fundamentals_analyst": ["financials"],
        }
        keys = routing.get(agent_id, [])
        return {k: market_data.get(k) for k in keys}

    def _calc_portfolio_value(
        self,
        positions: dict[str, Any],
        current_prices: dict[str, float],
    ) -> float:
        """Calculate total portfolio net liquidation value."""
        cash = float(positions.get("cash", 0.0))
        total = cash
        for symbol, pos in positions.get("positions", {}).items():
            price = current_prices.get(symbol, 0.0)
            if isinstance(pos, dict):
                long = int(pos.get("long", 0) or 0)
                short = int(pos.get("short", 0) or 0)
            else:
                long, short = 0, 0
            total += (long - short) * price
        return total

    def _synthesize(
        self,
        symbol: str,
        reports: list,
        bull_result: dict[str, Any],
        bear_result: dict[str, Any],
        current_positions: dict[str, Any],
        risk_limits: dict[str, PortfolioRisk],
    ) -> dict[str, Any]:
        """Synthesize all inputs into a final decision dict."""
        numeric_view = {"bullish": 1, "neutral": 0, "bearish": -1}

        if not reports:
            return {
                "symbol": symbol,
                "action": "hold",
                "quantity": 0,
                "confidence": 0.0,
                "reasoning": "No analyst reports available",
            }

        # Weighted analyst view
        total_conf = sum(r.confidence for r in reports)
        if total_conf > 0:
            weighted_view = sum(numeric_view.get(r.market_view, 0) * r.confidence for r in reports) / total_conf
        else:
            weighted_view = 0.0

        # Debate contribution
        bull_conf = bull_result.get("overall_confidence", 0.5)
        bear_conf = bear_result.get("overall_confidence", 0.5)
        debate_score = (bull_conf - bear_conf) * 0.3

        final_score = weighted_view + debate_score

        # Current position
        cash = float(current_positions.get("cash", 0))
        pos = current_positions.get("positions", {}).get(symbol, {})
        current_shares = int(pos.get("long", 0)) if isinstance(pos, dict) else 0

        # Price
        prices_list = []
        for r in reports:
            if r.agent_id == "market_analyst" and hasattr(r, "fundamentals"):
                pass
        # Use risk limit price if available
        risk = risk_limits.get(symbol)
        price = risk_limits.get(symbol, PortfolioRisk()).position_limit  # fallback
        if risk and risk.volatility_metrics:
            # use 100 as placeholder if no real price
            price = 100.0
        else:
            price = 100.0

        # Decision
        if final_score > 0.2 and cash >= price:
            action = "buy"
            remaining = risk.remaining_limit if risk else (cash * 0.5)
            quantity = max(int(remaining / price), 1) if price > 0 else 0
            entry = price
            stop_loss = price * 0.95
            take_profit = price * 1.20
        elif final_score < -0.2 and current_shares > 0:
            action = "sell"
            quantity = current_shares
            entry = price
            stop_loss = None
            take_profit = None
        else:
            action = "hold"
            quantity = 0
            entry = None
            stop_loss = None
            take_profit = None

        confidence = min(abs(final_score) * 1.5, 1.0) if final_score != 0 else 0.5

        reasoning_parts = [
            f"Analyst consensus: {'bullish' if weighted_view > 0 else 'bearish' if weighted_view < 0 else 'neutral'} (score={weighted_view:.3f})"
        ]
        if bull_result.get("claims"):
            reasoning_parts.append(f"Bull: {bull_result['claims'][0][:60]}")
        if bear_result.get("claims"):
            reasoning_parts.append(f"Bear: {bear_result['claims'][0][:60]}")

        return {
            "symbol": symbol,
            "action": action,
            "quantity": float(quantity),
            "entry_price": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": round(confidence, 4),
            "reasoning": " | ".join(reasoning_parts),
            "supporting_reports": [r.to_dict() for r in reports],
            "risk_assessment": self._assess_decision_risk(action, quantity, entry, stop_loss, cash, current_shares),
        }

    @staticmethod
    def _assess_decision_risk(
        action: str,
        quantity: float,
        entry: Optional[float],
        stop_loss: Optional[float],
        cash: float,
        current_shares: int,
    ) -> str:
        if action == "hold":
            return "No action — neutral market view."
        if action == "buy":
            cost = quantity * (entry or 0)
            risk = f"Buy {quantity} shares @ ${entry:.2f}, total ${cost:.2f}, cash ${cash:.2f}."
            if stop_loss:
                risk += f" Stop-loss: ${stop_loss:.2f}."
            return risk
        if action == "sell":
            return f"Sell {quantity} shares from {current_shares} held."
        return "Unknown action."
