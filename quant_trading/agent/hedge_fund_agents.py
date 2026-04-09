"""Hedge Fund Multi-Agent Architecture — 多智能体对冲基金架构.

Absorbed from D:/Hive/Data/trading_repos/Hedge_Fund_Agents/.

Multi-agent risk management architecture featuring:
  - PortfolioManagerAgent:    Portfolio allocation & rebalancing authority
  - RiskAnalystAgent:        Portfolio risk analysis & risk reporting
  - MacroStrategistAgent:   Macro-level asset allocation analysis
  - ExecutionAgent:         Trade execution via connectors
  - HedgeFundMultiAgent:    Coordinates all agents in a unified decision cycle

Agent communication: simple in-memory message passing (no graph framework).
Key method: run_cycle() executes one full decision cycle of the system.

Bilingual docstrings (English primary, Chinese notes).
REST-only for any LLM calls — no langchain/lgraph dependency.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Callable, Optional

__all__ = [
    "PortfolioManagerAgent",
    "RiskAnalystAgent",
    "MacroStrategistAgent",
    "ExecutionAgent",
    "HedgeFundMultiAgent",
    "TradeAction",
    "RiskLevel",
    "AgentMessage",
    "AllocationPlan",
    "RiskReport",
    "MacroSignal",
    "ExecutionResult",
    "CycleResult",
]


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class TradeAction(str, Enum):
    """Trade action types / 交易行为类型."""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    HOLD = "hold"


class RiskLevel(str, Enum):
    """Risk level classification / 风险等级分类."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentMessage:
    """Lightweight message envelope for agent communication.

    轻量级消息信封，用于智能体间通信.

    Attributes:
        sender:  发送方智能体名称
        receiver: 接收方智能体名称 ("*" for broadcast)
        subject: 消息主题
        payload: 消息数据载荷
        timestamp: 时间戳
    """

    sender: str
    receiver: str
    subject: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "subject": self.subject,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


@dataclass
class AllocationPlan:
    """Portfolio allocation plan produced by PortfolioManagerAgent.

    组合配置计划.

    Attributes:
        target_weights:   Dict mapping symbol -> target weight (0.0–1.0)
        current_weights:  Dict mapping symbol -> current weight
        rebalance_shares: Dict mapping symbol -> {action: shares}
        reasoning:        Human-readable reasoning
    """

    target_weights: dict[str, float]
    current_weights: dict[str, float]
    rebalance_shares: dict[str, dict[str, int]]
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_weights": self.target_weights,
            "current_weights": self.current_weights,
            "rebalance_shares": self.rebalance_shares,
            "reasoning": self.reasoning,
        }


@dataclass
class RiskReport:
    """Risk analysis report produced by RiskAnalystAgent.

    风险分析报告.

    Attributes:
        portfolio_volatility:     Annualized portfolio volatility
        max_drawdown:             Estimated max drawdown
        var_95:                   95% Value at Risk (dollar amount)
        correlation_matrix:       Symbol correlation dict
        position_limits:           Per-symbol risk limits
        risk_level:               Overall RiskLevel
        reasons:                  List of risk concern strings
    """

    portfolio_volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    position_limits: dict[str, float] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_volatility": self.portfolio_volatility,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "correlation_matrix": self.correlation_matrix,
            "position_limits": self.position_limits,
            "risk_level": self.risk_level.value,
            "reasons": self.reasons,
        }


@dataclass
class MacroSignal:
    """Macro strategy signal produced by MacroStrategistAgent.

    宏观策略信号.

    Attributes:
        market_regime:     "bull" | "bear" | "sideways"
        recommended_equity_exposure:  0.0–1.0
        recommended_bond_exposure:    0.0–1.0
        recommended_cash_exposure:    0.0–1.0
        recommended_risk_appetite:     "low" | "medium" | "high"
        key_themes:        List of macro theme strings
        reasoning:         Explanation
    """

    market_regime: str = "sideways"
    recommended_equity_exposure: float = 0.6
    recommended_bond_exposure: float = 0.2
    recommended_cash_exposure: float = 0.2
    recommended_risk_appetite: str = "medium"
    key_themes: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_regime": self.market_regime,
            "recommended_equity_exposure": self.recommended_equity_exposure,
            "recommended_bond_exposure": self.recommended_bond_exposure,
            "recommended_cash_exposure": self.recommended_cash_exposure,
            "recommended_risk_appetite": self.recommended_risk_appetite,
            "key_themes": self.key_themes,
            "reasoning": self.reasoning,
        }


@dataclass
class ExecutionResult:
    """Result of trade execution.

    交易执行结果.

    Attributes:
        symbol:        Ticker symbol
        action:        TradeAction that was executed
        requested_qty: Requested quantity
        executed_qty:  Actually executed quantity
        avg_price:     Average fill price
        status:        "filled" | "partial" | "rejected" | "pending"
        error:         Error message if rejected
    """

    symbol: str
    action: TradeAction
    requested_qty: int = 0
    executed_qty: int = 0
    avg_price: float = 0.0
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "requested_qty": self.requested_qty,
            "executed_qty": self.executed_qty,
            "avg_price": self.avg_price,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class CycleResult:
    """Result of one full run_cycle() on HedgeFundMultiAgent.

    完整决策周期的执行结果.

    Attributes:
        cycle_id:          Unique cycle identifier
        macro_signal:      MacroStrategistAgent output
        risk_report:       RiskAnalystAgent output
        allocation_plan:   PortfolioManagerAgent output
        execution_results: List[ExecutionResult]
        portfolio_state:   Updated portfolio dict
        duration_ms:      Cycle duration in milliseconds
        agent_messages:   Messages exchanged during cycle
    """

    cycle_id: int
    macro_signal: MacroSignal
    risk_report: RiskReport
    allocation_plan: AllocationPlan
    execution_results: list[ExecutionResult]
    portfolio_state: dict[str, Any]
    duration_ms: float
    agent_messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "macro_signal": self.macro_signal.to_dict(),
            "risk_report": self.risk_report.to_dict(),
            "allocation_plan": self.allocation_plan.to_dict(),
            "execution_results": [r.to_dict() for r in self.execution_results],
            "portfolio_state": self.portfolio_state,
            "duration_ms": self.duration_ms,
            "agent_messages": self.agent_messages,
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _safe_llm_json(
    prompt: str,
    llm_call: Optional[Callable[[str], str]] = None,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call LLM via REST and parse JSON response. Falls back to default.

    通过 REST 调用 LLM 并解析 JSON 响应; 失败时返回 default.

    Args:
        prompt:    The prompt string to send to the LLM.
        llm_call:  Callable that accepts str and returns str (REST response body).
                   If None, returns default immediately.
        default:   Fallback dict when LLM is unavailable or returns invalid JSON.

    Returns:
        Parsed JSON dict from LLM or default.
    """
    if llm_call is None:
        return default or {}
    try:
        raw = llm_call(prompt)
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)
    except Exception:
        return default or {}


def _calc_volatility(prices: list[float], lookback: int = 60) -> float:
    """Calculate annualized volatility from price series.

    从价格序列计算年化波动率.
    """
    if len(prices) < 2:
        return 0.20
    try:
        import numpy as _np

        arr = _np.array(prices)
        returns = _np.diff(arr) / arr[:-1]
        recent = returns[-lookback:] if len(returns) > lookback else returns
        daily_vol = float(_np.std(recent))
        return daily_vol * math.sqrt(252)
    except Exception:
        return 0.20


def _calc_var(portfolio_value: float, volatility: float, confidence: float = 0.95) -> float:
    """Calculate Value at Risk (parametric) for a portfolio.

    计算投资组合的参数 VaR (95% confidence, log-normal).
    """
    z = 1.645 if confidence == 0.95 else 1.282  # ~90% / ~95%
    return portfolio_value * volatility * z / math.sqrt(252)


def _correlation_matrix(price_data: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Build a symmetric correlation matrix from per-symbol price dicts.

    从各标的价格字典构建对称相关矩阵.
    """
    symbols = list(price_data.keys())
    n = len(symbols)
    if n == 0:
        return {}
    try:
        import numpy as _np

        min_len = min(len(price_data[s]) for s in symbols)
        if min_len < 5:
            return {s: {t: 0.0 for t in symbols} for s in symbols}
        aligned = _np.column_stack([price_data[s][:min_len] for s in symbols])
        corr = _np.corrcoef(aligned, rowvar=False)
        return {
            symbols[i]: {symbols[j]: float(corr[i][j]) for j in range(n)}
            for i in range(n)
        }
    except Exception:
        return {s: {t: 0.0 for t in symbols} for s in symbols}


def _rebalance_shares(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
    current_prices: dict[str, float],
    positions: dict[str, dict[str, Any]],
    cash: float,
) -> dict[str, dict[str, int]]:
    """Compute rebalance share quantities per symbol.

    计算各标的的调仓股数.

    Args:
        target_weights:  Target weight per symbol (fraction of total portfolio).
        current_weights: Current weight per symbol.
        current_prices:  Current price per symbol.
        positions:       Position dict per symbol {'long': int, 'short': int}.
        cash:            Current cash balance.

    Returns:
        Dict: {symbol: {"buy": int, "sell": int, "short": int, "cover": int}}.
    """
    result: dict[str, dict[str, int]] = {s: {"buy": 0, "sell": 0, "short": 0, "cover": 0} for s in target_weights}

    # Total portfolio value (approximate)
    total_value = cash
    for s, pos in positions.items():
        price = current_prices.get(s, 0.0)
        long = int(pos.get("long", 0) or 0)
        short = int(pos.get("short", 0) or 0)
        total_value += (long - short) * price

    for symbol, target_w in target_weights.items():
        current_w = current_weights.get(symbol, 0.0)
        price = current_prices.get(symbol, 0.0)
        if price <= 0:
            continue

        target_value = total_value * target_w
        current_pos = positions.get(symbol, {"long": 0, "short": 0})
        current_long = int(current_pos.get("long", 0) or 0)
        current_short = int(current_pos.get("short", 0) or 0)
        current_value = (current_long - current_short) * price

        delta_value = target_value - current_value
        delta_shares = int(round(delta_value / price))

        if delta_shares > 0:
            max_affordable = int(cash // price) if price > 0 else 0
            result[symbol]["buy"] = min(delta_shares, max_affordable)
        elif delta_shares < 0:
            result[symbol]["sell"] = min(abs(delta_shares), current_long)

    return result


# ---------------------------------------------------------------------------
# Agent Classes
# ---------------------------------------------------------------------------


class PortfolioManagerAgent:
    """Portfolio Manager Agent — 组合经理智能体.

    Responsible for overall portfolio allocation and rebalancing decisions.
    Integrates inputs from RiskAnalystAgent and MacroStrategistAgent to
    produce a target allocation plan.

    Responsibilities:
        1. Receive risk report from RiskAnalystAgent.
        2. Receive macro signal from MacroStrategistAgent.
        3. Compute optimal target weights across all symbols.
        4. Generate rebalance trade list (shares per symbol per action).
        5. Send allocation plan to HedgeFundMultiAgent for execution.

    Methods:
        analyze():     Main analysis entry — returns AllocationPlan.
        update_state(): Update internal portfolio state.
    """

    def __init__(
        self,
        name: str = "PortfolioManager",
        llm_call: Optional[Callable[[str], str]] = None,
        base_position_limit_pct: float = 0.20,
    ) -> None:
        self.name = name
        self._llm = llm_call
        self.base_position_limit_pct = base_position_limit_pct
        self._portfolio_state: dict[str, Any] = {}

    def analyze(
        self,
        symbols: list[str],
        macro_signal: MacroSignal,
        risk_report: RiskReport,
        current_prices: dict[str, float],
        current_positions: dict[str, dict[str, Any]],
        cash: float,
    ) -> AllocationPlan:
        """Compute target allocation plan integrating macro and risk inputs.

        Args:
            symbols:          List of ticker symbols under management.
            macro_signal:     Output from MacroStrategistAgent.
            risk_report:       Output from RiskAnalystAgent.
            current_prices:   Dict mapping symbol -> current price.
            current_positions: Dict mapping symbol -> {"long": int, "short": int}.
            cash:              Current cash balance.

        Returns:
            AllocationPlan with target_weights, current_weights, rebalance_shares.
        """
        # ---- Calculate current portfolio value and weights ----
        total_value = cash
        for sym, pos in current_positions.items():
            price = current_prices.get(sym, 0.0)
            long = int(pos.get("long", 0) or 0)
            short = int(pos.get("short", 0) or 0)
            total_value += (long - short) * price

        current_weights: dict[str, float] = {}
        for sym in symbols:
            price = current_prices.get(sym, 0.0)
            pos = current_positions.get(sym, {"long": 0, "short": 0})
            long = int(pos.get("long", 0) or 0)
            short = int(pos.get("short", 0) or 0)
            value = (long - short) * price
            current_weights[sym] = value / total_value if total_value > 0 else 0.0

        # ---- Determine equity exposure from macro signal ----
        equity_budget = macro_signal.recommended_equity_exposure
        n_assets = len(symbols)
        if n_assets == 0:
            target_weights: dict[str, float] = {}
        else:
            # Risk-adjusted weight: scale by inverse of position limit (from risk report)
            raw_weights: dict[str, float] = {}
            for sym in symbols:
                limit = risk_report.position_limits.get(sym, total_value * self.base_position_limit_pct)
                # Scale limit to budget
                raw_weights[sym] = min(limit / total_value, 1.0) if total_value > 0 else 0.0

            total_raw = sum(raw_weights.values())
            if total_raw > 0:
                scaled = {s: (w / total_raw) * equity_budget for s, w in raw_weights.items()}
            else:
                scaled = {s: equity_budget / n_assets for s in symbols}

            # Apply per-symbol caps from risk report
            target_weights = {}
            for sym in symbols:
                cap = (risk_report.position_limits.get(sym, total_value * self.base_position_limit_pct)
                       / total_value if total_value > 0 else self.base_position_limit_pct)
                target_weights[sym] = min(scaled.get(sym, 0.0), cap, equity_budget)

        # Ensure weights sum approximately to equity budget
        weight_sum = sum(target_weights.values())
        if weight_sum > 0 and abs(weight_sum - equity_budget) > 0.001:
            target_weights = {s: w * equity_budget / weight_sum for s, w in target_weights.items()}

        # Compute rebalance shares
        rebalance_shares = _rebalance_shares(
            target_weights=target_weights,
            current_weights=current_weights,
            current_prices=current_prices,
            positions=current_positions,
            cash=cash,
        )

        reasoning = (
            f"PortfolioManager: equity_budget={equity_budget:.0%}, "
            f"regime={macro_signal.market_regime}, "
            f"risk_level={risk_report.risk_level.value}, "
            f"symbols={symbols}"
        )

        return AllocationPlan(
            target_weights=target_weights,
            current_weights=current_weights,
            rebalance_shares=rebalance_shares,
            reasoning=reasoning,
        )

    def update_state(self, portfolio_state: dict[str, Any]) -> None:
        """Update internal portfolio state for next cycle."""
        self._portfolio_state = portfolio_state


class RiskAnalystAgent:
    """Risk Analyst Agent — 风险分析智能体.

    Analyzes portfolio risk and generates a comprehensive risk report.
    Evaluates volatility, correlation, drawdown, and VaR across all positions.

    Responsibilities:
        1. Compute per-symbol and portfolio-level volatility.
        2. Build correlation matrix from price data.
        3. Estimate max drawdown and VaR.
        4. Determine per-symbol position limits based on risk.
        5. Produce an overall risk level classification.

    Methods:
        analyze():   Main analysis entry — returns RiskReport.
    """

    def __init__(
        self,
        name: str = "RiskAnalyst",
        llm_call: Optional[Callable[[str], str]] = None,
        base_position_limit_pct: float = 0.20,
        var_confidence: float = 0.95,
    ) -> None:
        self.name = name
        self._llm = llm_call
        self.base_position_limit_pct = base_position_limit_pct
        self.var_confidence = var_confidence

    def analyze(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
        current_positions: dict[str, dict[str, Any]],
        current_prices: dict[str, float],
        cash: float,
    ) -> RiskReport:
        """Produce a comprehensive risk report.

        Args:
            symbols:           List of ticker symbols under management.
            price_data:        Dict mapping symbol -> list of historical closing prices.
            current_positions: Dict mapping symbol -> {"long": int, "short": int}.
            current_prices:    Dict mapping symbol -> current price.
            cash:              Current cash balance.

        Returns:
            RiskReport with volatility, VaR, drawdown, position limits, and risk level.
        """
        # Total portfolio value
        total_value = cash
        for sym, pos in current_positions.items():
            price = current_prices.get(sym, 0.0)
            long = int(pos.get("long", 0) or 0)
            short = int(pos.get("short", 0) or 0)
            total_value += (long - short) * price

        # Per-symbol volatility
        vol_by_symbol: dict[str, float] = {}
        for sym in symbols:
            prices = price_data.get(sym, [])
            vol_by_symbol[sym] = _calc_volatility(prices)

        # Portfolio volatility (weighted average, simplified)
        portfolio_vol = 0.0
        if total_value > 0 and symbols:
            for sym in symbols:
                price = current_prices.get(sym, 0.0)
                pos = current_positions.get(sym, {"long": 0, "short": 0})
                long = int(pos.get("long", 0) or 0)
                short = int(pos.get("short", 0) or 0)
                exposure = (long - short) * price
                weight = exposure / total_value
                portfolio_vol += vol_by_symbol.get(sym, 0.0) * weight
            portfolio_vol = abs(portfolio_vol)

        # Correlation matrix
        correlation_matrix = _correlation_matrix(price_data)

        # VaR
        var_95 = _calc_var(total_value, portfolio_vol, self.var_confidence)

        # Max drawdown estimate (simplified: 2x daily VaR as proxy)
        max_drawdown = var_95 * 2.0

        # Per-symbol position limits (volatility-adjusted)
        position_limits: dict[str, float] = {}
        for sym in symbols:
            ann_vol = vol_by_symbol.get(sym, 0.20)
            # Lower limit for higher volatility
            if ann_vol < 0.15:
                limit_pct = 0.25
            elif ann_vol < 0.30:
                limit_pct = 0.20
            elif ann_vol < 0.50:
                limit_pct = 0.10
            else:
                limit_pct = 0.05
            position_limits[sym] = total_value * limit_pct

        # Overall risk level
        reasons: list[str] = []
        risk_level = RiskLevel.LOW

        if portfolio_vol > 0.40:
            risk_level = RiskLevel.HIGH
            reasons.append(f"Portfolio volatility extremely elevated: {portfolio_vol:.1%}")
        elif portfolio_vol > 0.25:
            risk_level = RiskLevel.MEDIUM
            reasons.append(f"Portfolio volatility moderately high: {portfolio_vol:.1%}")

        if var_95 > total_value * 0.10:
            if risk_level == RiskLevel.HIGH:
                risk_level = RiskLevel.CRITICAL
            reasons.append(f"VaR(95%) is ${var_95:,.0f} ({var_95/total_value:.1%} of portfolio)")

        if max_drawdown > total_value * 0.20:
            reasons.append(f"Max drawdown estimate: ${max_drawdown:,.0f} ({max_drawdown/total_value:.1%} of portfolio)")

        # LLM enhancement (optional REST call)
        reasoning_prompt = (
            f"Risk analysis summary:\n"
            f"Portfolio value: ${total_value:,.0f}\n"
            f"Portfolio volatility: {portfolio_vol:.1%}\n"
            f"VaR(95%): ${var_95:,.0f}\n"
            f"Max drawdown est.: ${max_drawdown:,.0f}\n"
            f"Per-symbol vols: { {s: f'{v:.1%}' for s, v in vol_by_symbol.items()} }\n"
            f"Risk level: {risk_level.value}\n"
            f"Provide a brief (2-3 sentence) macro risk commentary in Chinese."
        )
        llm_reasoning = _safe_llm_json(
            reasoning_prompt,
            self._llm,
            default={"commentary": ""},
        )
        if llm_reasoning.get("commentary"):
            reasons.append(llm_reasoning["commentary"])

        return RiskReport(
            portfolio_volatility=portfolio_vol,
            max_drawdown=max_drawdown,
            var_95=var_95,
            correlation_matrix=correlation_matrix,
            position_limits=position_limits,
            risk_level=risk_level,
            reasons=reasons,
        )


class MacroStrategistAgent:
    """Macro Strategist Agent — 宏观策略智能体.

    Performs top-down macro analysis to set overall portfolio risk appetite
    and asset-class allocation (equity / bond / cash).

    Responsibilities:
        1. Analyze current macro regime (bull / bear / sideways).
        2. Recommend equity / bond / cash exposure weights.
        3. Identify key macro themes (inflation, rates, growth, etc.).
        4. Set risk appetite level for downstream agents.

    Methods:
        analyze(): Main analysis entry — returns MacroSignal.
    """

    def __init__(
        self,
        name: str = "MacroStrategist",
        llm_call: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.name = name
        self._llm = llm_call

    def analyze(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
        macro_context: Optional[dict[str, Any]] = None,
    ) -> MacroSignal:
        """Produce macro allocation signal.

        Args:
            symbols:      List of ticker symbols under management.
            price_data:   Dict mapping symbol -> list of historical closing prices.
            macro_context: Optional dict with external macro data (rates, CPI, etc.).

        Returns:
            MacroSignal with regime, recommended exposures, risk appetite, themes.
        """
        context = macro_context or {}

        # Default: equal-weight equity budget based on regime inference
        # Bull: 70% equity, Bear: 30% equity, Sideways: 50% equity
        regime = "sideways"
        equity_budget = 0.50

        if symbols and price_data:
            # Infer regime from recent trend (simplified: 20-day vs 60-day MA)
            try:
                for sym in symbols[:3]:  # Sample first 3 symbols
                    prices = price_data.get(sym, [])
                    if len(prices) >= 60:
                        short_ma = sum(prices[-20:]) / 20
                        long_ma = sum(prices[-60:]) / 60
                        if short_ma > long_ma * 1.05:
                            regime = "bull"
                            equity_budget = 0.70
                        elif short_ma < long_ma * 0.95:
                            regime = "bear"
                            equity_budget = 0.30
                        break
            except Exception:
                pass

        # Risk appetite tied to regime
        risk_appetite = "medium"
        if regime == "bull":
            risk_appetite = "high"
        elif regime == "bear":
            risk_appetite = "low"

        # Bond and cash complements
        bond_budget = round((1.0 - equity_budget) * 0.6, 2)
        cash_budget = round(1.0 - equity_budget - bond_budget, 2)

        # LLM enhancement for macro theme identification
        macro_prompt = (
            f"Macro analysis context:\n"
            f"Regime detected: {regime}\n"
            f"Equity budget: {equity_budget:.0%}\n"
            f"Symbols: {symbols}\n"
            f"External context: {json.dumps(context)}\n"
            f"Identify 3-5 key macro themes (e.g. inflation, rate hikes, growth slowdown) "
            f"and provide brief commentary. Respond in JSON with keys: "
            f"'themes' (list of strings) and 'commentary' (string in Chinese)."
        )
        llm_result = _safe_llm_json(
            macro_prompt,
            self._llm,
            default={"themes": [], "commentary": ""},
        )

        themes = llm_result.get("themes", [])
        commentary = llm_result.get("commentary", "")

        return MacroSignal(
            market_regime=regime,
            recommended_equity_exposure=equity_budget,
            recommended_bond_exposure=bond_budget,
            recommended_cash_exposure=cash_budget,
            recommended_risk_appetite=risk_appetite,
            key_themes=themes,
            reasoning=f"Regime={regime}, equity={equity_budget:.0%}, risk={risk_appetite}. {commentary}",
        )


class ExecutionAgent:
    """Execution Agent — 交易执行智能体.

    Executes trades generated by PortfolioManagerAgent through trade connectors.
    Supports market orders with basic slippage estimation.

    Responsibilities:
        1. Receive list of desired trades from PortfolioManagerAgent.
        2. Route each trade to the appropriate connector (broker API).
        3. Track fills and produce ExecutionResult per symbol.
        4. Update internal position state after execution.

    Note:
        This is a *dry-run* executor by default. Set `live_mode=True`
        and provide a valid `connector_fn` to enable live trading.

    Methods:
        execute():     Execute a list of rebalance shares — returns list[ExecutionResult].
        set_live_mode(): Enable/disable live execution.
    """

    def __init__(
        self,
        name: str = "ExecutionAgent",
        connector_fn: Optional[Callable[[str, TradeAction, int], ExecutionResult]] = None,
        live_mode: bool = False,
    ) -> None:
        self.name = name
        self._connector = connector_fn
        self.live_mode = live_mode
        self._position_state: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        rebalance_shares: dict[str, dict[str, int]],
        current_prices: dict[str, float],
        current_positions: dict[str, dict[str, Any]],
        cash: float,
    ) -> tuple[list[ExecutionResult], dict[str, Any]]:
        """Execute rebalance trades and return results + updated portfolio state.

        Args:
            rebalance_shares:  Dict {symbol: {"buy": int, "sell": int, "short": int, "cover": int}}.
            current_prices:    Current prices per symbol.
            current_positions: Current positions per symbol.
            cash:              Current cash balance.

        Returns:
            Tuple of (list of ExecutionResult, updated portfolio dict).
        """
        results: list[ExecutionResult] = []
        new_positions: dict[str, dict[str, Any]] = {
            s: dict(pos) for s, pos in current_positions.items()
        }
        new_cash = float(cash)

        for symbol, actions in rebalance_shares.items():
            price = current_prices.get(symbol, 0.0)
            pos = dict(new_positions.get(symbol, {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0}))
            long = int(pos.get("long", 0) or 0)
            short = int(pos.get("short", 0) or 0)

            # ---- Process buys ----
            buy_qty = actions.get("buy", 0)
            if buy_qty > 0 and price > 0:
                result = self._execute_single(symbol, TradeAction.BUY, buy_qty, price)
                if result.status == "filled":
                    cost = result.executed_qty * result.avg_price
                    new_cash -= cost
                    long += result.executed_qty
                results.append(result)

            # ---- Process sells ----
            sell_qty = actions.get("sell", 0)
            if sell_qty > 0 and price > 0:
                result = self._execute_single(symbol, TradeAction.SELL, sell_qty, price)
                if result.status == "filled":
                    proceeds = result.executed_qty * result.avg_price
                    new_cash += proceeds
                    long -= result.executed_qty
                results.append(result)

            # ---- Process shorts ----
            short_qty = actions.get("short", 0)
            if short_qty > 0 and price > 0:
                result = self._execute_single(symbol, TradeAction.SHORT, short_qty, price)
                if result.status == "filled":
                    proceeds = result.executed_qty * result.avg_price
                    new_cash += proceeds
                    short += result.executed_qty
                results.append(result)

            # ---- Process covers ----
            cover_qty = actions.get("cover", 0)
            if cover_qty > 0 and price > 0:
                result = self._execute_single(symbol, TradeAction.COVER, cover_qty, price)
                if result.status == "filled":
                    cost = result.executed_qty * result.avg_price
                    new_cash -= cost
                    short -= result.executed_qty
                results.append(result)

            # Ensure non-negative
            new_positions[symbol] = {
                "long": max(0, long),
                "short": max(0, short),
                "long_cost_basis": pos.get("long_cost_basis", 0.0),
                "short_cost_basis": pos.get("short_cost_basis", 0.0),
            }

        # Rebuild portfolio dict
        updated_portfolio: dict[str, Any] = {
            "cash": new_cash,
            "positions": new_positions,
            "updated_at": datetime.now().isoformat(),
        }
        return results, updated_portfolio

    def _execute_single(
        self,
        symbol: str,
        action: TradeAction,
        quantity: int,
        price: float,
    ) -> ExecutionResult:
        """Execute a single trade through the connector or simulate."""

        def _simulated_fill() -> ExecutionResult:
            # Simulate market order with 0.05% slippage
            slippage = 0.0005
            if action in (TradeAction.BUY, TradeAction.COVER):
                fill_price = price * (1 + slippage)
            else:
                fill_price = price * (1 - slippage)
            return ExecutionResult(
                symbol=symbol,
                action=action,
                requested_qty=quantity,
                executed_qty=quantity,
                avg_price=fill_price,
                status="filled",
            )

        if self.live_mode and self._connector is not None:
            try:
                return self._connector(symbol, action, quantity)
            except Exception as e:
                return ExecutionResult(
                    symbol=symbol,
                    action=action,
                    requested_qty=quantity,
                    executed_qty=0,
                    avg_price=price,
                    status="rejected",
                    error=str(e),
                )
        else:
            return _simulated_fill()

    def set_live_mode(self, live: bool, connector_fn: Optional[Callable] = None) -> None:
        """Enable or disable live execution mode."""
        self.live_mode = live
        if connector_fn is not None:
            self._connector = connector_fn

    def update_state(self, portfolio_state: dict[str, Any]) -> None:
        """Update internal position state after execution."""
        self._position_state = portfolio_state.get("positions", {})


class HedgeFundMultiAgent:
    """Hedge Fund Multi-Agent System — 对冲基金多智能体系统.

    Coordinates all four agents (MacroStrategist, RiskAnalyst, PortfolioManager,
    ExecutionAgent) in a unified decision cycle.

    Communication pattern (simple message passing):
        MacroStrategist → PortfolioManager   (macro signal)
        RiskAnalyst     → PortfolioManager   (risk report)
        PortfolioManager → ExecutionAgent    (allocation plan)
        ExecutionAgent  → PortfolioManager    (execution results)

    Cycle order:
        1. MacroStrategist analyzes macro regime
        2. RiskAnalyst analyzes portfolio risk
        3. PortfolioManager synthesizes allocation plan
        4. ExecutionAgent executes rebalance trades
        5. Return CycleResult

    Attributes:
        portfolio_manager: PortfolioManagerAgent instance
        risk_analyst:      RiskAnalystAgent instance
        macro_strategist:  MacroStrategistAgent instance
        execution_agent:   ExecutionAgent instance

    Methods:
        run_cycle():   Execute one full decision cycle — returns CycleResult.
    """

    def __init__(
        self,
        portfolio_manager: Optional[PortfolioManagerAgent] = None,
        risk_analyst: Optional[RiskAnalystAgent] = None,
        macro_strategist: Optional[MacroStrategistAgent] = None,
        execution_agent: Optional[ExecutionAgent] = None,
    ) -> None:
        self.portfolio_manager = portfolio_manager or PortfolioManagerAgent()
        self.risk_analyst = risk_analyst or RiskAnalystAgent()
        self.macro_strategist = macro_strategist or MacroStrategistAgent()
        self.execution_agent = execution_agent or ExecutionAgent()

        self._cycle_count: int = 0
        self._messages: list[AgentMessage] = []

    # ------------------------------------------------------------------
    # Message Bus
    # ------------------------------------------------------------------

    def _send(self, msg: AgentMessage) -> None:
        """Append a message to the internal message bus."""
        self._messages.append(msg)

    def _broadcast(self, sender: str, subject: str, payload: dict[str, Any], receivers: list[str]) -> None:
        """Broadcast a message to multiple receivers."""
        for receiver in receivers:
            self._send(AgentMessage(sender=sender, receiver=receiver, subject=subject, payload=payload))

    def _query_inbox(self, receiver: str, subject: Optional[str] = None) -> list[AgentMessage]:
        """Retrieve messages addressed to a specific agent, optionally filtered by subject."""
        return [
            m for m in self._messages
            if m.receiver == receiver and (subject is None or m.subject == subject)
        ]

    # ------------------------------------------------------------------
    # run_cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
        current_prices: dict[str, float],
        current_positions: dict[str, dict[str, Any]],
        cash: float,
        macro_context: Optional[dict[str, Any]] = None,
    ) -> CycleResult:
        """Run one full multi-agent decision cycle.

        执行一个完整的对冲基金多智能体决策周期.

        Args:
            symbols:           List of ticker symbols under management.
            price_data:       Dict mapping symbol -> list of historical closing prices.
            current_prices:   Dict mapping symbol -> current price (latest).
            current_positions: Dict mapping symbol -> {"long": int, "short": int}.
            cash:              Current cash balance.
            macro_context:     Optional external macro context dict (rates, CPI, etc.).

        Returns:
            CycleResult containing outputs from all agents and updated portfolio state.
        """
        t0 = time.time()
        self._cycle_count += 1
        cycle_id = self._cycle_count

        # ---- 1. MacroStrategist: top-down analysis ----
        macro_signal = self.macro_strategist.analyze(
            symbols=symbols,
            price_data=price_data,
            macro_context=macro_context,
        )
        self._broadcast(
            sender=self.macro_strategist.name,
            receivers=[self.portfolio_manager.name],
            subject="macro_signal",
            payload=macro_signal.to_dict(),
        )

        # ---- 2. RiskAnalyst: portfolio risk assessment ----
        risk_report = self.risk_analyst.analyze(
            symbols=symbols,
            price_data=price_data,
            current_positions=current_positions,
            current_prices=current_prices,
            cash=cash,
        )
        self._broadcast(
            sender=self.risk_analyst.name,
            receivers=[self.portfolio_manager.name],
            subject="risk_report",
            payload=risk_report.to_dict(),
        )

        # ---- 3. PortfolioManager: allocation decision ----
        # Retrieve messages from inbox (in practice, passed directly)
        allocation_plan = self.portfolio_manager.analyze(
            symbols=symbols,
            macro_signal=macro_signal,
            risk_report=risk_report,
            current_prices=current_prices,
            current_positions=current_positions,
            cash=cash,
        )
        self._broadcast(
            sender=self.portfolio_manager.name,
            receivers=[self.execution_agent.name],
            subject="allocation_plan",
            payload=allocation_plan.to_dict(),
        )

        # ---- 4. ExecutionAgent: execute rebalance trades ----
        execution_results, updated_portfolio = self.execution_agent.execute(
            rebalance_shares=allocation_plan.rebalance_shares,
            current_prices=current_prices,
            current_positions=current_positions,
            cash=cash,
        )
        self._broadcast(
            sender=self.execution_agent.name,
            receivers=[self.portfolio_manager.name],
            subject="execution_results",
            payload={"results": [r.to_dict() for r in execution_results], "portfolio": updated_portfolio},
        )

        # Update internal states
        self.portfolio_manager.update_state(updated_portfolio)
        self.execution_agent.update_state(updated_portfolio)

        duration_ms = (time.time() - t0) * 1000

        return CycleResult(
            cycle_id=cycle_id,
            macro_signal=macro_signal,
            risk_report=risk_report,
            allocation_plan=allocation_plan,
            execution_results=execution_results,
            portfolio_state=updated_portfolio,
            duration_ms=round(duration_ms, 2),
            agent_messages=[m.to_dict() for m in self._messages[-20:]],  # Last 20 messages
        )
