"""Risk Dashboard Data Generator.

Generates risk metrics and alert conditions for dashboard display:
- Portfolio risk summary
- Position risk breakdown
- Alert conditions (drawdown, VaR breach, etc.)
- Risk trend charts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RiskAlert:
    """A risk alert condition."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: Optional[int] = None


@dataclass
class DashboardData:
    """Complete dashboard data structure."""
    summary: Dict[str, Any]
    positions: List[Dict[str, Any]]
    alerts: List[RiskAlert]
    metrics: Dict[str, Any]
    equity_curve: List[float]
    drawdown_series: List[float]
    var_evolution: List[float]
    timestamp: int


class RiskDashboardGenerator:
    """Generate risk dashboard data for monitoring and display.

    Produces structured data suitable for:
    - Real-time monitoring dashboards
    - Alert generation and notification
    - Historical risk analysis
    """

    def __init__(
        self,
        drawdown_threshold: float = 0.10,
        var_threshold_95: float = 0.03,
        var_threshold_99: float = 0.05,
        exposure_warning: float = 0.80,
        exposure_critical: float = 0.95,
        correlation_warning: float = 0.70,
        position_drawdown_warning: float = 0.05,
    ):
        """
        Args:
            drawdown_threshold: Drawdown % to trigger warning alert
            var_threshold_95: VaR 95% threshold to trigger warning
            var_threshold_99: VaR 99% threshold to trigger critical alert
            exposure_warning: Exposure % to trigger warning
            exposure_critical: Exposure % to trigger critical alert
            correlation_warning: Correlation to trigger warning
            position_drawdown_warning: Position drawdown to trigger warning
        """
        self.drawdown_threshold = drawdown_threshold
        self.var_threshold_95 = var_threshold_95
        self.var_threshold_99 = var_threshold_99
        self.exposure_warning = exposure_warning
        self.exposure_critical = exposure_critical
        self.correlation_warning = correlation_warning
        self.position_drawdown_warning = position_drawdown_warning

        # Historical tracking
        self.alert_history: List[RiskAlert] = []
        self._alert_id_counter = 0

    def generate(
        self,
        risk_manager: Any,
        metrics_calculator: Any,
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> DashboardData:
        """
        Generate complete dashboard data.

        Args:
            risk_manager: UnifiedRiskManager instance
            metrics_calculator: RealTimeMetricsCalculator instance
            correlation_matrix: Optional correlation data

        Returns:
            DashboardData with all risk information
        """
        import time

        # Get risk summary from manager
        risk_summary = risk_manager.get_risk_summary()
        portfolio_risk = risk_manager.get_portfolio_risk()

        # Get metrics
        metrics = metrics_calculator.calculate()
        metrics_dict = metrics_calculator.get_metrics_dict()

        # Generate alerts
        alerts = self._generate_alerts(
            risk_summary=risk_summary,
            portfolio_risk=portfolio_risk,
            metrics=metrics,
            correlation_matrix=correlation_matrix or {},
        )

        # Position breakdown
        positions = self._generate_position_breakdown(risk_manager)

        # Equity curve and drawdown series
        equity_curve = metrics_calculator.equity_curve
        drawdown_series = self._generate_drawdown_series(equity_curve)
        var_evolution = self._generate_var_evolution(metrics_calculator)

        return DashboardData(
            summary={
                "account_balance": risk_summary["account_balance"],
                "peak_balance": risk_summary["peak_balance"],
                "total_exposure": risk_summary["total_exposure"],
                "exposure_pct": risk_summary["exposure_pct"],
                "current_drawdown": risk_summary["current_drawdown"],
                "position_count": risk_summary["position_count"],
                "daily_trades": risk_summary["daily_trades"],
                "daily_pnl": risk_summary["daily_pnl"],
            },
            positions=positions,
            alerts=alerts,
            metrics=metrics_dict,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            var_evolution=var_evolution,
            timestamp=int(time.time() * 1000),
        )

    def _generate_alerts(
        self,
        risk_summary: Dict[str, Any],
        portfolio_risk: Dict[str, Any],
        metrics: Any,
        correlation_matrix: Dict[Tuple[str, str], float],
    ) -> List[RiskAlert]:
        """Generate alerts based on risk conditions."""
        alerts = []

        # Check drawdown threshold
        current_dd = risk_summary["current_drawdown"]
        if current_dd >= self.drawdown_threshold:
            severity = AlertSeverity.CRITICAL if current_dd >= self.drawdown_threshold * 1.5 else AlertSeverity.WARNING
            alerts.append(self._create_alert(
                severity=severity,
                title="Drawdown Alert",
                message=f"Current drawdown {current_dd*100:.2f}% exceeds threshold {self.drawdown_threshold*100}%",
                value=current_dd,
                threshold=self.drawdown_threshold,
            ))

        # Check VaR breach (95%)
        var_95 = getattr(metrics, 'var_95', 0)
        if abs(var_95) >= self.var_threshold_95:
            alerts.append(self._create_alert(
                severity=AlertSeverity.WARNING,
                title="VaR 95% Alert",
                message=f"VaR 95% ({abs(var_95)*100:.2f}%) exceeds warning threshold",
                value=var_95,
                threshold=self.var_threshold_95,
            ))

        # Check VaR breach (99%)
        var_99 = getattr(metrics, 'var_99', 0)
        if abs(var_99) >= self.var_threshold_99:
            alerts.append(self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title="VaR 99% Alert",
                message=f"VaR 99% ({abs(var_99)*100:.2f}%) exceeds critical threshold",
                value=var_99,
                threshold=self.var_threshold_99,
            ))

        # Check exposure levels
        exposure_pct = risk_summary["exposure_pct"]
        if exposure_pct >= self.exposure_critical:
            alerts.append(self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title="Exposure Critical",
                message=f"Portfolio exposure {exposure_pct*100:.1f}% at critical level",
                value=exposure_pct,
                threshold=self.exposure_critical,
            ))
        elif exposure_pct >= self.exposure_warning:
            alerts.append(self._create_alert(
                severity=AlertSeverity.WARNING,
                title="Exposure Warning",
                message=f"Portfolio exposure {exposure_pct*100:.1f}% at warning level",
                value=exposure_pct,
                threshold=self.exposure_warning,
            ))

        # Check position drawdowns
        position_drawdowns = portfolio_risk.get("position_drawdowns", {})
        for symbol, dd in position_drawdowns.items():
            if dd <= -self.position_drawdown_warning:
                alerts.append(self._create_alert(
                    severity=AlertSeverity.WARNING,
                    title=f"Position Drawdown: {symbol}",
                    message=f"Position {symbol} drawdown {abs(dd)*100:.2f}% exceeds warning",
                    value=dd,
                    threshold=-self.position_drawdown_warning,
                ))

        # Check sector limits
        sector_counts = risk_summary.get("sector_counts", {})
        sector_limits = risk_summary.get("sector_limits", {})
        for sector, count in sector_counts.items():
            if sector in sector_limits:
                limit = sector_limits[sector]
                if count >= limit["max_positions"]:
                    alerts.append(self._create_alert(
                        severity=AlertSeverity.WARNING,
                        title=f"Sector Limit: {sector}",
                        message=f"Sector {sector} has {count} positions (max: {limit['max_positions']})",
                        value=float(count),
                        threshold=float(limit["max_positions"]),
                    ))

        # Store in history
        self.alert_history.extend(alerts)

        return alerts

    def _generate_position_breakdown(self, risk_manager: Any) -> List[Dict[str, Any]]:
        """Generate position risk breakdown."""
        positions = []
        for symbol, pos in risk_manager.positions.items():
            if pos.side == "long":
                pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - pos.current_price) / pos.entry_price

            exposure = pos.quantity * pos.current_price
            exposure_pct = exposure / risk_manager.account_balance if risk_manager.account_balance > 0 else 0

            positions.append({
                "symbol": symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl_pct": pnl_pct,
                "pnl_abs": (pos.current_price - pos.entry_price) * pos.quantity if pos.side == "long"
                           else (pos.entry_price - pos.current_price) * pos.quantity,
                "exposure": exposure,
                "exposure_pct": exposure_pct,
                "sector": pos.sector,
                "stop_loss": pos.stop_loss,
                "trailing_stop": pos.trailing_stop,
            })
        return positions

    def _generate_drawdown_series(self, equity_curve: List[float]) -> List[float]:
        """Generate drawdown series from equity curve."""
        if not equity_curve:
            return []
        drawdowns = []
        peak = equity_curve[0]
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            drawdowns.append(dd)
        return drawdowns

    def _generate_var_evolution(self, metrics_calculator: Any, window: int = 30) -> List[float]:
        """Generate rolling VaR evolution."""
        if len(metrics_calculator.returns) < window:
            return []
        sorted_returns = sorted(metrics_calculator.returns[-window:])
        var = sorted_returns[int(len(sorted_returns) * 0.05)]
        return [var]

    def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> RiskAlert:
        """Create a new alert."""
        import time
        self._alert_id_counter += 1
        return RiskAlert(
            alert_id=f"alert_{self._alert_id_counter}",
            severity=severity,
            title=title,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=int(time.time() * 1000),
        )

    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by severity."""
        return {
            "critical": len([a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL]),
            "warning": len([a for a in self.alert_history if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in self.alert_history if a.severity == AlertSeverity.INFO]),
        }

    def clear_alerts(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()


# Convenience functions

def create_risk_dashboard(
    risk_manager: Any,
    metrics_calculator: Any,
    **kwargs,
) -> DashboardData:
    """Create a risk dashboard from manager and calculator."""
    generator = RiskDashboardGenerator(**kwargs)
    return generator.generate(risk_manager, metrics_calculator)


def format_alert_message(alert: RiskAlert) -> str:
    """Format an alert as a human-readable message."""
    msg = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
    if alert.value is not None and alert.threshold is not None:
        msg += f" (value: {alert.value:.4f}, threshold: {alert.threshold:.4f})"
    return msg


__all__ = [
    "AlertSeverity",
    "RiskAlert",
    "DashboardData",
    "RiskDashboardGenerator",
    "create_risk_dashboard",
    "format_alert_message",
]
