"""
Backtest to Execution Bridge
=============================

将回测结果（BacktestResult）桥接到执行管道（VersionControl → Deployer）。

使用方式:
    from quant_trading.execution.version_control import StrategyVersionControl
    from quant_trading.execution.strategy_deployer import StrategyDeployer, DeploymentConfig
    from quant_trading.execution.backtest_bridge import BacktestBridge

    vc = StrategyVersionControl()
    deployer = StrategyDeployer(vc)
    bridge = BacktestBridge(vc, deployer)

    # 回测完成后，直接将结果注册为新版本
    version = bridge.register_backtest_result(
        strategy_name="TrendFollowing",
        strategy_source_file="/path/to/strategy.py",
        backtest_result=result,  # BacktestResult from backtest/engine.py
        parameters={"atr_period": 14, "atr_multiplier": 3.0},
        notes="Initial version from backtest"
    )

    # 一键部署（经 validation + 自动回滚）
    result = bridge.deploy_with_safety(
        version.version_id,
        strategy_name="TrendFollowing",
        config=DeploymentConfig(
            shadow_trading_hours=24,
            gradual_rollout=True,
            auto_rollback_enabled=True,
        )
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from quant_trading.backtest.engine import BacktestResult
from quant_trading.execution.version_control import StrategyVersionControl, StrategyVersion, VersionStatus
from quant_trading.execution.strategy_deployer import StrategyDeployer, DeploymentConfig, DeploymentResult


class BacktestBridge:
    """
    Backtest → Execution 桥接器。

    将 BacktestResult 转换为 StrategyVersion，并提供一键部署能力。
    核心流程:
        BacktestResult
            ↓ BacktestResult.to_dict()
            ↓ _to_backtest_metrics()  格式对齐
            ↓ StrategyVersionControl.create_version()
            ↓ StrategyDeployer.deploy()  (含 validation + rollback)
    """

    def __init__(
        self,
        version_control: StrategyVersionControl,
        deployer: StrategyDeployer,
    ):
        self.version_control = version_control
        self.deployer = deployer

    @staticmethod
    def _to_backtest_metrics(result: BacktestResult) -> Dict[str, Any]:
        """
        将 BacktestResult 转换为 GeneTrader 风格的 backtest_metrics 字典。

        对齐字段名:
            final_equity         → final_equity
            max_drawdown_pct     → max_drawdown  (转为小数)
            total_trades         → total_trades
            win_rate             → win_rate
            profit_factor        → profit_factor
            sharpe_ratio         → sharpe_ratio
            sortino_ratio        → sortino_ratio
            额外计算:
                total_profit_pct = (final_equity - initial) / initial * 100
        """
        m = result.to_dict()

        # 计算总收益率百分比
        initial_equity = m.get(
            "initial_equity",
            m.get("final_equity", 0) / (1 + m.get("total_profit_pct", 0) / 100),
        )
        total_profit_pct = (
            (m["final_equity"] - initial_equity) / initial_equity * 100
            if initial_equity > 0
            else 0
        )

        return {
            "final_equity": m["final_equity"],
            "total_profit_pct": total_profit_pct,
            "max_drawdown": m["max_drawdown_pct"] / 100,  # 转为小数供 VersionControl 使用
            "max_drawdown_pct": m["max_drawdown_pct"],
            "total_trades": m["total_trades"],
            "winning_trades": m["winning_trades"],
            "losing_trades": m["losing_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "sharpe_ratio": m["sharpe_ratio"],
            "sortino_ratio": m["sortino_ratio"],
            "avg_win": m["avg_win"],
            "avg_loss": m["avg_loss"],
        }

    def register_backtest_result(
        self,
        strategy_name: str,
        strategy_source_file: str,
        backtest_result: BacktestResult,
        parameters: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> StrategyVersion:
        """
        将回测结果注册为策略新版本。

        等价于:
            version_control.create_version(
                strategy_name=strategy_name,
                source_file=strategy_source_file,
                parameters=parameters,
                backtest_metrics=BacktestBridge._to_backtest_metrics(backtest_result),
                notes=notes
            )
        """
        backtest_metrics = self._to_backtest_metrics(backtest_result)

        # 基本验证：回测盈利且回撤可接受才允许注册
        if backtest_metrics["total_profit_pct"] <= 0:
            raise ValueError(
                f"Backtest profit is negative: {backtest_metrics['total_profit_pct']:.2f}%"
            )
        if backtest_metrics["max_drawdown"] > 0.5:
            raise ValueError(
                f"Backtest drawdown too high: {backtest_metrics['max_drawdown']:.1%}"
            )

        version = self.version_control.create_version(
            strategy_name=strategy_name,
            source_file=strategy_source_file,
            parameters=parameters or {},
            backtest_metrics=backtest_metrics,
            notes=notes,
        )
        return version

    def deploy_with_safety(
        self,
        version_id: str,
        strategy_name: str,
        config: Optional[DeploymentConfig] = None,
    ) -> DeploymentResult:
        """
        使用安全管道部署策略版本。

        包含:
        1. Validation（回测指标检查）
        2. Gradual Rollout（可选）
        3. 自动回滚（性能下降超过阈值时触发）
        """
        config = config or DeploymentConfig()
        return self.deployer.deploy(
            strategy_name=strategy_name,
            version_id=version_id,
            config=config,
        )
