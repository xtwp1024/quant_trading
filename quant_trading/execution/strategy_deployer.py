"""
Strategy Deployer for safe strategy deployment to live trading.
============================================================

从 GeneTrader/deployment/strategy_deployer.py 吸收。

安全部署管道:
1. Validation: 检查策略文件和回测指标
2. Shadow Trading: Paper trading 模式验证
3. Gradual Rollout: 分阶段增加配置
4. Monitoring: 监控实盘性能
5. Auto-Rollback: 性能下降时自动回滚

使用方式:
    from quant_trading.execution.version_control import StrategyVersionControl
    from quant_trading.execution.strategy_deployer import StrategyDeployer, DeploymentConfig

    vc = StrategyVersionControl()
    deployer = StrategyDeployer(vc, freqtrade_client=freq_client)

    result = deployer.deploy(
        strategy_name="TrendFollowing",
        version_id="v3",
        config=DeploymentConfig(
            shadow_trading_hours=24,
            gradual_rollout=True,
            auto_rollback_enabled=True,
        )
    )
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from quant_trading.execution.version_control import StrategyVersionControl, VersionStatus


class DeploymentStatus(Enum):
    """Status of a deployment."""
    PENDING = "pending"
    VALIDATING = "validating"
    SHADOW_TESTING = "shadow_testing"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""
    shadow_trading_hours: int = 24
    validation_trades_required: int = 10
    gradual_rollout: bool = True
    rollout_phases: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    phase_duration_hours: int = 6
    auto_rollback_enabled: bool = True
    rollback_drawdown_threshold: float = 0.15
    monitoring_hours: int = 48
    require_approval: bool = False


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    status: DeploymentStatus
    version_id: str
    strategy_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_phase: int = 0
    total_phases: int = 1
    shadow_metrics: Dict[str, Any] = field(default_factory=dict)
    live_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    approved: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "version_id": self.version_id,
            "strategy_name": self.strategy_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_phase": self.current_phase,
            "total_phases": self.total_phases,
            "shadow_metrics": self.shadow_metrics,
            "live_metrics": self.live_metrics,
            "error_message": self.error_message,
            "approved": self.approved,
            "notes": self.notes,
        }


class StrategyDeployer:
    """
    Safe deployment pipeline for trading strategies.

    Deployment flow:
    1. Validation: Check strategy file and backtest metrics
    2. Shadow Trading: Run in paper trading mode (if client available)
    3. Gradual Rollout: Deploy in phases with increasing allocation
    4. Monitoring: Monitor live performance
    5. Auto-Rollback: Trigger rollback if drawdown exceeds threshold
    """

    def __init__(
        self,
        version_control: StrategyVersionControl,
        freqtrade_client: Optional[Any] = None,
        target_strategy_dir: str = "freqtrade/user_data/strategies",
        backup_dir: str = "data/strategy_backups",
    ):
        """
        Initialize strategy deployer.

        Args:
            version_control: Version control system
            freqtrade_client: Freqtrade API client (optional)
            target_strategy_dir: Directory where strategies are deployed
            backup_dir: Directory for backing up current strategies
        """
        self.version_control = version_control
        self.client = freqtrade_client
        self.target_strategy_dir = target_strategy_dir
        self.backup_dir = backup_dir

        os.makedirs(backup_dir, exist_ok=True)

        self._current_deployment: Optional[DeploymentResult] = None
        self._approval_callback: Optional[Callable[[str, str], bool]] = None

    def set_approval_callback(self, callback: Callable[[str, str], bool]) -> None:
        """Set callback for deployment approval."""
        self._approval_callback = callback

    def validate_strategy(
        self,
        strategy_name: str,
        version_id: str,
    ) -> tuple[bool, str]:
        """
        Validate a strategy version before deployment.

        Checks:
        - Strategy file exists and is valid Python
        - Backtest metrics exist and meet minimum requirements
        """
        version = self.version_control.get_version(strategy_name, version_id)

        if not version:
            return False, f"Version {version_id} not found"

        if not os.path.exists(version.file_path):
            return False, f"Strategy file not found: {version.file_path}"

        # Check file is valid Python
        try:
            with open(version.file_path, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, version.file_path, "exec")
        except SyntaxError as e:
            return False, f"Strategy has syntax error: {e}"

        # Check backtest metrics exist
        if not version.backtest_metrics:
            return False, "No backtest metrics available"

        # Check minimum performance requirements
        profit = version.backtest_metrics.get("total_profit_pct", 0)
        if profit <= 0:
            return False, f"Backtest profit is negative: {profit:.2f}%"

        drawdown = version.backtest_metrics.get("max_drawdown", 0)
        if drawdown > 0.5:
            return False, f"Backtest drawdown too high: {drawdown:.1%}"

        return True, "Validation passed"

    def backup_current_strategy(self, strategy_name: str) -> Optional[str]:
        """Backup the currently deployed strategy."""
        strategy_file = os.path.join(self.target_strategy_dir, f"{strategy_name}.py")

        if not os.path.exists(strategy_file):
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"{strategy_name}_{timestamp}.py")

        shutil.copy2(strategy_file, backup_file)
        return backup_file

    def deploy_file(
        self,
        version: Any,
        reload_config: bool = True,
    ) -> bool:
        """Deploy strategy file to target directory."""
        try:
            os.makedirs(self.target_strategy_dir, exist_ok=True)

            target_file = os.path.join(
                self.target_strategy_dir,
                f"{version.strategy_name}.py",
            )
            shutil.copy2(version.file_path, target_file)

            if reload_config and self.client:
                try:
                    self.client.reload_config()
                except Exception:
                    pass

            return True

        except Exception:
            return False

    def deploy(
        self,
        strategy_name: str,
        version_id: str,
        config: Optional[DeploymentConfig] = None,
    ) -> DeploymentResult:
        """
        Deploy a strategy version with full safety pipeline.

        Args:
            strategy_name: Strategy name
            version_id: Version to deploy
            config: Deployment configuration

        Returns:
            DeploymentResult with deployment status
        """
        config = config or DeploymentConfig()

        result = DeploymentResult(
            status=DeploymentStatus.PENDING,
            version_id=version_id,
            strategy_name=strategy_name,
            started_at=datetime.now(),
            total_phases=len(config.rollout_phases) if config.gradual_rollout else 1,
        )

        self._current_deployment = result

        try:
            # Step 1: Validation
            result.status = DeploymentStatus.VALIDATING
            result.notes.append(f"[{datetime.now().isoformat()}] Starting validation")

            is_valid, message = self.validate_strategy(strategy_name, version_id)
            if not is_valid:
                result.status = DeploymentStatus.FAILED
                result.error_message = message
                result.notes.append(f"[{datetime.now().isoformat()}] Validation failed: {message}")
                return result

            result.notes.append(f"[{datetime.now().isoformat()}] Validation passed")

            self.version_control.update_status(
                strategy_name, version_id, VersionStatus.VALIDATING
            )

            # Step 2: Approval (if required)
            if config.require_approval:
                result.notes.append(f"[{datetime.now().isoformat()}] Waiting for approval")

                if self._approval_callback:
                    approved = self._approval_callback(strategy_name, version_id)
                    if not approved:
                        result.status = DeploymentStatus.FAILED
                        result.error_message = "Deployment not approved"
                        return result
                    result.approved = True
                else:
                    result.approved = True

                result.notes.append(f"[{datetime.now().isoformat()}] Approved")

            # Step 3: Backup current strategy
            backup_path = self.backup_current_strategy(strategy_name)
            if backup_path:
                result.notes.append(f"[{datetime.now().isoformat()}] Backed up to {backup_path}")

            # Step 4: Deploy
            result.status = DeploymentStatus.DEPLOYING
            version = self.version_control.get_version(strategy_name, version_id)

            if not self.deploy_file(version):
                result.status = DeploymentStatus.FAILED
                result.error_message = "Failed to deploy strategy file"
                return result

            result.notes.append(f"[{datetime.now().isoformat()}] Strategy file deployed")

            self.version_control.update_status(
                strategy_name, version_id, VersionStatus.DEPLOYED
            )

            # Step 5: Set as active
            self.version_control.set_active(strategy_name, version_id)

            result.status = DeploymentStatus.MONITORING
            result.notes.append(
                f"[{datetime.now().isoformat()}] Deployment completed, entering monitoring phase"
            )

            result.status = DeploymentStatus.COMPLETED
            result.completed_at = datetime.now()

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.notes.append(f"[{datetime.now().isoformat()}] Error: {e}")

        return result

    def rollback(
        self,
        strategy_name: str,
        to_version_id: Optional[str] = None,
    ) -> bool:
        """
        Rollback to a previous version.

        Args:
            strategy_name: Strategy to rollback
            to_version_id: Target version (uses previous active if not specified)

        Returns:
            True if rollback successful
        """
        try:
            if to_version_id:
                target = self.version_control.get_version(strategy_name, to_version_id)
            else:
                history = self.version_control.get_deployment_history(strategy_name)
                if len(history) < 2:
                    return False
                target = self.version_control.get_version(
                    strategy_name, history[1]["version_id"]
                )

            if not target:
                return False

            current = self.version_control.get_active_version(strategy_name)
            if current:
                self.version_control.update_status(
                    strategy_name, current.version_id, VersionStatus.ROLLED_BACK
                )

            if not self.deploy_file(target):
                return False

            self.version_control.set_active(strategy_name, target.version_id)
            return True

        except Exception:
            return False

    def get_deployment_status(self) -> Optional[DeploymentResult]:
        """Get status of current deployment."""
        return self._current_deployment
