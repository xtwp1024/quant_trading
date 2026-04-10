"""
Strategy Pool Manager
=====================

Manages strategy pool with automatic addition and removal based on performance.
Auto-remove underperforming strategies, auto-add new strategies that pass criteria.

Criteria for inclusion:
- Sharpe > 0.5
- Win Rate > 45%
- Total Trades > 100
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from .strategy_tracker import StrategyTracker


class PoolStatus(Enum):
    """Strategy status in the pool."""
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    REMOVED = "removed"
    PROBATION = "probation"


@dataclass
class PoolConfig:
    """Configuration for strategy pool management."""
    # Inclusion criteria
    min_sharpe: float = 0.5
    min_win_rate: float = 45.0
    min_trades: int = 100

    # Removal thresholds
    max_drawdown_removal: float = 40.0  # %
    min_roi_removal: float = -10.0     # %
    min_sharpe_removal: float = 0.2

    # Review periods
    review_period_days: int = 7
    probation_period_days: int = 14

    # Pool limits
    max_pool_size: int = 20
    min_pool_size: int = 3


@dataclass
class StrategyPoolEntry:
    """Entry in the strategy pool."""
    strategy_name: str
    symbol: str
    timeframe: str
    status: PoolStatus

    # Performance metrics
    sharpe_30d: float = 0.0
    win_rate_30d: float = 0.0
    roi_30d: float = 0.0
    total_trades_30d: int = 0
    max_drawdown_30d: float = 0.0

    # Allocation
    allocated_weight: float = 0.0

    # Timestamps
    added_at: datetime = field(default_factory=datetime.now)
    last_reviewed: datetime = field(default_factory=datetime.now)
    status_since: datetime = field(default_factory=datetime.now)

    # History
    review_count: int = 0
    consecutive_failures: int = 0


class StrategyPoolManager:
    """
    Strategy Pool Manager with Auto-Rebalancing

    Manages the strategy pool with:
    - Automatic addition of new strategies that meet criteria
    - Automatic removal of underperforming strategies
    - Dynamic allocation based on recent performance
    - Regime-aware strategy selection
    """

    def __init__(
        self,
        tracker: Optional[StrategyTracker] = None,
        config: Optional[PoolConfig] = None,
        db_path: str = "optimization/strategy_pool.json"
    ):
        self.tracker = tracker or StrategyTracker()
        self.config = config or PoolConfig()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.pool: Dict[str, StrategyPoolEntry] = {}
        self.regime: str = "unknown"
        self.last_rebalance: datetime = datetime.now()

        self._load_pool()

    def _load_pool(self) -> None:
        """Load pool from disk."""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.pool = {}
            for name, entry_data in data.get('pool', {}).items():
                entry_data['added_at'] = datetime.fromisoformat(entry_data['added_at'])
                entry_data['last_reviewed'] = datetime.fromisoformat(entry_data['last_reviewed'])
                entry_data['status_since'] = datetime.fromisoformat(entry_data['status_since'])
                entry_data['status'] = PoolStatus(entry_data['status'])
                self.pool[name] = StrategyPoolEntry(**entry_data)

            self.regime = data.get('regime', 'unknown')
            self.last_rebalance = datetime.fromisoformat(
                data.get('last_rebalance', datetime.now().isoformat())
            )

    def _save_pool(self) -> None:
        """Save pool to disk."""
        pool_data = {}
        for name, entry in self.pool.items():
            pool_data[name] = {
                'strategy_name': entry.strategy_name,
                'symbol': entry.symbol,
                'timeframe': entry.timeframe,
                'status': entry.status.value,
                'sharpe_30d': entry.sharpe_30d,
                'win_rate_30d': entry.win_rate_30d,
                'roi_30d': entry.roi_30d,
                'total_trades_30d': entry.total_trades_30d,
                'max_drawdown_30d': entry.max_drawdown_30d,
                'allocated_weight': entry.allocated_weight,
                'added_at': entry.added_at.isoformat(),
                'last_reviewed': entry.last_reviewed.isoformat(),
                'status_since': entry.status_since.isoformat(),
                'review_count': entry.review_count,
                'consecutive_failures': entry.consecutive_failures
            }

        data = {
            'pool': pool_data,
            'regime': self.regime,
            'last_rebalance': self.last_rebalance.isoformat(),
            'saved_at': datetime.now().isoformat()
        }

        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def check_strategy_passes_criteria(
        self,
        strategy_name: str,
        symbol: str = "BTC-USDT",
        timeframe: str = "15m"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a strategy meets inclusion criteria.

        Returns:
            (passes, details_dict)
        """
        metrics = self.tracker.get_rolling_metrics(
            strategy_name, symbol, timeframe
        ).get('30d', {})

        sharpe = metrics.get('sharpe', 0)
        win_rate = metrics.get('win_rate', 0)
        trades = metrics.get('total_trades', 0)

        passes_sharpe = sharpe >= self.config.min_sharpe
        passes_win_rate = win_rate >= self.config.min_win_rate
        passes_trades = trades >= self.config.min_trades

        passes = passes_sharpe and passes_win_rate and passes_trades

        details = {
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_trades': trades,
            'min_sharpe': self.config.min_sharpe,
            'min_win_rate': self.config.min_win_rate,
            'min_trades': self.config.min_trades,
            'passes_sharpe': passes_sharpe,
            'passes_win_rate': passes_win_rate,
            'passes_trades': passes_trades,
            'passes_all': passes
        }

        return passes, details

    def add_strategy(
        self,
        strategy_name: str,
        symbol: str = "BTC-USDT",
        timeframe: str = "15m"
    ) -> bool:
        """
        Add a strategy to the pool if it meets criteria.

        Returns:
            True if added, False if not eligible
        """
        # Check if already in pool
        if strategy_name in self.pool:
            return False

        # Check criteria
        passes, details = self.check_strategy_passes_criteria(
            strategy_name, symbol, timeframe
        )

        if not passes:
            return False

        # Check pool size limit
        if len(self.pool) >= self.config.max_pool_size:
            # Try to remove underperforming first
            self._remove_worst_strategy()

        # Get latest metrics
        metrics_30d = self.tracker.get_rolling_metrics(
            strategy_name, symbol, timeframe
        ).get('30d', {})

        # Create entry
        entry = StrategyPoolEntry(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            status=PoolStatus.ACTIVE,
            sharpe_30d=metrics_30d.get('sharpe', 0),
            win_rate_30d=metrics_30d.get('win_rate', 0),
            roi_30d=metrics_30d.get('roi', 0),
            total_trades_30d=metrics_30d.get('total_trades', 0),
            max_drawdown_30d=metrics_30d.get('max_drawdown', 0)
        )

        self.pool[strategy_name] = entry
        self._rebalance_allocations()
        self._save_pool()

        return True

    def remove_strategy(
        self,
        strategy_name: str,
        reason: str = "underperformance"
    ) -> bool:
        """Remove a strategy from the pool."""
        if strategy_name not in self.pool:
            return False

        entry = self.pool[strategy_name]
        entry.status = PoolStatus.REMOVED
        entry.status_since = datetime.now()

        # Move to tracker as removed
        self.tracker.update_strategy_status(strategy_name, 'removed')

        # Remove from pool
        del self.pool[strategy_name]

        self._rebalance_allocations()
        self._save_pool()

        return True

    def _remove_worst_strategy(self) -> Optional[str]:
        """Remove the worst performing active strategy."""
        if not self.pool:
            return None

        active = [
            (name, entry) for name, entry in self.pool.items()
            if entry.status == PoolStatus.ACTIVE
        ]

        if not active:
            return None

        # Sort by Sharpe ratio (lowest first)
        active.sort(key=lambda x: x[1].sharpe_30d)

        worst_name, worst_entry = active[0]

        # Only remove if below removal thresholds
        if (worst_entry.roi_30d < self.config.min_roi_removal or
            worst_entry.sharpe_30d < self.config.min_sharpe_removal or
            worst_entry.max_drawdown_30d > self.config.max_drawdown_removal):

            self.remove_strategy(worst_name, "auto_removed_worst")
            return worst_name

        return None

    def update_pool_metrics(self) -> None:
        """Update all strategy metrics from tracker."""
        for strategy_name, entry in self.pool.items():
            metrics = self.tracker.get_rolling_metrics(
                strategy_name, entry.symbol, entry.timeframe
            )

            for period in ['7d', '30d', '90d']:
                period_metrics = metrics.get(period, {})

                if period == '30d':
                    entry.sharpe_30d = period_metrics.get('sharpe', 0)
                    entry.win_rate_30d = period_metrics.get('win_rate', 0)
                    entry.roi_30d = period_metrics.get('roi', 0)
                    entry.total_trades_30d = period_metrics.get('total_trades', 0)
                    entry.max_drawdown_30d = period_metrics.get('max_drawdown', 0)

            entry.last_reviewed = datetime.now()

        self._save_pool()

    def review_and_update_pool(self) -> Dict[str, Any]:
        """
        Review the pool and make necessary updates.

        - Remove underperforming strategies
        - Move strategies to probation if needed
        - Re-enable strategies from probation if improved

        Returns:
            Dictionary with review results
        """
        self.update_pool_metrics()

        actions_taken = []
        review_time = datetime.now()

        for strategy_name, entry in list(self.pool.items()):
            entry.review_count += 1

            # Check removal criteria
            should_remove = (
                entry.roi_30d < self.config.min_roi_removal or
                entry.sharpe_30d < self.config.min_sharpe_removal or
                entry.max_drawdown_30d > self.config.max_drawdown_removal
            )

            if should_remove and entry.status == PoolStatus.ACTIVE:
                entry.consecutive_failures += 1

                if entry.consecutive_failures >= 2:
                    # Move to under review first
                    entry.status = PoolStatus.UNDER_REVIEW
                    entry.status_since = review_time
                    actions_taken.append({
                        'strategy': strategy_name,
                        'action': 'under_review',
                        'reason': f"ROI={entry.roi_30d:.2f}%, Sharpe={entry.sharpe_30d:.2f}"
                    })
                else:
                    actions_taken.append({
                        'strategy': strategy_name,
                        'action': 'probation_warning',
                        'reason': f"First failure: ROI={entry.roi_30d:.2f}%"
                    })

            elif should_remove and entry.status == PoolStatus.UNDER_REVIEW:
                entry.status = PoolStatus.REMOVED
                entry.status_since = review_time
                actions_taken.append({
                    'strategy': strategy_name,
                    'action': 'removed',
                    'reason': f"Still underperforming: ROI={entry.roi_30d:.2f}%"
                })
                # Actually remove from pool
                del self.pool[strategy_name]
                self.tracker.update_strategy_status(strategy_name, 'removed')

            elif not should_remove and entry.status in (PoolStatus.UNDER_REVIEW, PoolStatus.PROBATION):
                # Recovered - back to active
                entry.status = PoolStatus.ACTIVE
                entry.status_since = review_time
                entry.consecutive_failures = 0
                actions_taken.append({
                    'strategy': strategy_name,
                    'action': 'recovered',
                    'reason': f"Improved: Sharpe={entry.sharpe_30d:.2f}"
                })

            elif not should_remove:
                # Reset failures for active strategies
                entry.consecutive_failures = 0

        # Re-balance allocations
        self._rebalance_allocations()
        self._save_pool()

        return {
            'review_time': review_time.isoformat(),
            'actions': actions_taken,
            'pool_size': len(self.pool),
            'regime': self.regime
        }

    def _rebalance_allocations(self) -> None:
        """
        Rebalance strategy allocations based on Sharpe ratios.
        Uses risk-adjusted weighting.
        """
        active_strategies = [
            entry for entry in self.pool.values()
            if entry.status == PoolStatus.ACTIVE
        ]

        if not active_strategies:
            return

        # Calculate total risk-adjusted score
        total_score = sum(max(0, entry.sharpe_30d) for entry in active_strategies)

        if total_score <= 0:
            # Equal weighting if all Sharpe <= 0
            weight = 1.0 / len(active_strategies)
            for entry in active_strategies:
                entry.allocated_weight = weight
        else:
            # Risk-adjusted weighting
            for entry in active_strategies:
                entry.allocated_weight = max(0, entry.sharpe_30d) / total_score

        # Normalize to sum to 1
        total_weight = sum(entry.allocated_weight for entry in active_strategies)
        if total_weight > 0:
            for entry in active_strategies:
                entry.allocated_weight /= total_weight

    def set_regime(self, regime: str) -> None:
        """Set current market regime."""
        self.regime = regime
        self.tracker.record_regime(regime, 0.5, 0.5)
        self._save_pool()

    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get list of active strategies with their allocations."""
        self._rebalance_allocations()

        result = []
        for entry in self.pool.values():
            if entry.status == PoolStatus.ACTIVE:
                result.append({
                    'strategy_name': entry.strategy_name,
                    'symbol': entry.symbol,
                    'timeframe': entry.timeframe,
                    'weight': entry.allocated_weight,
                    'sharpe_30d': entry.sharpe_30d,
                    'win_rate_30d': entry.win_rate_30d,
                    'roi_30d': entry.roi_30d,
                    'total_trades_30d': entry.total_trades_30d
                })

        result.sort(key=lambda x: x['sharpe_30d'], reverse=True)
        return result

    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies regardless of status."""
        result = []
        for entry in self.pool.values():
            result.append({
                'strategy_name': entry.strategy_name,
                'symbol': entry.symbol,
                'timeframe': entry.timeframe,
                'status': entry.status.value,
                'weight': entry.allocated_weight,
                'sharpe_30d': entry.sharpe_30d,
                'win_rate_30d': entry.win_rate_30d,
                'roi_30d': entry.roi_30d,
                'total_trades_30d': entry.total_trades_30d,
                'added_at': entry.added_at.isoformat(),
                'last_reviewed': entry.last_reviewed.isoformat()
            })

        return result

    def get_pool_summary(self) -> Dict[str, Any]:
        """Get summary of the pool."""
        active = [e for e in self.pool.values() if e.status == PoolStatus.ACTIVE]
        under_review = [e for e in self.pool.values() if e.status == PoolStatus.UNDER_REVIEW]
        probation = [e for e in self.pool.values() if e.status == PoolStatus.PROBATION]

        return {
            'total_strategies': len(self.pool),
            'active_count': len(active),
            'under_review_count': len(under_review),
            'probation_count': len(probation),
            'regime': self.regime,
            'last_rebalance': self.last_rebalance.isoformat(),
            'config': {
                'min_sharpe': self.config.min_sharpe,
                'min_win_rate': self.config.min_win_rate,
                'min_trades': self.config.min_trades
            },
            'total_allocated_weight': sum(e.allocated_weight for e in active),
            'avg_sharpe_active': np.mean([e.sharpe_30d for e in active]) if active else 0
        }

    def auto_discover_and_add(self, available_strategies: List[str]) -> Dict[str, Any]:
        """
        Automatically discover and add new strategies that meet criteria.

        Args:
            available_strategies: List of strategy names to check

        Returns:
            Summary of discovery and addition results
        """
        added = []
        not_eligible = []

        for strategy_name in available_strategies:
            if strategy_name in self.pool:
                continue

            passes, details = self.check_strategy_passes_criteria(strategy_name)

            if passes:
                if self.add_strategy(strategy_name):
                    added.append(strategy_name)
            else:
                not_eligible.append({
                    'strategy': strategy_name,
                    'sharpe': details['sharpe'],
                    'win_rate': details['win_rate'],
                    'trades': details['total_trades']
                })

        return {
            'added': added,
            'not_eligible': not_eligible,
            'total_checked': len(available_strategies)
        }

    def generate_report(self) -> str:
        """Generate a human-readable pool report."""
        summary = self.get_pool_summary()
        active_strategies = self.get_active_strategies()

        lines = [
            "=" * 80,
            "STRATEGY POOL REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Regime: {summary['regime']}",
            f"Last Rebalance: {summary['last_rebalance']}",
            "",
            "-" * 40,
            "POOL SUMMARY",
            "-" * 40,
            f"Total Strategies: {summary['total_strategies']}",
            f"Active: {summary['active_count']}",
            f"Under Review: {summary['under_review_count']}",
            f"Probation: {summary['probation_count']}",
            f"Avg Sharpe (Active): {summary['avg_sharpe_active']:.2f}",
            "",
            "-" * 40,
            "ACTIVE STRATEGIES",
            "-" * 40,
        ]

        if active_strategies:
            lines.append(
                f"{'Strategy':<25} {'Weight':>8} {'Sharpe':>8} {'WinRate':>8} {'ROI':>10}"
            )
            lines.append("-" * 80)

            for s in active_strategies:
                lines.append(
                    f"{s['strategy_name']:<25} "
                    f"{s['weight']:>7.2%} "
                    f"{s['sharpe_30d']:>8.2f} "
                    f"{s['win_rate_30d']:>7.1f}% "
                    f"{s['roi_30d']:>9.2f}%"
                )
        else:
            lines.append("  No active strategies in pool.")

        lines.extend([
            "",
            "-" * 40,
            "INCLUSION CRITERIA",
            "-" * 40,
            f"  Min Sharpe: {summary['config']['min_sharpe']}",
            f"  Min Win Rate: {summary['config']['min_win_rate']}%",
            f"  Min Trades: {summary['config']['min_trades']}",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)


def main():
    """Test the strategy pool manager."""
    print("\n" + "="*80)
    print("Strategy Pool Manager Test")
    print("="*80)

    # Initialize tracker with sample data
    tracker = StrategyTracker()
    print("\n[1] Adding sample data to tracker...")
    tracker.add_sample_data()

    # Initialize pool manager
    config = PoolConfig(
        min_sharpe=0.5,
        min_win_rate=45.0,
        min_trades=100,
        max_pool_size=20
    )
    manager = StrategyPoolManager(
        tracker=tracker,
        config=config,
        db_path="optimization/strategy_pool.json"
    )

    # Add some strategies
    print("\n[2] Adding strategies to pool...")
    strategies_to_add = [
        "TrendFollowing_EMAs",
        "MeanReversion_RSI",
        "GridTrading_BTC",
        "Breakout_Squeeze",
    ]

    for strat in strategies_to_add:
        success = manager.add_strategy(strat)
        status = "Added" if success else "Not eligible"
        print(f"    {strat}: {status}")

    # Get pool summary
    print("\n[3] Pool Summary:")
    summary = manager.get_pool_summary()
    for key, value in summary.items():
        if key != 'config':
            print(f"    {key}: {value}")

    # Get active strategies
    print("\n[4] Active Strategies:")
    active = manager.get_active_strategies()
    for s in active:
        print(f"    {s['strategy_name']}: Weight={s['weight']:.2%}, "
              f"Sharpe={s['sharpe_30d']:.2f}")

    # Review and update
    print("\n[5] Reviewing pool...")
    review_result = manager.review_and_update_pool()
    print(f"    Actions taken: {len(review_result['actions'])}")
    for action in review_result['actions']:
        print(f"      - {action['strategy']}: {action['action']}")

    # Test auto-discover
    print("\n[6] Auto-discover test:")
    all_strategies = [
        "TrendFollowing_EMAs", "MeanReversion_RSI", "GridTrading_BTC",
        "Breakout_Squeeze", "MarketMaker_PRO", "NewStrategy_XYZ"
    ]
    discover_result = manager.auto_discover_and_add(all_strategies)
    print(f"    Checked: {discover_result['total_checked']}")
    print(f"    Added: {discover_result['added']}")

    # Generate report
    print("\n[7] Pool Report:")
    print(manager.generate_report())

    print("\n" + "="*80)
    print("Pool Manager test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
