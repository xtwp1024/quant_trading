"""Walk-Forward Analysis — absorbed from GeneTrader.

Provides anti-overfitting validation with three methods:
- Rolling Window: fixed-size train/test, rolls forward
- Expanding Window: train grows, test fixed
- Anchored: fixed test end, multiple training window sizes

Key innovation: composite fitness penalizes overfitting
(train >> test triggers overfit_penalty).
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationPeriod:
    """Single train/test period for walk-forward analysis."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    fold_number: int

    @property
    def train_timerange(self) -> str:
        """Freqtrade-compatible timerange string for training."""
        return (
            f"{self.train_start.strftime('%Y%m%d')}"
            f"-{self.train_end.strftime('%Y%m%d')}"
        )

    @property
    def test_timerange(self) -> str:
        """Freqtrade-compatible timerange string for testing."""
        return (
            f"{self.test_start.strftime('%Y%m%d')}"
            f"-{self.test_end.strftime('%Y%m%d')}"
        )

    @property
    def train_weeks(self) -> int:
        return int((self.train_end - self.train_start).days / 7)

    @property
    def test_weeks(self) -> int:
        return int((self.test_end - self.test_start).days / 7)


class WalkForwardValidator:
    """Walk-forward validation framework for trading strategies.

    Generates train/test splits to measure out-of-sample performance
    and detect overfitting before production deployment.
    """

    def __init__(
        self,
        total_weeks: int = 52,
        train_weeks: int = 26,
        test_weeks: int = 4,
        min_train_weeks: int = 12,
        method: str = "rolling",
    ):
        if method.lower() not in ("rolling", "expanding", "anchored"):
            raise ValueError(
                f"Unknown method: {method}. Use 'rolling', 'expanding', or 'anchored'"
            )
        if train_weeks < min_train_weeks:
            raise ValueError(
                f"train_weeks ({train_weeks}) must be >= min_train_weeks ({min_train_weeks})"
            )
        if train_weeks + test_weeks > total_weeks:
            raise ValueError(
                f"train_weeks + test_weeks ({train_weeks + test_weeks}) "
                f"exceeds total_weeks ({total_weeks})"
            )

        self.total_weeks = total_weeks
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.min_train_weeks = min_train_weeks
        self.method = method.lower()

    def generate_periods(
        self, end_date: Optional[datetime] = None
    ) -> List[ValidationPeriod]:
        """Generate train/test periods for walk-forward analysis."""
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(weeks=self.total_weeks)

        if self.method == "rolling":
            return self._generate_rolling_periods(start_date, end_date)
        elif self.method == "expanding":
            return self._generate_expanding_periods(start_date, end_date)
        else:
            return self._generate_anchored_periods(start_date, end_date)

    def _generate_rolling_periods(
        self, start_date: datetime, end_date: datetime
    ) -> List[ValidationPeriod]:
        """Fixed-size train/test windows that roll forward."""
        periods = []
        fold = 0
        current_train_start = start_date

        while True:
            train_end = current_train_start + timedelta(weeks=self.train_weeks)
            test_start = train_end
            test_end = test_start + timedelta(weeks=self.test_weeks)

            if test_end > end_date:
                break

            periods.append(
                ValidationPeriod(
                    train_start=current_train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    fold_number=fold,
                )
            )
            fold += 1
            current_train_start = current_train_start + timedelta(weeks=self.test_weeks)

        return periods

    def _generate_expanding_periods(
        self, start_date: datetime, end_date: datetime
    ) -> List[ValidationPeriod]:
        """Training grows from fixed start; test window is fixed."""
        periods = []
        fold = 0
        current_train_end = start_date + timedelta(weeks=self.min_train_weeks)

        while True:
            test_start = current_train_end
            test_end = test_start + timedelta(weeks=self.test_weeks)

            if test_end > end_date:
                break

            periods.append(
                ValidationPeriod(
                    train_start=start_date,
                    train_end=current_train_end,
                    test_start=test_start,
                    test_end=test_end,
                    fold_number=fold,
                )
            )
            fold += 1
            current_train_end = current_train_end + timedelta(weeks=self.test_weeks)

        return periods

    def _generate_anchored_periods(
        self, start_date: datetime, end_date: datetime
    ) -> List[ValidationPeriod]:
        """Multiple training windows against a single fixed test period."""
        test_start = end_date - timedelta(weeks=self.test_weeks)
        test_end = end_date

        periods = []
        fold = 0
        current_train_weeks = self.min_train_weeks

        while current_train_weeks <= self.train_weeks:
            train_end = test_start
            train_start = train_end - timedelta(weeks=current_train_weeks)

            if train_start < start_date:
                train_start = start_date

            periods.append(
                ValidationPeriod(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    fold_number=fold,
                )
            )
            fold += 1
            current_train_weeks += self.test_weeks

        return periods

    def calculate_composite_fitness(
        self,
        fold_results: List[Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Combine multiple fold results into a single fitness score.

        Components:
        - test_mean:      average out-of-sample performance (40%)
        - test_min:       worst-case robustness (20%)
        - consistency:    inverse variance across folds (20%)
        - overfit_penalty: train >> test triggers penalty (20%)
        """
        if not fold_results:
            return float("-inf")

        default_weights = {
            "test_mean": 0.40,
            "test_min": 0.20,
            "consistency": 0.20,
            "overfit_penalty": 0.20,
        }
        weights = weights or default_weights

        train_scores = [r["train_fitness"] for r in fold_results if r["train_fitness"] is not None]
        test_scores = [r["test_fitness"] for r in fold_results if r["test_fitness"] is not None]

        if not test_scores:
            return float("-inf")

        test_mean = sum(test_scores) / len(test_scores)
        test_min = min(test_scores)

        if len(test_scores) > 1:
            test_variance = sum((x - test_mean) ** 2 for x in test_scores) / len(test_scores)
            consistency = 1.0 / (1.0 + test_variance ** 0.5)
        else:
            consistency = 1.0

        if train_scores and test_scores:
            train_mean = sum(train_scores) / len(train_scores)
            if train_mean > 0 and test_mean > 0:
                overfit_ratio = train_mean / test_mean
                overfit_penalty = 1.0 / (1.0 + max(0, overfit_ratio - 1.5) ** 2)
            else:
                overfit_penalty = 0.5
        else:
            overfit_penalty = 1.0

        composite = (
            weights["test_mean"] * test_mean
            + weights["test_min"] * test_min
            + weights["consistency"] * consistency
            + weights["overfit_penalty"] * overfit_penalty
        )

        return composite


def create_validator_from_settings(settings: Any) -> WalkForwardValidator:
    """Build a WalkForwardValidator from a settings object."""
    return WalkForwardValidator(
        total_weeks=getattr(settings, "total_data_weeks", 52),
        train_weeks=getattr(settings, "walk_forward_train_weeks", 26),
        test_weeks=getattr(settings, "walk_forward_test_weeks", 4),
        min_train_weeks=getattr(settings, "walk_forward_min_train", 12),
        method=getattr(settings, "walk_forward_method", "rolling"),
    )
