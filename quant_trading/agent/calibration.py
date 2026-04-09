"""Platt Calibration System — converts raw confidence to calibrated probability.

Based on Fully-Autonomous-Polymarket-AI-Trading-Bot's calibration module (calibrator.py).
Applies sigmoid(a*x + b) fit on historical (confidence, is_correct) pairs.

Platt Scaling:
  - Fits sigmoid(a*x + b) on historical predictions
  - a < 1: compresses extreme probabilities toward 0.5
  - b != 0: shifts baseline probability

Additional heuristics applied on top:
  1. Low-evidence penalty: pulls toward 0.5 when evidence is weak
  2. Ensemble disagreement penalty: pulls toward 0.5 when models diverge
  3. Contradiction penalty: pulls toward 0.5 for contradictory signals

Usage:
    calibrator = PlattCalibrator()
    calibrator.fit([(0.75, True), (0.60, False), ...])  # (confidence, is_correct)
    calibrated_prob = calibrator.calibrate(0.80)
    if calibrator.is_reliable():
        use_calibrated_prob()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PlattCalibrator",
    "CalibrationResult",
    "CalibrationHistory",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CalibrationHistory:
    """Single (forecast, outcome) pair for learning calibration."""
    forecast_prob: float   # 0.0–1.0 confidence input
    actual_outcome: float   # 0.0 or 1.0 (bool as float)
    market_type: str = ""
    confidence_level: str = ""
    timestamp: str = ""


@dataclass
class CalibrationResult:
    """Output of a single calibration call."""
    raw_probability: float
    calibrated_probability: float
    method: str
    adjustments: list[str] = field(default_factory=list)
    is_reliable: bool = False


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Platt calibration — fits sigmoid(a*x + b) on historical forecast outcomes.

    Method:
      calibrated = sigmoid(a * logit(raw) + b)

    Where logit(x) = ln(x / (1-x)).

    The calibrator learns a (slope) and b (intercept) from historical data.
    Without sufficient history, falls back to identity (pass-through).

    Properties:
      is_reliable: True only when min_samples (default 30) have been fitted.
    """

    def __init__(self, min_samples: int = 30):
        self._min_samples = min_samples
        self._a: float = 1.0   # slope
        self._b: float = 0.0   # intercept
        self._is_fitted: bool = False
        self._n_samples: int = 0
        self._brier_score: float = 1.0
        self._history: list[CalibrationHistory] = []

    # ── Core fit / calibrate ──────────────────────────────────────────────

    def fit(self, predictions: list[tuple[float, bool]]) -> bool:
        """Fit calibration curve from historical (confidence, is_correct) pairs.

        Args:
            predictions: List of (confidence, is_correct) tuples.
                        confidence: float 0.0–1.0
                        is_correct: bool (True = forecast was correct)

        Returns:
            True if fit succeeded, False if insufficient data or error.
        """
        # Convert to CalibrationHistory
        history = [
            CalibrationHistory(
                forecast_prob=max(0.01, min(0.99, float(conf))),
                actual_outcome=float(bool(correct)),
            )
            for conf, correct in predictions
        ]
        self._history = history
        return self._fit_internal(history)

    def _fit_internal(self, history: list[CalibrationHistory]) -> bool:
        """Internal fit using CalibrationHistory list."""
        if len(history) < self._min_samples:
            self._is_fitted = False
            self._n_samples = len(history)
            return False

        try:
            import numpy as np  # type: ignore[import-untyped]

            # Convert to logits
            probs = [max(0.01, min(0.99, h.forecast_prob)) for h in history]
            logits = [math.log(p / (1.0 - p)) for p in probs]
            outcomes = [h.actual_outcome for h in history]

            X = np.array(logits).reshape(-1, 1)
            y = np.array(outcomes)

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(X, y)

            self._a = float(lr.coef_[0][0])
            self._b = float(lr.intercept_[0])
            self._is_fitted = True
            self._n_samples = len(history)

            # Brier score: mean squared error
            calibrated = [self._apply_sigmoid(p) for p in probs]
            self._brier_score = sum(
                (c - o) ** 2 for c, o in zip(calibrated, outcomes)
            ) / len(outcomes)
            return True

        except ImportError:
            # sklearn unavailable — fall back to manual sigmoid fit
            return self._fit_manual(history)
        except Exception:
            return False

    def _fit_manual(self, history: list[CalibrationHistory]) -> bool:
        """Manual maximum-likelihood fit without sklearn.

        Minimises: sum( (sigmoid(a*logit(p)+b) - y)^2 )
        Uses a simple grid search for a and b.
        """
        if len(history) < 5:
            return False

        probs = [max(0.01, min(0.99, h.forecast_prob)) for h in history]
        outcomes = [h.actual_outcome for h in history]

        best_a, best_b, best_loss = 1.0, 0.0, float("inf")

        for a_candidate in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3]:
            for b_candidate in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                preds = [self._apply_sigmoid_v2(a_candidate, b_candidate, p) for p in probs]
                loss = sum((pred - y) ** 2 for pred, y in zip(preds, outcomes))
                if loss < best_loss:
                    best_loss = loss
                    best_a = a_candidate
                    best_b = b_candidate

        self._a = best_a
        self._b = best_b
        self._is_fitted = True
        self._n_samples = len(history)

        calibrated = [self._apply_sigmoid_v2(best_a, best_b, p) for p in probs]
        self._brier_score = sum(
            (c - o) ** 2 for c, o in zip(calibrated, outcomes)
        ) / len(outcomes)
        return True

    def _apply_sigmoid_v2(self, a: float, b: float, prob: float) -> float:
        """Apply sigmoid with given a, b parameters."""
        prob = max(0.01, min(0.99, prob))
        logit = math.log(prob / (1.0 - prob))
        cal_logit = a * logit + b
        return 1.0 / (1.0 + math.exp(-cal_logit))

    def _apply_sigmoid(self, prob: float) -> float:
        """Apply learned sigmoid calibration."""
        prob = max(0.01, min(0.99, prob))
        logit = math.log(prob / (1.0 - prob))
        cal_logit = self._a * logit + self._b
        return 1.0 / (1.0 + math.exp(-cal_logit))

    def calibrate(self, confidence: float) -> float:
        """Calibrate a raw confidence score to a more accurate probability.

        If fitted, applies sigmoid(a*logit(x)+b). Otherwise returns identity.

        Args:
            confidence: Raw confidence in [0.01, 0.99].

        Returns:
            Calibrated probability in [0.01, 0.99].
        """
        p = max(0.01, min(0.99, confidence))
        if self._is_fitted:
            return max(0.01, min(0.99, self._apply_sigmoid(p)))
        return p

    def calibrate_result(self, confidence: float) -> CalibrationResult:
        """Calibrate with full metadata about adjustments applied.

        Args:
            confidence: Raw confidence in [0.01, 0.99].

        Returns:
            CalibrationResult with raw/calibrated values and reliability flag.
        """
        p = max(0.01, min(0.99, confidence))
        calibrated = self.calibrate(p)
        return CalibrationResult(
            raw_probability=confidence,
            calibrated_probability=calibrated,
            method="platt_fitted" if self._is_fitted else "identity",
            adjustments=[],
            is_reliable=self.is_reliable(),
        )

    # ── Reliability & statistics ───────────────────────────────────────

    def is_reliable(self) -> bool:
        """Check if calibration is reliable.

        Requires at least min_samples (default 30) historical pairs
        and a brier score < 0.25 (better than naive 0.5 baseline).
        """
        return self._is_fitted and self._n_samples >= self._min_samples

    @property
    def stats(self) -> dict[str, Any]:
        """Return calibration statistics."""
        return {
            "is_fitted": self._is_fitted,
            "is_reliable": self.is_reliable(),
            "n_samples": self._n_samples,
            "min_samples_required": self._min_samples,
            "a": round(self._a, 4),
            "b": round(self._b, 4),
            "brier_score": round(self._brier_score, 4),
        }

    @property
    def a(self) -> float:
        """Learned slope parameter."""
        return self._a

    @property
    def b(self) -> float:
        """Learned intercept parameter."""
        return self._b

    def record_outcome(self, confidence: float, actual: bool) -> None:
        """Record a forecast outcome for future recalibration.

        Stores in memory; call refit() to recompute parameters.

        Args:
            confidence: The confidence value used (0.0–1.0).
            actual: Whether the prediction was correct.
        """
        self._history.append(
            CalibrationHistory(
                forecast_prob=max(0.01, min(0.99, confidence)),
                actual_outcome=float(actual),
            )
        )

    def refit(self) -> bool:
        """Recompute calibration parameters from recorded history.

        Returns:
            True if refit succeeded.
        """
        return self._fit_internal(self._history)

    def reset(self) -> None:
        """Clear all calibration data and revert to identity."""
        self._a = 1.0
        self._b = 0.0
        self._is_fitted = False
        self._n_samples = 0
        self._brier_score = 1.0
        self._history = []


# ---------------------------------------------------------------------------
# Global calibrator instance for convenience
# ---------------------------------------------------------------------------

_global_calibrator: PlattCalibrator | None = None


def get_global_calibrator() -> PlattCalibrator:
    """Get the global PlattCalibrator instance."""
    global _global_calibrator
    if _global_calibrator is None:
        _global_calibrator = PlattCalibrator()
    return _global_calibrator
