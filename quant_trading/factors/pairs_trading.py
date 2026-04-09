"""
quant_trading.factors.pairs_trading — Dynamic hedge ratio pairs trading strategies.

Based on KalmanBOT_ICASSP23 (ICASSP 2023) - Kalman Filter / KalmanNet for pairs trading

Classes
-------
PairsTradingStrategy : Base class for mean-reversion pairs trading.
KalmanFilterStrategy : Linear Kalman Filter for hedge ratio estimation.
KalmanNetStrategy : KalmanNet (GRU-enhanced) for adaptive hedge ratio estimation.
HedgeRatioEstimator : sklearn-compatible hedge ratio wrapper.
"""

from __future__ import annotations

from typing import Optional, Tuple, NamedTuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from quant_trading.factors.kalman_filter import (
    KalmanFilter,
    LinearSystemModel,
    KalmanNetNN,
    KalmanNetSystemModel,
    _DEV,
)


# =============================================================================
# Data structures
# =============================================================================
class PairsPosition(NamedTuple):
    """Trading position for a pairs trade."""

    hedge_ratio: float
    intercept: float
    spread: float
    spread_std: float
    signal: int  # 1 = long spread, -1 = short spread, 0 = flat
    units: float  # position size in spread units


# =============================================================================
# Position signal models
# =============================================================================
class BollingerSignal:
    """
    Bollinger band entry/exit for spread mean-reversion.

    - Long spread when spread < -threshold * spread_std
    - Short spread when spread > +threshold * spread_std
    - Exit when spread crosses zero
    """

    def __init__(self, threshold: float = 1.0):
        """
        Parameters
        ----------
        threshold : float
            Number of standard deviations for entry (default 1.0).
        """
        self.threshold = threshold

    def compute_signal(
        self, spread: np.ndarray, spread_var: np.ndarray
    ) -> np.ndarray:
        """
        Compute trading signals from spread and variance.

        Parameters
        ----------
        spread : (T,) estimated spread (innovation)
        spread_var : (T,) spread variance estimate

        Returns
        -------
        signals : (T,) position signals (-1, 0, 1)
        """
        spread_std = np.sqrt(spread_var)
        signals = np.zeros_like(spread, dtype=np.int32)

        long_entry = spread < -self.threshold * spread_std
        short_entry = spread > self.threshold * spread_std
        long_exit = spread >= 0
        short_exit = spread <= 0

        # Forward-fill signals
        positions = np.zeros_like(spread, dtype=np.float64)

        positions[long_entry] = 1.0
        positions[long_exit] = 0.0
        positions[short_entry] = -1.0
        positions[short_exit] = 0.0

        # Fill forward
        df = _pd_DataFrame(positions)
        df.fillna(method="ffill", inplace=True)
        df.fillna(0.0, inplace=True)

        return df.values.flatten().astype(np.int32)


class LearnableBollingerSignal(nn.Module):
    """
    Learnable Bollinger band thresholds via differentiable programming.

    Threshold is learned end-to-end with the KalmanNet model to maximize
    strategy Sharpe ratio.
    """

    def __init__(self, threshold_init: float = 0.01, scale_init: float = 50.0):
        super().__init__()
        self.threshold = torch.tensor(threshold_init, requires_grad=True)
        self.scale = torch.tensor(scale_init, requires_grad=True)

    def forward(
        self, dy: torch.Tensor, S: torch.Tensor, prev_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute position from innovation, covariance, and previous position.

        Parameters
        ----------
        dy : innovation (spread residual)
        S : innovation variance estimate
        prev_pos : previous position (-1, 0, 1)

        Returns
        -------
        new_pos : new position
        """
        threshold = torch.abs(self.threshold) + 1e-6
        spread_std = torch.sqrt(S.abs() + 1e-6) * self.scale

        long_entry = dy < -threshold * spread_std
        short_entry = dy > threshold * spread_std
        long_exit = dy >= 0
        short_exit = dy <= 0

        pos = torch.zeros_like(prev_pos)
        pos = torch.where(long_entry, torch.ones_like(pos), pos)
        pos = torch.where(long_exit, torch.zeros_like(pos), pos)
        pos = torch.where(short_entry, -torch.ones_like(pos), pos)
        pos = torch.where(short_exit, torch.zeros_like(pos), pos)
        pos = torch.where(pos == 0, prev_pos, pos)

        return pos


# =============================================================================
# Pairs Trading Strategy Base
# =============================================================================
class PairsTradingStrategy:
    """
    Base class for mean-reversion pairs trading with dynamic hedge ratio.

    Workflow
    --------
    1. estimate_hedge_ratio() -> beta (hedge_ratio, intercept)
    2. compute_spread() -> innovation sequence
    3. generate_signals() -> entry/exit positions
    4. compute_pnl() -> profit and loss

    Subclasses must implement _estimate_impl() for specific estimation methods.
    """

    def __init__(
        self,
        delta: float = 1e-5,
        r2: float = 1e-4,
        bollinger_threshold: float = 1.0,
        position_size: float = 1.0,
    ):
        """
        Parameters
        ----------
        delta : float
            State transition variance parameter.
            delta=1 gives fastest hedge ratio change.
            delta->0 recovers static OLS hedge ratio.
        r2 : float
            Observation noise variance (scaled).
        bollinger_threshold : float
            Standard deviations for Bollinger band entry.
        position_size : float
            Base position size in spread units.
        """
        self.delta = delta
        self.r2 = r2
        self.bollinger_threshold = bollinger_threshold
        self.position_size = position_size

        self._fitted = False
        self._beta_0: Optional[torch.Tensor] = None
        self._R_0: Optional[torch.Tensor] = None
        self._F: Optional[torch.Tensor] = None
        self._H: Optional[torch.Tensor] = None

    def fit(
        self, x: np.ndarray, y: np.ndarray, train_size: int, traj_length: int
    ) -> "PairsTradingStrategy":
        """
        Fit the hedge ratio model on training data.

        Parameters
        ----------
        x : (T, 2) array, first asset price + constant term
        y : (T,) array, second asset price (spread target)
        train_size : int, number of training observations
        traj_length : int, trajectory length for batch training

        Returns
        -------
        self
        """
        # Linear regression for initial hedge ratio
        reg = LinearRegression()
        reg.fit(x[:train_size], y[:train_size])
        hedge_ratio_init = reg.coef_[0] if x.shape[1] > 1 else reg.coef_
        intercept_init = reg.intercept_

        self._beta_0 = torch.tensor(
            [[hedge_ratio_init], [intercept_init]], dtype=torch.float32
        )
        self._R_0 = torch.zeros(2, 2)

        q2 = self.delta / (1 - self.delta)
        self._q2 = q2

        self._F = torch.eye(2)
        self._H = torch.tensor([[1.0, 1.0]])

        self._init_model(x[:train_size], y[:train_size])

        self._fitted = True
        return self

    def _init_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Initialize the specific estimation model. Override in subclass."""
        raise NotImplementedError

    def predict(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run prediction on test data.

        Parameters
        ----------
        x : (T, 2) array
        y : (T,) array

        Returns
        -------
        innovations : (T,) spread residuals (innovation sequence)
        variances : (T,) innovation variances
        beta : (T, 2) hedge ratios over time
        """
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")

        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        x_aug = np.column_stack([np.ones(len(x)), x]) if x.ndim == 1 else x
        x_t = torch.tensor(x_aug, dtype=torch.float32)

        innovations, variances, beta = self._predict_impl(y_t, x_t)

        return (
            innovations.squeeze().cpu().numpy(),
            variances.squeeze().cpu().numpy(),
            beta.squeeze().cpu().numpy(),
        )

    def _predict_impl(
        self, y: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run model-specific prediction. Override in subclass."""
        raise NotImplementedError

    def get_positions(
        self, innovations: np.ndarray, variances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert spread estimates to trading positions.

        Returns
        -------
        signals : (T,) entry/exit signals (-1, 0, 1)
        units : (T,) position sizes
        """
        signal_model = BollingerSignal(threshold=self.bollinger_threshold)
        signals = signal_model.compute_signal(innovations, variances)
        units = signals * self.position_size
        return signals, units

    def compute_pnl(
        self,
        signals: np.ndarray,
        units: np.ndarray,
        beta: np.ndarray,
        prices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute profit and loss for a pairs trade.

        Parameters
        ----------
        signals : (T,) trading signals
        units : (T,) position sizes
        beta : (T, 2) hedge ratios [hedge_ratio, intercept]
        prices : (T, 2) price series [asset1, asset2]

        Returns
        -------
        pnl : (T,) daily P&L
        cum_pnl : (T,) cumulative P&L
        """
        # position = -beta * units for asset1, +units for asset2
        hedge = -beta[:, 0] * units
        position = np.column_stack([hedge, units])

        # Price changes
        price_diff = np.diff(prices.T).T
        pnl = np.zeros_like(signals, dtype=np.float64)
        pnl[1:] = np.sum(price_diff[:-1] * position[:-1], axis=1)

        cum_pnl = np.cumsum(pnl)
        cum_pnl[0] = 0.0

        return pnl, cum_pnl

    def compute_spread(
        self, y: np.ndarray, beta: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """
        Compute spread series: y - beta * x.

        For pairs trading: spread = y_asset - hedge_ratio * x_asset - intercept
        """
        if x.ndim == 1:
            x = np.column_stack([np.ones(len(x)), x])
        return y - np.sum(beta * x, axis=1)


# =============================================================================
# Kalman Filter Strategy
# =============================================================================
class KalmanFilterStrategy(PairsTradingStrategy):
    """
    Pairs trading with linear Kalman Filter for hedge ratio estimation.

    State: x = [beta, intercept]^T
    Observation: y = H @ x + v (spread)

    The Kalman Filter provides:
    - Online hedge ratio updates
    - Innovation (spread residual) sequence
    - Innovation variance for signal generation
    """

    def _init_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Initialize linear Kalman Filter."""
        import statsmodels.api as sm

        x_aug = sm.add_constant(x_train)
        x_aug_t = torch.tensor(x_aug, dtype=torch.float32)

        q2 = self._q2
        r2 = self.r2

        self._ss_model = LinearSystemModel(
            F=self._F,
            q=np.sqrt(q2),
            H=x_aug_t,
            r=np.sqrt(r2),
            T=len(y_train),
            T_test=1,
            hedge=1,
        )
        self._ss_model.init_sequence(self._beta_0, self._R_0)

        self._kf = KalmanFilter(self._ss_model, ratio=1)
        self._kf.init_sequence(self._ss_model.m1x_0, self._ss_model.m2x_0)

    def _predict_impl(
        self, y: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Kalman Filter prediction."""
        T = y.shape[1]

        # Reconstruct with full H matrix for test
        x_aug = torch.cat([torch.ones(T, 1), x], dim=1)
        self._ss_model.H = x_aug

        self._kf.T_test = T
        self._kf.init_sequence(self._ss_model.m1x_0, self._ss_model.m2x_0)

        beta_all = torch.empty(2, T)
        innovations = torch.empty(T)
        variances = torch.empty(T)

        m1x = self._ss_model.m1x_0.clone()
        m2x = self._ss_model.m2x_0.clone()

        for t in range(T):
            yt = y[:, t : t + 1]
            H_t = x_aug[t : t + 1]
            H_T_t = H_t.t()

            self._kf.m1x_posterior = m1x
            self._kf.m2x_posterior = m2x

            self._kf.predict(H_t, H_T_t)
            self._kf.k_gain(H_T_t)
            self._kf.innovate(yt)
            self._kf.correct()

            m1x = self._kf.m1x_posterior
            m2x = self._kf.m2x_posterior

            beta_all[:, t] = m1x.squeeze()
            innovations[t] = self._kf.dy.squeeze()
            variances[t] = self._kf.m2y.squeeze()

        return innovations, variances, beta_all


# =============================================================================
# KalmanNet Strategy
# =============================================================================
class KalmanNetStrategy(PairsTradingStrategy):
    """
    Pairs trading with KalmanNet (GRU-enhanced) for adaptive hedge ratio.

    KalmanNet learns the Kalman gain jointly with the state estimation,
    providing more robust hedge ratio estimates in non-linear or
    non-stationary market conditions.

    Reference: ICASSP 2023 KalmanBOT
    """

    def __init__(
        self,
        delta: float = 1e-5,
        r2: float = 1e-4,
        bollinger_threshold: float = 1.0,
        position_size: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        n_epochs: int = 10,
    ):
        super().__init__(delta, r2, bollinger_threshold, position_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self._model: Optional[KalmanNetNN] = None

    def _init_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Initialize KalmanNet model."""
        q2 = self._q2
        r2 = self.r2

        f = lambda x: torch.matmul(self._F, x)

        def h(x):
            return torch.matmul(self._H, x)

        self._ss_model_kn = KalmanNetSystemModel(
            F=self._F,
            f=f,
            q=np.sqrt(q2),
            H=self._H,
            h=h,
            r=np.sqrt(r2),
            T=len(y_train),
            hedge=1,
        )
        self._ss_model_kn.init_sequence(self._beta_0, self._R_0)

        self._model = KalmanNetNN()
        self._model.build(self._ss_model_kn)

    def _predict_impl(
        self, y: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run KalmanNet prediction."""
        if self._model is None:
            raise ValueError("Model not initialized")

        T = y.shape[1]

        # Prepare augmented data [y, x_aug]
        x_aug = torch.cat([torch.ones(T, 1), x], dim=1)
        data = torch.cat([y.t(), x_aug], dim=1).t().unsqueeze(0)  # (1, 3, T)

        self._model.init_hidden()
        self._model.init_sequence(self._ss_model_kn.m1x_0, T)

        beta_all = torch.empty(2, T)
        innovations = torch.empty(T)
        variances = torch.empty(T)

        with torch.no_grad():
            self._model.eval()
            for t in range(T):
                yt = data[:, :, t : t + 1]  # (1, 3, 1)
                beta = self._model(yt)
                beta_all[:, t] = beta

                if hasattr(self._model, "dy"):
                    innovations[t] = self._model.dy.squeeze()
                if hasattr(self._model, "S_t"):
                    variances[t] = torch.abs(self._model.S_t).squeeze()

        return innovations, variances, beta_all


# =============================================================================
# sklearn-compatible HedgeRatioEstimator
# =============================================================================
class HedgeRatioEstimator(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible hedge ratio estimator using Kalman Filter.

    Provides a familiar interface for integration with sklearn pipelines
    and factor evaluation frameworks.

    Parameters
    ----------
    delta : float, default=1e-5
        State transition variance parameter.
    r2 : float, default=1e-4
        Observation noise variance.

    Example
    -------
    >>> from quant_trading.factors.pairs_trading import HedgeRatioEstimator
    >>> estimator = HedgeRatioEstimator(delta=1e-5, r2=1e-4)
    >>> estimator.fit(X_train, y_train)
    >>> hedge_ratios = estimator.predict(X_test)
    """

    def __init__(self, delta: float = 1e-5, r2: float = 1e-4):
        self.delta = delta
        self.r2 = r2
        self._strategy: Optional[KalmanFilterStrategy] = None
        self._beta_history: Optional[np.ndarray] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "HedgeRatioEstimator":
        """
        Fit the Kalman Filter hedge ratio model.

        Parameters
        ----------
        X : (n_samples, n_features) price series of first asset
        y : (n_samples,) price series of second asset

        Returns
        -------
        self
        """
        # Use first 50% for training
        n_train = len(y) // 2
        self._strategy = KalmanFilterStrategy(delta=self.delta, r2=self.r2)
        self._strategy.fit(X, y, train_size=n_train, traj_length=n_train)

        _, _, beta = self._strategy.predict(X[n_train:], y[n_train:])
        self._beta_history = beta

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hedge ratios.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        hedge_ratios : (n_samples,) dynamic hedge ratios
        """
        if self._strategy is None:
            raise ValueError("Must call fit() before predict()")

        innovations, variances, beta = self._strategy.predict(X, np.zeros(len(X)))
        self._beta_history = beta
        return beta[:, 0]

    @property
    def innovations_(self) -> Optional[np.ndarray]:
        """Last predicted innovation (spread) sequence."""
        if self._strategy is None:
            return None
        return self._beta_history

    @property
    def intercepts_(self) -> Optional[np.ndarray]:
        """Last predicted intercept sequence."""
        if self._strategy is None or self._beta_history is None:
            return None
        return self._beta_history[1, :] if self._beta_history.ndim > 1 else None


# =============================================================================
# Helper utilities
# =============================================================================
def _pd_DataFrame(data: np.ndarray):
    """Lazy pandas import for signal forward-fill."""
    import pandas as pd
    return pd.DataFrame(data)


def prepare_forex_data(
    df_train: np.ndarray, asset1_col: int = 0, asset2_col: int = 1
) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Prepare Kalman Filter parameters from forex data.

    Returns
    -------
    q2 : process noise variance
    r2 : observation noise variance
    R_0 : initial state covariance (2x2 zeros)
    beta_0 : initial hedge ratio [beta, intercept]
    """
    reg = LinearRegression()
    reg.fit(df_train[:, asset1_col].reshape(-1, 1), df_train[:, asset2_col])
    hedge_ratio = reg.coef_.item()
    intercept = reg.intercept_.item()

    residuals = df_train[:, asset2_col] - reg.predict(
        df_train[:, asset1_col].reshape(-1, 1)
    )
    delta = 1e-5
    q2 = delta / (1 - delta)
    r2 = residuals.var() * 1e-4

    R_0 = torch.zeros(2, 2)
    beta_0 = torch.tensor([[hedge_ratio], [intercept]], dtype=torch.float32)

    return q2, r2, R_0, beta_0


def compute_innovation_portfolio(
    prices: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """
    Compute spread (innovation) for a pairs portfolio.

    spread_t = price2_t - sum_i(beta_i * price1_i_t) - intercept

    Parameters
    ----------
    prices : (T, 2) price series
    beta : (T, 2) hedge ratios over time

    Returns
    -------
    spread : (T,) spread series
    """
    return (
        prices[:, 1]
        - beta[:, 0] * prices[:, 0]
        - (beta[:, 1] if beta.ndim > 1 else 0)
    )
