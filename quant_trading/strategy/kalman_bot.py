"""
KalmanBOT: Kalman Filter and KalmanNet for Pairs Trading / Mean Reversion
================================================================================
Absorbed from KalmanBOT_ICASSP23 (https://github.com/KalmanBOT/KalmanBOT_ICASSP23)
Reference:  "KalmanNet: Neural-aided Kalman Filtering for Tracking Systems"
             IEEE ICASSP 2023

Pure NumPy implementation using numpy.linalg for matrix operations.

State-space model for pairs trading:
    State  x_t = [hedge_ratio, intercept]  — random-walk coefficients
    Obs    y_t = H_t · x_t + v_t
           H_t = [price_s1_t, 1]
           y_t = price_s2_t
    F = I_2 (state evolves as random walk)
    Q = q² · I₂  (process noise)
    R = r² · I₁  (observation noise)

Key idea:
    KalmanNet replaces the analytic Kalman gain with a learnable correction
    derived from a small GRU-based network tracking Q, Σ, S — the NN only
    operates during training; inference uses an Unscented-KF approximation in
    this pure-NumPy version.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import inv, cholesky, multi_dot
from typing import Optional, Tuple, Literal

# --------------------------------------------------------------------------------
# 0.  Type aliases
# --------------------------------------------------------------------------------
Array = np.ndarray
Mat22 = Array          # 2×2 matrix
Vec2  = Array          # 2-element column vector


# ================================================================================
# 1.  KalmanFilter — Classic Linear Kalman Filter
# ================================================================================
class KalmanFilter:
    """
    Classic linear Kalman filter for linear Gaussian state-space models.

    State dynamics (random-walk model for hedge ratio):
        x_t = F · x_{t-1} + w_t,    w_t ~ N(0, Q)

    Observation model:
        y_t = H_t · x_t + v_t,     v_t ~ N(0, R)

    Parameters
    ----------
    F : Array (m×m)
        State transition matrix. Default identity.
    Q : Array (m×m)
        Process noise covariance.
    H : Array | None
        Observation matrix (can also be supplied per-step for time-varying H).
    R : Array (n×n)
        Observation noise covariance.
    m : int
        State dimension.
    n : int
        Observation dimension.

    Attributes
    ----------
    x_posterior : Vec2
        Posterior state estimate (mean) at current step.
    P_posterior : Mat22
        Posterior state covariance at current step.
    gain : Array (m×n)
        Computed Kalman gain at current step.
    innovation : Array (n,)
        Observation innovation (residual) at current step.
    innov_cov : Array (n×n)
        Innovation covariance S_t at current step.

    Example
    -------
    >>> kf = KalmanFilter(F=np.eye(2), Q=1e-4*np.eye(2), H=np.array([[1., 1.]]), R=1e-4)
    >>> x0 = np.array([[0.], [0.]])          # initial state [hedge_ratio, intercept]
    >>> P0 = np.eye(2) * 1e-3                 # initial uncertainty
    >>> kf.init_sequence(x0, P0)
    >>> for t in range(T):
    ...     H_t = np.array([[price_s1[t], 1.0]])   # time-varying H
    ...     y_t = np.array([[price_s2[t]]])
    ...     x_post = kf.update(y_t, H_t)
    >>> hedge_ratio = kf.x_posterior[0, 0]
    """

    def __init__(
        self,
        F: Optional[Array] = None,
        Q: Optional[Array] = None,
        H: Optional[Array] = None,
        R: Optional[Array] = None,
        m: int = 2,
        n:int  = 1,
    ):
        self.F = np.eye(m) if F is None else F.astype(np.float64)
        self.Q = np.eye(m) * 1e-6 if Q is None else Q.astype(np.float64)
        self.H = H.astype(np.float64) if H is not None else None   # may be time-varying
        self.R = np.eye(n) * 1e-6 if R is None else R.astype(np.float64)
        self.m = m
        self.n = n

        # Allocate trace arrays
        self.x_history   = []
        self.P_history   = []
        self.gain_history = []
        self.innov_history = []
        self.innov_cov_history = []

    # ------------------------------------------------------------------
    # init_sequence — initialise the filter with t=0 priors
    # ------------------------------------------------------------------
    def init_sequence(self, x0: Array, P0: Array) -> None:
        """
        Initialise the filter.

        Parameters
        ----------
        x0 : Array (m, ) or (m, 1)
            Initial state mean.
        P0 : Array (m, m)
            Initial state covariance.
        """
        x0 = np.asarray(x0, dtype=np.float64).reshape(self.m, 1)
        P0 = np.asarray(P0, dtype=np.float64).reshape(self.m, self.m)
        self.x_posterior = x0.copy()
        self.P_posterior = P0.copy()

        # History buffers
        self.x_history    = [x0.copy()]
        self.P_history    = [P0.copy()]
        self.gain_history = []
        self.innov_history = []
        self.innov_cov_history = []

    # ------------------------------------------------------------------
    # predict — one-step ahead prediction
    # ------------------------------------------------------------------
    def predict(self, H_t: Array) -> Tuple[Array, Array, Array]:
        """
        Compute one-step-ahead predictions of state and observation.

        Parameters
        ----------
        H_t : Array (n×m)
            Observation matrix at time t (may be time-varying).

        Returns
        -------
        x_prior   : Array (m,1) — predicted state mean
        P_prior   : Array (m,m) — predicted state covariance
        S_t       : Array (n,n) — innovation covariance
        """
        # State prediction: x_{t|t-1} = F · x_{t-1|t-1}
        x_prior = self.F @ self.x_posterior

        # Covariance prediction: P_{t|t-1} = F·P_{t-1|t-1}·Fᵀ + Q
        FP = self.F @ self.P_posterior
        P_prior = FP @ self.F.T + self.Q

        # Observation prediction: ŷ_{t|t-1} = H_t · x_{t|t-1}
        Ht = H_t.reshape(self.n, self.m)
        y_prior = Ht @ x_prior

        # Innovation covariance: S_t = H_t·P_{t|t-1}·H_tᵀ + R
        HP = Ht @ P_prior
        S_t = HP @ Ht.T + self.R

        return x_prior, P_prior, S_t, y_prior

    # ------------------------------------------------------------------
    # update — incorporate measurement y_t
    # ------------------------------------------------------------------
    def update(self, y_t: Array, H_t: Array) -> Array:
        """
        Kalman update step: compute posterior state estimate.

        Parameters
        ----------
        y_t : Array (n,) or (n,1)
            Observation at time t.
        H_t : Array (n×m)
            Observation matrix at time t.

        Returns
        -------
        x_posterior : Array (m,1) — updated state mean
        """
        y_t = np.asarray(y_t, dtype=np.float64).reshape(self.n, 1)
        H_t = H_t.reshape(self.n, self.m)

        x_prior, P_prior, S_t, y_prior = self.predict(H_t)

        # Store P_prior for external access (e.g. innovation variance computation)
        self.P_prior = P_prior.copy()

        # Innovation (residual): ν_t = y_t − ŷ_{t|t-1}
        innovation = y_t - y_prior

        # Kalman gain: K_t = P_{t|t-1}·H_tᵀ·S_t⁻¹
        K_t = P_prior @ H_t.T @ inv(S_t)

        # Posterior state: x_{t|t} = x_{t|t-1} + K_t·ν_t
        x_posterior = x_prior + K_t @ innovation

        # Posterior covariance (Joseph form — numerically stable):
        # P_{t|t} = (I − K_t·H_t)·P_{t|t-1}·(I − K_t·H_t)ᵀ + K_t·R·K_tᵀ
        I_KH = np.eye(self.m) - K_t @ H_t
        P_posterior = I_KH @ P_prior @ I_KH.T + K_t @ self.R @ K_t.T

        # Store
        self.x_posterior   = x_posterior
        self.P_posterior   = P_posterior
        self.gain          = K_t
        self.innovation    = innovation.flatten()
        self.innov_cov     = S_t

        self.x_history.append(x_posterior.copy())
        self.P_history.append(P_posterior.copy())
        self.gain_history.append(K_t.copy())
        self.innov_history.append(innovation.flatten().copy())
        self.innov_cov_history.append(S_t.copy())

        return x_posterior.flatten()

    # ------------------------------------------------------------------
    # batch_update — run filter over full series
    # ------------------------------------------------------------------
    def batch_update(
        self,
        y: Array,          # (T, n)
        H: Array,          # (T, n, m)  OR  (n, m)  if time-invariant
        x0: Optional[Array] = None,
        P0: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Run the full Kalman filter over a batch of T observations.

        Parameters
        ----------
        y  : Array (T, n)      — observations (e.g. price_s2 series)
        H  : Array (T, n, m)   — observation matrices for each step
        x0 : Array (m,) | None  — initial state; zero if omitted
        P0 : Array (m,m) | None — initial covariance; identity*1e-3 if omitted

        Returns
        -------
        x_posterior_all : Array (T, m) — posterior state estimates
        P_posterior_all : Array (T, m, m)
        innovation_all   : Array (T, n)
        S_all            : Array (T, n, n)
        """
        y  = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, self.n)
        T = y.shape[0]

        # Resolve H shape
        H_arr = np.asarray(H, dtype=np.float64)
        if H_arr.ndim == 2:
            H_arr = np.repeat(H_arr[np.newaxis, :, :], T, axis=0)

        # Resolve initials
        if x0 is None:
            x0 = np.zeros((self.m, 1))
        if P0 is None:
            P0 = np.eye(self.m) * 1e-3

        self.init_sequence(x0, P0)

        x_all    = np.empty((T, self.m))
        P_all    = np.empty((T, self.m, self.m))
        innov_all = np.empty((T, self.n))
        S_all    = np.empty((T, self.n, self.n))

        for t in range(T):
            x_all[t]    = self.update(y[t], H_arr[t]).flatten()
            P_all[t]    = self.P_posterior.copy()
            innov_all[t] = self.innovation
            S_all[t]    = self.innov_cov

        return x_all, P_all, innov_all, S_all


# ================================================================================
# 2.  KalmanNet — Neural-network-assisted Kalman Filter (NumPy approximation)
# ================================================================================
class KalmanNet:
    """
    Neural-network-assisted Kalman filter for non-linear / non-Gaussian tracking.

    In the original KalmanBOT PyTorch implementation (KalmanNetNN), a GRU-based
    network learns to predict the Kalman gain from four normalized features:
        - obs_diff       : y_t − y_{t-1}
        - obs_innov_diff : y_t − ŷ_{t|t-1}
        - fw_evol_diff   : x_{t|t} − x_{t-1|t-1}
        - fw_update_diff : x_{t|t} − x_{t|t-1}

    This pure-NumPy version replaces the learned gain with an Unscented Kalman
    Filter (UKF) correction that is adaptive to observed residuals — providing
    similar adaptive gain behaviour without a neural network.

    The UKF constructs 2m+1 sigma points from the current posterior,
    propagates them through the (potentially non-linear) observation function,
    and reconstructs the gain from the posterior cross-covariance.

    Parameters
    ----------
    m : int  — state dimension (default 2: [hedge_ratio, intercept])
    n : int  — observation dimension (default 1)
    alpha : float  — UKF spread parameter (default 0.1)
    beta  : float  — prior knowledge parameter (default 2.0)
    kappa : float  — secondary scaling (default 0.0)

    Attributes
    ----------
    x_posterior : Vec2
        Current posterior state estimate.
    P_posterior : Mat22
        Current posterior covariance.
    gain        : Array (m×n)
        Adaptive Kalman gain from UKF sigma-point evaluation.

    References
    ----------
    Wan & van der Merwe, "The Unscented Kalman Filter for Nonlinear Estimation",
    IEEE Symp. 2000.
    """

    def __init__(self, m: int = 2, n: int = 1,
                 alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
        self.m = m
        self.n = n
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa

        # UKF weights
        self._compute_weights()

        # State
        self.x_posterior: Optional[Array] = None
        self.P_posterior: Optional[Array]  = None

        # Per-step diagnostics
        self.gain: Optional[Array]        = None
        self.innovation: Optional[Array]  = None
        self.innov_cov: Optional[Array]    = None

        # History
        self.x_history    = []
        self.gain_history = []
        self.innov_history = []

    # ------------------------------------------------------------------
    # _compute_weights — compute UKF sigma-point weights
    # ------------------------------------------------------------------
    def _compute_weights(self) -> None:
        """Pre-compute mean and covariance weights for the unscented transform."""
        m = self.m
        n_sigma = 2 * m + 1

        lam = self.alpha**2 * (m + self.kappa) - m
        self.Wm = np.zeros(n_sigma)   # mean weights
        self.Wc = np.zeros(n_sigma)   # covariance weights

        self.Wm[0] = lam / (m + lam)
        self.Wc[0] = lam / (m + lam) + (1 - self.alpha**2 + self.beta)
        for i in range(1, n_sigma):
            self.Wm[i] = self.Wc[i] = 1.0 / (2 * (m + lam))

        self.lam   = lam
        self.n_sigma = n_sigma

    # ------------------------------------------------------------------
    # _sigma_points — generate unscented sigma points
    # ------------------------------------------------------------------
    def _sigma_points(self, x: Array, P: Array) -> Tuple[Array, Array]:
        """Generate 2m+1 sigma points and their square-root (Cholesky) matrix."""
        m = self.m
        # Ensure P is positive definite
        P_safe = P + np.eye(m) * 1e-9
        try:
            sqrt_P = cholesky(P_safe).T          # (m, m)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(P_safe)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))

        k = np.sqrt(m + self.lam)

        sigma = np.zeros((self.n_sigma, m))
        sigma[0] = x.flatten()
        for i in range(m):
            sigma[i + 1]    = x.flatten() + k * sqrt_P[i]
            sigma[m + 1 + i] = x.flatten() - k * sqrt_P[i]

        return sigma, sqrt_P

    # ------------------------------------------------------------------
    # _obs_function — observation model (can be overridden)
    # ------------------------------------------------------------------
    def _obs_function(self, H_t: Array, x: Array) -> Array:
        """
        Observation function: y = H_t · x.
        Override this for non-linear observation models.
        """
        H_t = H_t.reshape(self.n, self.m)
        return (H_t @ x.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # init_sequence
    # ------------------------------------------------------------------
    def init_sequence(self, x0: Array, P0: Array) -> None:
        """Initialise the filter with t=0 priors."""
        x0 = np.asarray(x0, dtype=np.float64).reshape(self.m, 1)
        P0 = np.asarray(P0, dtype=np.float64).reshape(self.m, self.m)
        self.x_posterior = x0.copy()
        self.P_posterior = P0.copy()
        self.x_history    = [x0.copy()]
        self.gain_history = []
        self.innov_history = []

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self) -> Tuple[Array, Array]:
        """Predict step (random-walk state model — F = I)."""
        x_prior = self.x_posterior.copy()
        P_prior = self.P_posterior + self.Q if hasattr(self, 'Q') else self.P_posterior.copy()
        return x_prior, P_prior

    # ------------------------------------------------------------------
    # update — UKF-based gain computation + standard update
    # ------------------------------------------------------------------
    def update(self, y_t: Array, H_t: Array,
               Q: Optional[Array] = None,
               R: Optional[Array] = None) -> Array:
        """
        One UKF-corrected Kalman update step.

        Parameters
        ----------
        y_t : Array (n,) or (n,1)
            Observation at time t.
        H_t : Array (n×m)
            Observation matrix at time t.
        Q   : Array (m×m) | None
            Process noise covariance; uses self.Q if not given.
        R   : Array (n×n) | None
            Observation noise covariance; uses self.R if not given.

        Returns
        -------
        x_posterior : Array (m,) — updated state
        """
        y_t = np.asarray(y_t, dtype=np.float64).reshape(self.n, 1)
        H_t = H_t.reshape(self.n, self.m)

        if Q is None:
            Q = getattr(self, 'Q', np.eye(self.m) * 1e-6)
        if R is None:
            R = getattr(self, 'R', np.eye(self.n) * 1e-6)

        # ---- Predict ----
        x_prior = self.x_posterior.copy()
        P_prior = self.P_posterior + Q

        # ---- Generate sigma points ----
        sigma_x, _ = self._sigma_points(x_prior, P_prior)

        # ---- Propagate sigma points through observation model ----
        sigma_y = np.zeros((self.n_sigma, self.n))
        for i in range(self.n_sigma):
            sigma_y[i] = self._obs_function(H_t, sigma_x[i])

        # ---- Reconstruct predicted observation ----
        y_pred = np.sum(self.Wm[:, np.newaxis] * sigma_y, axis=0).reshape(self.n, 1)

        # ---- Innovation covariance S_t ----
        S_t = np.zeros((self.n, self.n))
        for i in range(self.n_sigma):
            diff = (sigma_y[i] - y_pred.flatten()).reshape(self.n, 1)
            S_t += self.Wc[i] * (diff @ diff.T)
        S_t += R

        # ---- Cross-covariance P_xy ----
        P_xy = np.zeros((self.m, self.n))
        for i in range(self.n_sigma):
            dx = (sigma_x[i] - x_prior.flatten()).reshape(self.m, 1)
            dy = (sigma_y[i] - y_pred.flatten()).reshape(self.n, 1)
            P_xy += self.Wc[i] * (dx @ dy.T)

        # ---- UKF Kalman gain ----
        try:
            K_t = P_xy @ inv(S_t)
        except np.linalg.LinAlgError:
            K_t = P_xy @ pinv(S_t)

        # ---- Innovation ----
        innovation = y_t - y_pred

        # ---- Posterior update ----
        x_posterior = x_prior + K_t @ innovation
        # Joseph form for numerical stability
        I_KH = np.eye(self.m) - K_t @ H_t
        P_posterior = I_KH @ P_prior @ I_KH.T + K_t @ R @ K_t.T

        self.x_posterior = x_posterior
        self.P_posterior = P_posterior
        self.gain       = K_t
        self.innovation  = innovation.flatten()
        self.innov_cov   = S_t

        self.x_history.append(x_posterior.copy())
        self.gain_history.append(K_t.copy())
        self.innov_history.append(innovation.flatten().copy())

        return x_posterior.flatten()

    # ------------------------------------------------------------------
    # batch_update
    # ------------------------------------------------------------------
    def batch_update(
        self,
        y: Array,
        H: Array,
        Q: Optional[Array] = None,
        R: Optional[Array] = None,
        x0: Optional[Array] = None,
        P0: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Run the full UKF-based KalmanNet filter over a batch of observations.

        Parameters
        ----------
        y  : Array (T, n)
        H  : Array (T, n, m) or (n, m) if time-invariant
        Q  : Array (m, m) | None
        R  : Array (n, n) | None
        x0 : Array (m,) | None
        P0 : Array (m, m) | None

        Returns
        -------
        x_all     : Array (T, m)
        gain_all  : Array (T, m, n)
        innov_all : Array (T, n)
        S_all     : Array (T, n, n)
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, self.n)
        T = y.shape[0]

        H_arr = np.asarray(H, dtype=np.float64)
        if H_arr.ndim == 2:
            H_arr = np.repeat(H_arr[np.newaxis, :, :], T, axis=0)

        if Q is not None:
            self.Q = Q.astype(np.float64)
        if R is not None:
            self.R = R.astype(np.float64)

        if x0 is None:
            x0 = np.zeros((self.m, 1))
        if P0 is None:
            P0 = np.eye(self.m) * 1e-3

        self.init_sequence(x0, P0)

        x_all    = np.empty((T, self.m))
        gain_all = np.empty((T, self.m, self.n))
        innov_all = np.empty((T, self.n))
        S_all    = np.empty((T, self.n, self.n))

        for t in range(T):
            x_all[t]     = self.update(y[t], H_arr[t], Q, R).flatten()
            gain_all[t]  = self.gain
            innov_all[t]  = self.innovation
            S_all[t]      = self.innov_cov

        return x_all, gain_all, innov_all, S_all


# Helper: pseudo-inverse for singular matrices
def pinv(A: Array) -> Array:
    """Moore–Penrose pseudo-inverse via SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T


# ================================================================================
# 3.  SpreadSignalGenerator — entry / exit signals from spread
# ================================================================================
class SpreadSignalGenerator:
    """
    Generate entry / exit / flat signals from a spread (residual) series.

    The spread e_t = y_t − H_t·β̂_t is the Kalman filter innovation.
    A Bollinger-band approach is used (standard deviation of the spread as
    the volatility envelope).

    Signal convention:
        +1  — long  spread (expect mean-reversion upward)
         0  — flat
        -1  — short spread

    Bollinger band entry/exit:
        Entry  long  : e_t < −√Q_t
        Entry  short : e_t > +√Q_t
        Exit   long  : e_t ≥ 0
        Exit   short : e_t ≤ 0

    Parameters
    ----------
    threshold_mult : float
        Multiplier of std-dev for entry threshold (default 1.0 → √1 = 1σ).
        Use higher values (e.g. 2.0) for fewer, higher-confidence signals.

    Attributes
    ----------
    signals   : Array (T,) — integer signals (+1, 0, -1)
    positions : Array (T,) — floating-point position size (continuous)

    References
    ----------
    KalmanBOT_ICASSP23 — `position_Bollinger` in myUtils.py
    """

    def __init__(self, threshold_mult: float = 1.0):
        self.threshold_mult = threshold_mult
        self.signals   = None
        self.positions = None

    # ------------------------------------------------------------------
    # generate — compute signals from spread and spread-variance series
    # ------------------------------------------------------------------
    def generate(
        self,
        e: Array,       # (T,) — spread / innovation series
        Q: Array,        # (T,) — innovation variance series
    ) -> Tuple[Array, Array]:
        """
        Parameters
        ----------
        e : Array (T,) — spread (innovation) at each step
        Q : Array (T,) — innovation variance (Kalman filter variance Q_t)

        Returns
        -------
        signals   : Array (T,) — discrete signals  {+1, 0, -1}
        positions : Array (T,) — continuous position size (for backtesting)
        """
        e = np.asarray(e, dtype=np.float64).flatten()
        Q = np.asarray(Q, dtype=np.float64).flatten()

        T = len(e)
        signals   = np.zeros(T, dtype=np.int32)
        positions = np.zeros(T, dtype=np.float64)

        # Bollinger threshold
        threshold = self.threshold_mult * np.sqrt(Q)

        # Initialise position state machine
        current_pos = 0.0   # current holding: -1, 0, or +1

        for t in range(T):
            et  = e[t]
            thr = threshold[t]

            # Entry logic
            if current_pos == 0.0:
                if et < -thr:
                    current_pos = 1.0     # long  spread
                elif et > thr:
                    current_pos = -1.0    # short spread
            # Exit logic (mean-reversion realised)
            elif current_pos == 1.0:
                if et >= 0:
                    current_pos = 0.0     # exit long
            elif current_pos == -1.0:
                if et <= 0:
                    current_pos = 0.0     # exit short

            signals[t]   = int(current_pos)
            positions[t]  = current_pos

        self.signals   = signals
        self.positions = positions
        return signals, positions

    # ------------------------------------------------------------------
    # continuous_position — soft (linear) position from z-score
    # ------------------------------------------------------------------
    @staticmethod
    def continuous_position(
        e: Array,
        Q: Array,
        max_position: float = 1.0,
    ) -> Array:
        """
        Continuous (linear) position sizing proportional to z-score.

        position = −e / √Q   clamped to [−max_position, +max_position]

        This provides smoother signals than the binary Bollinger approach.
        """
        e  = np.asarray(e, dtype=np.float64).flatten()
        Q  = np.asarray(Q, dtype=np.float64).flatten()
        std = np.sqrt(Q) + 1e-12
        z   = -e / std                     # negative because long spread = short hedge
        return np.clip(z, -max_position, max_position)


# ================================================================================
# 4.  PairsTradingStrategy — cointegration-based pairs trading with Kalman filter
# ================================================================================
class PairsTradingStrategy:
    """
    Full pairs-trading strategy using a Kalman filter to track a time-varying
    hedge ratio (cointegration coefficient) between two assets.

    Model
    -----
    price_s2_t = β_t[0] · price_s1_t + β_t[1] + ε_t
    β_t       = β_{t-1} + η_t         (random walk)

    State   x_t = β_t  ∈ ℝ²
    Observe y_t = price_s2_t
    H_t     = [price_s1_t, 1]

    Trading rule (Bollinger):
        long  spread  when e_t < −√Q_t   (spread undervalued → expect rebound)
        short spread  when e_t > +√Q_t   (spread overvalued  → expect contraction)
        exit         when e_t crosses 0

    Parameters
    ----------
    q : float
        Process noise standard deviation for hedge ratio (controls how fast β changes).
        Smaller ≈ more stable; larger ≈ more adaptive.
        Typical range: 1e-4 to 1e-2.
    r : float
        Observation noise standard deviation.
        Typical: set from OLS regression residual std-dev × sqrt(1e-4) scale.
    threshold_mult : float
        Bollinger band multiplier (default 1.0).
    hedge : str, default "long"
        Direction of hedge: "long"  = go long the spread (short s1, long s2),
                            "short" = opposite.

    Attributes
    ----------
    kf          : KalmanFilter instance
    signals     : Array (T,) — +1 / 0 / -1
    positions   : Array (T, 2) — dollar position in [s1, s2]
    hedge_ratios : Array (T,) — estimated hedge ratio β_t[0]
    spreads     : Array (T,) — spread series e_t
    spread_vars : Array (T,) — innovation variance Q_t

    Example
    -------
    >>> strategy = PairsTradingStrategy(q=1e-3, r=1e-4)
    >>> signals, positions = strategy.fit_predict(price_s1, price_s2)
    >>> pnl = strategy.backtest(price_s1, price_s2, signals, positions)
    """

    def __init__(
        self,
        q: float  = 1e-3,
        r: float  = 1e-4,
        threshold_mult: float = 1.0,
        hedge: Literal["long", "short"] = "long",
    ):
        self.q   = q
        self.r   = r
        self.threshold_mult = threshold_mult
        self.hedge = hedge

        # State-space matrices
        self.F = np.eye(2)                     # random-walk state transition
        self.Q = (q ** 2) * np.eye(2)         # process noise
        self.R = (r ** 2) * np.eye(1)         # obs noise (scalar → 1×1)

        self.kf: KalmanFilter = KalmanFilter(
            F=self.F, Q=self.Q, R=self.R, m=2, n=1
        )

        self.signal_generator = SpreadSignalGenerator(threshold_mult)

        # Fitted results
        self.beta_all: Optional[Array]   = None   # (T, 2)
        self.spreads:  Optional[Array]   = None
        self.spread_vars: Optional[Array] = None
        self.signals:  Optional[Array]   = None
        self.positions: Optional[Array] = None

    # ------------------------------------------------------------------
    # fit_hedge_ratio — fit Kalman filter over training window
    # ------------------------------------------------------------------
    def fit_hedge_ratio(
        self,
        price_s1: Array,    # (T_train,)
        price_s2: Array,    # (T_train,)
        x0: Optional[Array] = None,
        P0: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Fit Kalman filter on training data to estimate time-varying hedge ratio.

        Parameters
        ----------
        price_s1 : Array (T_train,) — price series of asset S1 (the hedge asset)
        price_s2 : Array (T_train,) — price series of asset S2 (the target asset)
        x0       : Array (2,) | None — initial state [hedge_ratio, intercept]
        P0       : Array (2,2) | None — initial state covariance

        Returns
        -------
        beta_all   : Array (T_train, 2) — [hedge_ratio, intercept] at each step
        P_all      : Array (T_train, 2, 2) — posterior covariances
        e_all      : Array (T_train,) — spread (innovation) series
        Q_all      : Array (T_train,) — innovation variance series
        """
        price_s1 = np.asarray(price_s1, dtype=np.float64).flatten()
        price_s2 = np.asarray(price_s2, dtype=np.float64).flatten()
        T = len(price_s1)

        # Build time-varying H_t: [price_s1[t], 1]
        H_arr = np.stack([price_s1, np.ones(T)], axis=1).reshape(T, 1, 2)   # (T,1,2)
        y_arr = price_s2.reshape(T, 1)                                        # (T,1)

        # Initial state
        if x0 is None:
            # OLS initialization for hedge ratio
            from numpy.linalg import lstsq
            X_aug = np.column_stack([price_s1, np.ones(T)])
            beta_ols, *_ = lstsq(X_aug, price_s2, rcond=None)
            x0 = np.array([[beta_ols[0]], [beta_ols[1]]])

        if P0 is None:
            P0 = np.eye(2) * 1e-3

        self.kf.init_sequence(x0, P0)

        beta_all  = np.empty((T, 2))
        P_all     = np.empty((T, 2, 2))
        e_all     = np.empty(T)
        Q_all     = np.empty(T)

        for t in range(T):
            H_t  = H_arr[t]                     # (1, 2)
            y_t  = y_arr[t]                     # (1,)
            beta = self.kf.update(y_t, H_t)     # (2,)

            beta_all[t]  = beta
            P_all[t]     = self.kf.P_posterior.copy()
            e_all[t]     = self.kf.innovation[0]
            # Innovation variance: H·P_prior·Hᵀ + R  (scalar)
            P_prior = self.kf.P_prior   # prior covariance (saved before overwrite)
            Q_all[t]     = float((H_t @ P_prior @ H_t.T)[0, 0] + self.R[0, 0])

        self.beta_all   = beta_all
        self.spreads    = e_all
        self.spread_vars = Q_all
        return beta_all, P_all, e_all, Q_all

    # ------------------------------------------------------------------
    # compute_spread — compute spread series given prices and beta
    # ------------------------------------------------------------------
    @staticmethod
    def compute_spread(
        price_s1: Array,
        price_s2: Array,
        beta:     Array,
    ) -> Array:
        """
        Compute spread:  e_t = price_s2_t − (β_t[0]·price_s1_t + β_t[1])

        Parameters
        ----------
        price_s1 : Array (T,)
        price_s2 : Array (T,)
        beta     : Array (T, 2) — [hedge_ratio, intercept] per step

        Returns
        -------
        spread : Array (T,)
        """
        price_s1 = np.asarray(price_s1, dtype=np.float64).flatten()
        price_s2 = np.asarray(price_s2, dtype=np.float64).flatten()
        beta     = np.asarray(beta, dtype=np.float64)
        if beta.ndim == 1:
            beta = np.tile(beta, (len(price_s1), 1))
        return price_s2 - (beta[:, 0] * price_s1 + beta[:, 1])

    # ------------------------------------------------------------------
    # generate_signals — run signal generator on fitted spread
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        e:    Optional[Array] = None,
        Q:    Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Generate entry/exit signals from fitted spread and variance series.

        If e and Q are not provided, uses self.spreads and self.spread_vars
        from a prior `fit_hedge_ratio` call.

        Returns
        -------
        signals   : Array (T,) — +1 (long) / 0 (flat) / -1 (short)
        positions : Array (T,) — continuous position (same scale)
        """
        if e is None:
            e = self.spreads
        if Q is None:
            Q = self.spread_vars

        self.signals, self.positions = self.signal_generator.generate(e, Q)

        # Flip if short-hedge mode
        if self.hedge == "short":
            self.signals  = -self.signals
            self.positions = -self.positions

        return self.signals, self.positions

    # ------------------------------------------------------------------
    # fit_predict — fit and generate signals in one call
    # ------------------------------------------------------------------
    def fit_predict(
        self,
        price_s1: Array,
        price_s2: Array,
        x0: Optional[Array] = None,
        P0: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Convenience: run fit_hedge_ratio + generate_signals together.

        Returns (signals, positions).
        """
        self.fit_hedge_ratio(price_s1, price_s2, x0, P0)
        return self.generate_signals()

    # ------------------------------------------------------------------
    # backtest — compute P&L from signals and price series
    # ------------------------------------------------------------------
    def backtest(
        self,
        price_s1:  Array,
        price_s2:  Array,
        signals:   Optional[Array] = None,
        positions: Optional[Array] = None,
        beta:      Optional[Array] = None,
        mode:      Literal["discrete", "continuous"] = "discrete",
    ) -> Tuple[Array, Array]:
        """
        Compute P&L of the pairs trading strategy.

        For each pair of assets (s1, s2), the dollar-neutral position is:
            position_s1 = −sign · hedge_ratio
            position_s2 = +sign

        P&L = Δprice_s1 · position_s1_{t-1} + Δprice_s2 · position_s2_{t-1}

        Parameters
        ----------
        price_s1  : Array (T,) — price series of asset S1
        price_s2  : Array (T,) — price series of asset S2
        signals   : Array (T,) | None — discrete signals; uses self.signals if None
        positions : Array (T,) | None — continuous positions; uses self.positions if None
        beta      : Array (T, 2) | None — hedge ratios; uses self.beta_all if None
        mode      : "discrete" (signals +1/0/-1) or "continuous" (linear position)

        Returns
        -------
        pnl     : Array (T,) — per-step P&L
        cum_pnl : Array (T,) — cumulative P&L
        """
        price_s1 = np.asarray(price_s1, dtype=np.float64).flatten()
        price_s2 = np.asarray(price_s2, dtype=np.float64).flatten()
        T = len(price_s1)

        if signals is None:
            signals = self.signals
        if positions is None:
            positions = self.positions
        if beta is None:
            beta = self.beta_all

        if signals is None or positions is None:
            raise ValueError("Must call fit_predict or generate_signals first, "
                             "or pass signals/positions.")

        if beta is None:
            raise ValueError("beta must be provided or from fit_hedge_ratio.")

        beta = np.asarray(beta)
        if beta.ndim == 1:
            beta = np.tile(beta, (T, 1))

        if mode == "continuous":
            pos_s1 = -positions * beta[:, 0]
            pos_s2 =  positions.copy()
        else:
            # Discrete: use sign of position * hedge ratio
            pos_s1 = -signals * beta[:, 0]
            pos_s2 =  signals.astype(np.float64)

        # Price changes
        dP1 = np.diff(price_s1, prepend=price_s1[0])
        dP2 = np.diff(price_s2, prepend=price_s2[0])

        # P&L: hold position from t-1, profit from change at t
        pnl     = pos_s1[:-1] * dP1[1:] + pos_s2[:-1] * dP2[1:]
        pnl     = np.concatenate([[0.0], pnl])
        cum_pnl = np.cumsum(pnl)

        return pnl, cum_pnl


# ================================================================================
# 5.  MeanReversionStrategy — z-score based mean reversion with KF adaptation
# ================================================================================
class MeanReversionStrategy:
    """
    Z-score based mean-reversion strategy with adaptive volatility estimation
    via a Kalman filter.

    Unlike the pairs-trading strategy which tracks the hedge ratio directly,
    this strategy:
      1. Fits a Kalman filter to the price spread (same as PairsTradingStrategy).
      2. Computes a rolling or KF-adaptive z-score of the spread:
             z_t = (e_t − μ_e) / σ_e
         where μ_e, σ_e are estimated from the KF state and innovation variance.
      3. Generates signals based on |z_t| exceeding a threshold.

    Signal logic:
        |z_t| > entry_threshold → enter in direction of sign(e_t)
        |z_t| < exit_threshold  → exit

    Parameters
    ----------
    q : float
        Process noise for the spread Kalman filter.
    r : float
        Observation noise for the spread Kalman filter.
    entry_threshold : float (default 2.0)
        Z-score magnitude to trigger entry.
    exit_threshold  : float (default 0.5)
        Z-score magnitude to trigger exit.
    rolling_window   : int | None (default None → use KF adaptive)
        If set, uses a rolling window std-dev instead of KF variance.

    Attributes
    ----------
    z_scores  : Array (T,) — z-score series
    signals   : Array (T,) — +1 / 0 / -1
    positions : Array (T,) — floating position size
    pnl, cum_pnl

    References
    ----------
    KalmanBOT_ICASSP23 — mean-reversion variant with Kalman filter
    """

    def __init__(
        self,
        q: float  = 1e-3,
        r: float  = 1e-4,
        entry_threshold: float = 2.0,
        exit_threshold:  float = 0.5,
        rolling_window:  Optional[int] = None,
    ):
        self.q   = q
        self.r   = r
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.rolling_window  = rolling_window

        self.F = np.eye(2)
        self.Q = (q ** 2) * np.eye(2)
        self.R = (r ** 2) * np.eye(1)

        self.kf: KalmanFilter = KalmanFilter(
            F=self.F, Q=self.Q, R=self.R, m=2, n=1
        )

        self.z_scores  = None
        self.signals   = None
        self.positions = None

    # ------------------------------------------------------------------
    # fit — estimate KF parameters and compute z-scores
    # ------------------------------------------------------------------
    def fit(
        self,
        price_s1: Array,
        price_s2: Array,
        x0: Optional[Array] = None,
        P0: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Run Kalman filter on price pair and compute z-score series.

        Returns
        -------
        e_all     : Array (T,) — spread series
        Q_all     : Array (T,) — innovation variance series
        z_scores  : Array (T,) — z-score series
        beta_all  : Array (T, 2) — hedge ratios
        """
        price_s1 = np.asarray(price_s1, dtype=np.float64).flatten()
        price_s2 = np.asarray(price_s2, dtype=np.float64).flatten()
        T = len(price_s1)

        H_arr = np.stack([price_s1, np.ones(T)], axis=1).reshape(T, 1, 2)
        y_arr = price_s2.reshape(T, 1)

        if x0 is None:
            from numpy.linalg import lstsq
            X_aug = np.column_stack([price_s1, np.ones(T)])
            beta_ols, *_ = lstsq(X_aug, price_s2, rcond=None)
            x0 = np.array([[beta_ols[0]], [beta_ols[1]]])

        if P0 is None:
            P0 = np.eye(2) * 1e-3

        self.kf.init_sequence(x0, P0)

        e_all    = np.empty(T)
        Q_all    = np.empty(T)
        beta_all = np.empty((T, 2))

        for t in range(T):
            beta = self.kf.update(y_arr[t], H_arr[t])
            e_all[t]    = self.kf.innovation[0]
            P_prior     = self.kf.P_prior   # prior covariance (saved before overwrite)
            Q_all[t]    = float((H_arr[t] @ P_prior @ H_arr[t].T)[0, 0] + self.R[0, 0])
            beta_all[t] = beta

        self.beta_all = beta_all

        # Z-score computation
        if self.rolling_window is not None:
            # Rolling-window z-score
            window = self.rolling_window
            roll_mean = np.convolve(e_all, np.ones(window) / window, mode='same')
            roll_std  = np.array([e_all[max(0, t-window):t+1].std()
                                   for t in range(T)])
            roll_std  = np.maximum(roll_std, 1e-12)
            z_scores  = (e_all - roll_mean) / roll_std
        else:
            # KF-adaptive z-score: centre = 0 (spread is mean-reverting by construction)
            # scale = sqrt(Q_t) = innovation std-dev
            sigma_e = np.sqrt(Q_all) + 1e-12
            z_scores = e_all / sigma_e

        self.e_all    = e_all
        self.Q_all    = Q_all
        self.z_scores = z_scores
        return e_all, Q_all, z_scores, beta_all

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------
    def generate_signals(
        self,
        z: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Generate signals from z-score series.

        Parameters
        ----------
        z : Array (T,) | None — z-scores; uses self.z_scores if None

        Returns
        -------
        signals   : Array (T,) — +1 / 0 / -1
        positions : Array (T,) — continuous position (z-score scaled)
        """
        if z is None:
            z = self.z_scores

        z = np.asarray(z, dtype=np.float64).flatten()
        T = len(z)

        signals   = np.zeros(T, dtype=np.int32)
        positions = np.zeros(T, dtype=np.float64)
        current_pos = 0.0

        entry_mult = self.entry_threshold
        exit_mult  = self.exit_threshold

        for t in range(T):
            zt = z[t]

            if current_pos == 0.0:
                # Entry
                if zt < -entry_mult:
                    current_pos = -1.0    # short when z-score too negative
                elif zt > entry_mult:
                    current_pos = +1.0    # long  when z-score too positive
            else:
                # Exit on mean reversion
                if current_pos == 1.0 and zt <= exit_mult:
                    current_pos = 0.0
                elif current_pos == -1.0 and zt >= -exit_mult:
                    current_pos = 0.0

            signals[t]   = int(current_pos)
            positions[t] = current_pos

        self.signals   = signals
        self.positions = positions
        return signals, positions

    # ------------------------------------------------------------------
    # fit_predict
    # ------------------------------------------------------------------
    def fit_predict(
        self,
        price_s1: Array,
        price_s2: Array,
    ) -> Tuple[Array, Array]:
        """Convenience: fit + generate signals."""
        self.fit(price_s1, price_s2)
        return self.generate_signals()

    # ------------------------------------------------------------------
    # backtest
    # ------------------------------------------------------------------
    def backtest(
        self,
        price_s1:  Array,
        price_s2:  Array,
        signals:   Optional[Array] = None,
        positions: Optional[Array] = None,
        beta:      Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Backtest the mean-reversion strategy.

        Position construction (same as PairsTradingStrategy):
            pos_s1 = −sign · hedge_ratio
            pos_s2 = +sign
        """
        price_s1 = np.asarray(price_s1, dtype=np.float64).flatten()
        price_s2 = np.asarray(price_s2, dtype=np.float64).flatten()
        T = len(price_s1)

        if signals is None:
            signals = self.signals
        if positions is None:
            positions = self.positions
        if beta is None:
            beta = self.beta_all

        if beta is None:
            raise ValueError("Must fit first or provide beta.")

        beta = np.asarray(beta)
        if beta.ndim == 1:
            beta = np.tile(beta, (T, 1))

        pos_s1 = -signals.astype(np.float64) * beta[:, 0]
        pos_s2 =  signals.astype(np.float64)

        dP1 = np.diff(price_s1, prepend=price_s1[0])
        dP2 = np.diff(price_s2, prepend=price_s2[0])

        pnl     = pos_s1[:-1] * dP1[1:] + pos_s2[:-1] * dP2[1:]
        pnl     = np.concatenate([[0.0], pnl])
        cum_pnl = np.cumsum(pnl)

        return pnl, cum_pnl


# ================================================================================
# 6.  Convenience factory
# ================================================================================
def create_pairs_strategy(
    price_s1: Array,
    price_s2: Array,
    q:        float = 1e-3,
    r:        float = 1e-4,
    threshold_mult: float = 1.0,
    hedge: Literal["long", "short"] = "long",
) -> PairsTradingStrategy:
    """
    Factory: create and fit a PairsTradingStrategy in one call.

    Parameters
    ----------
    price_s1, price_s2 : Array — price series
    q, r                : float — noise parameters
    threshold_mult      : float — Bollinger band width
    hedge               : str   — "long" or "short"

    Returns
    -------
    strategy : PairsTradingStrategy (already fitted)
    """
    strategy = PairsTradingStrategy(q=q, r=r,
                                    threshold_mult=threshold_mult,
                                    hedge=hedge)
    strategy.fit_predict(price_s1, price_s2)
    return strategy
