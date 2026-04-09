"""
HMM Market Regime Detection — 隐马尔可夫模型市场状态识别
=========================================================

A pure-NumPy Gaussian HMM implementation for market regime detection.
No hmmlearn dependency required; uses Baum-Welch for parameter estimation
and scipy.stats.norm for multivariate normal PDF evaluation.

Regimes / 市场状态
------------------
- Bull (Bullish)       : 上涨趋势状态
- Bear (Bearish)       : 下跌趋势状态
- Sideways             : 区间震荡状态
- HighVol              : 高波动状态
- LowVol               : 低波动状态
- Unknown              : 未知状态（未初始化时）

Core Classes / 核心类
----------------------
HMMRegimeDetector      : Gaussian HMM 状态检测器
RegimeSwitchingStrategy: 状态切换交易策略

Example / 示例
--------------
>>> import numpy as np
>>> import pandas as pd
>>> from quant_trading.factors.hmm_regime import HMMRegimeDetector, RegimeSwitchingStrategy, MarketRegime
>>>
>>> # Generate synthetic returns
>>> returns = pd.Series(np.random.randn(1000) * 0.02)
>>>
>>> detector = HMMRegimeDetector(n_states=3, n_iter=200)
>>> detector.fit(returns)
>>>
>>> current = detector.predict_current(returns.iloc[-20:].values)
>>> print(current.regime, current.probability)
>>>
>>> history = detector.get_regime_history(returns)
>>> print(history.head())
>>>
>>> strategy = RegimeSwitchingStrategy(detector, base_strategy=lambda r: np.sign(r))
>>> signals = strategy.generate_signals(returns, price=returns.cumsum() + 1)
"""

from __future__ import annotations

__all__ = [
    "MarketRegime",
    "RegimeState",
    "HMMRegimeDetector",
    "RegimeSwitchingStrategy",
]

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm


# ===========================================================================
# MarketRegime — 市场状态枚举
# ===========================================================================


class MarketRegime(Enum):
    """
    市场状态枚举 / Market regime enumeration.

    Attributes
    ----------
    BULL         : 上涨趋势 / Bullish uptrend
    BEAR         : 下跌趋势 / Bearish downtrend
    SIDEWAYS     : 区间震荡 / Sideways / range-bound
    HIGH_VOL     : 高波动状态 / High volatility regime
    LOW_VOL      : 低波动状态 / Low volatility regime
    UNKNOWN      : 未知状态 / Unknown (not yet fitted)
    """

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    UNKNOWN = "unknown"


# ===========================================================================
# RegimeState — 状态数据类
# ===========================================================================


@dataclass
class RegimeState:
    """
    单个时间点的市场状态快照 / Snapshot of market regime at a single time point.

    Attributes
    ----------
    regime     : MarketRegime — 当前识别的市场状态
    probability: float        — 该状态的概率（0-1）
    timestamp  : datetime     — 状态对应的时间戳
    """

    regime: MarketRegime
    probability: float
    timestamp: datetime


# ===========================================================================
# GaussianHMM — Pure-NumPy HMM Implementation
# ===========================================================================


class GaussianHMM:
    """
    纯NumPy实现的高斯隐马尔可夫模型 / Pure-NumPy Gaussian Hidden Markov Model.

    使用 Baum-Welch 算法估计参数，scipy.stats.norm 用于计算多元正态分布概率密度。

    Parameters
    ----------
    n_states      : int   — 隐藏状态数量（默认3）
    n_iter        : int   — Baum-Welch 最大迭代次数（默认100）
    random_state  : int   — 随机种子（默认42）
    tol           : float — 对数似然收敛阈值（默认1e-4）
    verbose       : bool  — 是否打印训练过程（默认False）
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

        # Model parameters (set during fit)
        self.start_probs_: np.ndarray | None = None  # initial state probabilities (π)
        self.transmat_: np.ndarray | None = None      # state transition matrix (A)
        self.means_: np.ndarray | None = None         # state means (μ)
        self.covars_: np.ndarray | None = None        # state covariances (Σ)
        self.n_features_: int | None = None
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L1 normalize a 1-D array / 归一化向量（使元素和为1）。"""
        return vec / (vec.sum() + 1e-12)

    def _init_params(self, X: np.ndarray) -> None:
        """
        Initialize HMM parameters via temporal segmentation.

        Split the observation sequence into n_states equal temporal segments
        and use each segment's empirical mean/cov as initial parameters.
        This respects the temporal structure of the data.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        seg_len = n_samples // self.n_states
        self.means_ = np.empty((self.n_states, n_features))
        self.covars_ = np.empty((self.n_states, n_features, n_features))
        global_var = float(np.var(X, ddof=1).ravel()[0])

        for k in range(self.n_states):
            start = k * seg_len
            end = n_samples if k == self.n_states - 1 else (k + 1) * seg_len
            seg = X[start:end]
            self.means_[k] = np.mean(seg, axis=0)
            seg_cov = np.cov(seg, rowvar=False, ddof=1)
            if n_features == 1:
                seg_cov = float(seg_cov.ravel()[0]) if seg_cov.ndim == 2 else float(seg_cov)
                var_floor = max(seg_cov * 0.5, global_var * 0.05, 1e-8)
                self.covars_[k] = np.atleast_2d([[max(seg_cov, var_floor)]])
            else:
                seg_cov = (seg_cov + seg_cov.T) / 2.0
                min_eig = float(np.min(np.linalg.eigvalsh(seg_cov)))
                var_floor = max(min_eig * 0.5, global_var * 0.05, 1e-8)
                if min_eig < var_floor:
                    seg_cov += np.eye(n_features) * (var_floor - min_eig)
                self.covars_[k] = seg_cov

        # Small random perturbation to break symmetry
        self.means_ += rng.randn(self.n_states, n_features) * global_var * 0.02

        # Initial state probabilities — uniform
        self.start_probs_ = self._normalize(np.ones(self.n_states))

        # Transition matrix — slightly higher diagonal (persistence)
        transmat = np.ones((self.n_states, self.n_states)) * 0.05
        transmat += np.eye(self.n_states) * 0.85
        self.transmat_ = self._normalize(transmat)

    def _compute_emit_logprob(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(x_t | s_t) for each state using scipy.stats.norm.

        Returns (n_samples × n_states) log emission probabilities.
        """
        n_samples = X.shape[0]
        log_emit = np.full((n_samples, self.n_states), -np.inf)

        for s in range(self.n_states):
            mean = self.means_[s]
            cov = self.covars_[s]

            if self.n_features_ == 1:
                # Univariate case — use norm.pdf directly
                cov_val = float(cov.ravel()[0])
                log_emit[:, s] = norm.logpdf(
                    X.ravel(), loc=float(mean[0]), scale=np.sqrt(cov_val + 1e-12)
                )
            else:
                # Multivariate case — use norm.logpdf with cov
                # scipy.stats.norm.logpdf(x, mean, cov) is available in scipy >= 1.14
                # Fallback to manual Mahalanobis for older scipy
                try:
                    log_emit[:, s] = norm.logpdf(X, mean, cov)
                except Exception:
                    # Manual multivariate normal log PDF
                    diff = X - mean
                    # Use pseudo-inverse for singular covariances
                    try:
                        cov_inv = np.linalg.inv(cov)
                    except np.linalg.LinAlgError:
                        cov_inv = np.linalg.pinv(cov)
                    eigvals = np.linalg.eigvalsh(cov)
                    sign, logdet = np.linalg.slogdet(cov)
                    logdet = logdet if sign > 0 else -np.inf
                    mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
                    log_emit[:, s] = -0.5 * (
                        self.n_features_ * np.log(2 * np.pi)
                        + logdet
                        + mahalanobis
                    )

        return log_emit

    def _forward(self, X: np.ndarray, log_emit: np.ndarray):
        """
        Forward algorithm — compute α_t(s) = P(o_1..o_t, q_t=s).

        Returns (n_samples × n_states) forward log-probabilities.
        """
        n_samples = X.shape[0]
        n_states = self.n_states
        log_start = np.log(self.start_probs_ + 1e-12)

        log_alpha = np.full((n_samples, n_states), -np.inf)
        log_alpha[0] = log_start + log_emit[0]

        for t in range(1, n_samples):
            # α_t(s) = Σ_i α_{t-1}(i) * A[i,s] * B_s(o_t)
            # log: logsumexp_i(log α_{t-1}(i) + log A[i,s])
            # transmat_.T[j, i] = log A[i, j], shape (n_states, n_states)
            # broadcast (n_states,) + (n_states, n_states) → columns get the vector
            log_alpha[t] = np.logaddexp.reduce(
                np.log(self.transmat_.T + 1e-12) + log_alpha[t - 1],
                axis=0,
            )
            log_alpha[t] += log_emit[t]

        return log_alpha

    def _backward(self, X: np.ndarray, log_emit: np.ndarray):
        """
        Backward algorithm — compute β_t(s) = P(o_{t+1}..o_T | q_t=s).

        Returns (n_samples × n_states) backward log-probabilities.
        """
        n_samples = X.shape[0]
        n_states = self.n_states

        log_beta = np.full((n_samples, n_states), -np.inf)
        log_beta[-1] = 0.0  # β_T(s) = 1

        for t in reversed(range(n_samples - 1)):
            # β_t(s) = Σ_j A[s,j] * B_j(o_{t+1}) * β_{t+1}(j)
            # log: logsumexp_j(log A[s,j] + log β_{t+1}(j) + log B_j(o_{t+1}))
            # transmat_[s, :] = log A[s, :], shape (n_states,)
            # (n_states,) + (n_states,) + (n_states,) = (n_states,) → reduce over j
            log_beta[t] = np.logaddexp.reduce(
                np.log(self.transmat_ + 1e-12) + log_beta[t + 1] + log_emit[t + 1],
                axis=1,
            )

        return log_beta

    def _baum_welch(self, X: np.ndarray) -> float:
        """
        Baum-Welch re-estimation (one iteration).

        Returns the change in log-likelihood for convergence checking.
        """
        log_emit = self._compute_emit_logprob(X)
        log_alpha = self._forward(X, log_emit)
        log_beta = self._backward(X, log_emit)

        # Log-likelihood of the observation sequence
        ll_new = np.logaddexp.reduce(log_alpha[-1])

        # Posterior probability of being in state s at time t
        log_gamma = log_alpha + log_beta - ll_new

        # Re-estimate start probabilities
        self.start_probs_ = self._normalize(np.exp(log_gamma[0]))

        # Re-estimate transition matrix — fully vectorized
        # log_xi[t, i, j] = log_alpha[t, i] + log A[i,j] + log_emit[t+1, j] + log_beta[t+1, j]
        # Shapes:
        #   log_alpha[:-1]      : (n_samples-1, n_states)
        #   log_trans           : (n_states, n_states)
        #   log_emit[1:]        : (n_samples-1, n_states)
        #   log_beta[1:]        : (n_samples-1, n_states)
        log_trans = np.log(self.transmat_ + 1e-12)   # (n_states, n_states)

        # Build (n_samples-1, n_states, n_states) tensor via outer products
        log_alpha_ext = log_alpha[:-1, :, np.newaxis]          # (n_samples-1, n_states, 1)
        log_emit_ext = log_emit[1:, np.newaxis, :]            # (n_samples-1, 1, n_states)
        log_beta_ext = log_beta[1:, :, np.newaxis]             # (n_samples-1, n_states, 1)
        log_trans_ext = log_trans[np.newaxis, :, :]            # (1, n_states, n_states)

        log_xi_3d = (
            log_alpha_ext
            + log_trans_ext
            + log_emit_ext
            + log_beta_ext
        )  # → (n_samples-1, n_states, n_states)

        # logsumexp over time → (n_states, n_states)
        log_xi_mat = np.logaddexp.reduce(log_xi_3d, axis=0)   # sum over t

        # Normalize rows: A[i,j] ∝ ξ[i,j],  A[i,:] sums to 1
        log_row_sum = np.logaddexp.reduce(log_xi_mat, axis=1, keepdims=True)  # (n_states, 1)
        self.transmat_ = np.exp(log_xi_mat - log_row_sum) + 1e-12

        # Re-estimate emission parameters (means and covariances)
        gamma = np.exp(log_gamma)
        gamma_sum = gamma.sum(axis=0) + 1e-12

        for s in range(self.n_states):
            self.means_[s] = (gamma[:, s] @ X) / gamma_sum[s]
            diff = X - self.means_[s]
            # Weighted outer product: (n_samples, n_features) weighted by gamma
            if self.n_features_ == 1:
                # Univariate — use scalar arithmetic to avoid shape ambiguity
                weighted_var = np.sum(gamma[:, s] * diff.ravel() ** 2) / gamma_sum[s]
                var_floor = max(float(np.var(X, ddof=1).ravel()[0]) * 0.05, 1e-8)
                cov_s = np.atleast_2d([[max(weighted_var, var_floor)]])
            else:
                cov_s = (gamma[:, s].reshape(-1, 1) * diff).T @ diff / gamma_sum[s]
                # Ensure positive definite: symmetrize + add diagonal regularization
                cov_s = (cov_s + cov_s.T) / 2.0
                # Eigenvalue floor: ensure min eigenvalue >= floor
                eigvals = np.linalg.eigvalsh(cov_s)
                min_eig = float(np.min(eigvals))
                var_floor = max(float(np.var(X, ddof=1).mean()) * 0.05, 1e-8)
                if min_eig < var_floor:
                    cov_s += np.eye(self.n_features_) * (var_floor - min_eig)
            self.covars_[s] = cov_s

        return float(ll_new)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit the Gaussian HMM to the observation sequence.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._init_params(X)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            ll_delta = self._baum_welch(X)
            if iteration > 0 and ll_delta - prev_ll < self.tol:
                if self.verbose:
                    print(f"HMM converged at iteration {iteration + 1} (Δ={ll_delta - prev_ll:.6f})")
                break
            prev_ll = ll_delta
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: log-likelihood = {ll_delta:.4f}")

        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely hidden state sequence using Viterbi.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        states : np.ndarray of shape (n_samples,) — state index for each observation
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        log_emit = self._compute_emit_logprob(X)
        log_delta = np.full((n_samples, self.n_states), -np.inf)
        psi = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialization
        log_delta[0] = np.log(self.start_probs_ + 1e-12) + log_emit[0]

        # Recursion
        for t in range(1, n_samples):
            for s in range(self.n_states):
                log_trans = np.log(self.transmat_[:, s] + 1e-12)
                log_delta[t, s] = np.max(log_delta[t - 1] + log_trans)
                psi[t, s] = np.argmax(log_delta[t - 1] + log_trans)
            log_delta[t] += log_emit[t]

        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in reversed(range(n_samples - 1)):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def score(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the observation sequence.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        log_likelihood : float
        """
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before scoring.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        log_emit = self._compute_emit_logprob(X)
        log_alpha = self._forward(X, log_emit)
        return float(np.logaddexp.reduce(log_alpha[-1]))


# ===========================================================================
# HMMRegimeDetector — 市场状态检测器
# ===========================================================================


class HMMRegimeDetector:
    """
    Hidden Markov Model Market Regime Detector / 隐马尔可夫模型市场状态检测器.

    Uses a pure-NumPy Gaussian HMM to identify market regimes from return sequences.
    Supports 3-state (Bull / Bear / Sideways) or 5-state (adds HighVol / LowVol) models.

    Parameters
    ----------
    n_states   : int   — Number of hidden states (3 or 5, default 3)
    n_iter     : int   — Maximum Baum-Welch iterations (default 100)
    random_state: int  — Random seed (default 42)

    Attributes
    ----------
    model_        : GaussianHMM — The underlying HMM
    regime_map_   : dict        — Maps state index → MarketRegime (set after fit)
    """

    # Default position sizing per regime (fraction of max position)
    _DEFAULT_POSITION_SCALES: dict[MarketRegime, float] = {
        MarketRegime.BULL: 1.0,
        MarketRegime.BEAR: -0.5,   # reduced / short bias
        MarketRegime.SIDEWAYS: 0.3,
        MarketRegime.HIGH_VOL: 0.4,   # reduced in high vol
        MarketRegime.LOW_VOL: 0.8,
        MarketRegime.UNKNOWN: 0.0,
    }

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        if n_states < 2:
            raise ValueError("n_states must be >= 2.")
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_: GaussianHMM | None = None
        self.regime_map_: dict[int, MarketRegime] = {}
        self._regime_order_: list[MarketRegime] = []
        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Regime mapping helpers
    # ------------------------------------------------------------------

    def _build_regime_map(self, means: np.ndarray) -> None:
        """
        Assign MarketRegime labels to HMM states based on their mean returns.

        States are sorted by mean return (ascending).
        - Lowest mean  → BEAR
        - Highest mean → BULL
        - All other states → distributed across SIDEWAYS, LOW_VOL, HIGH_VOL

        Mapping for common n_states:
          2-state : [BEAR, BULL]
          3-state : [BEAR, SIDEWAYS, BULL]
          4-state : [BEAR, LOW_VOL, HIGH_VOL, BULL]
          5-state : [BEAR, LOW_VOL, SIDEWAYS, HIGH_VOL, BULL]
          N>5     : [BEAR, LOW_VOL, SIDEWAYS, HIGH_VOL, BULL] + SIDEWAYS for rest
        """
        sorted_idx = np.argsort(means.ravel())
        self._regime_order_ = []

        if self.n_states == 2:
            labels = [MarketRegime.BEAR, MarketRegime.BULL]
        elif self.n_states == 3:
            labels = [MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.BULL]
        elif self.n_states == 4:
            labels = [MarketRegime.BEAR, MarketRegime.LOW_VOL, MarketRegime.HIGH_VOL, MarketRegime.BULL]
        elif self.n_states >= 5:
            labels = [
                MarketRegime.BEAR,
                MarketRegime.LOW_VOL,
                MarketRegime.SIDEWAYS,
                MarketRegime.HIGH_VOL,
                MarketRegime.BULL,
            ]
            # Fill remaining slots with SIDEWAYS
            while len(labels) < self.n_states:
                labels.insert(len(labels) // 2, MarketRegime.SIDEWAYS)
        else:
            labels = [MarketRegime.SIDEWAYS] * self.n_states

        for rank, idx in enumerate(sorted_idx):
            label = labels[rank] if rank < len(labels) else MarketRegime.SIDEWAYS
            self.regime_map_[idx] = label
            self._regime_order_.append(label)

    # ------------------------------------------------------------------
    # Core fit / predict
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray | pd.Series) -> "HMMRegimeDetector":
        """
        Fit the HMM to a return series.

        Parameters
        ----------
        returns : np.ndarray or pd.Series of shape (n_samples,)
            — Financial returns (e.g., daily pct changes)

        Returns
        -------
        self
        """
        data = np.asarray(returns, dtype=np.float64).ravel()
        if data.ndim != 1:
            raise ValueError("returns must be a 1-D sequence.")

        self.model_ = GaussianHMM(
            n_states=self.n_states,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model_.fit(data.reshape(-1, 1))
        self._build_regime_map(self.model_.means_)
        self.fitted_ = True
        return self

    def predict(self, returns: np.ndarray | pd.Series) -> list[RegimeState]:
        """
        Predict the market regime at each time point.

        Parameters
        ----------
        returns : np.ndarray or pd.Series

        Returns
        -------
        list[RegimeState] — regime label, probability, and timestamp for each point
        """
        if not self.fitted_:
            raise RuntimeError("Detector must be fitted before predict().")

        data = np.asarray(returns, dtype=np.float64).ravel()
        if isinstance(returns, pd.Series):
            timestamps = returns.index
        else:
            timestamps = range(len(data))

        state_probs = self._get_state_posteriors(data)
        state_indices = self.model_.predict(data.reshape(-1, 1))

        states = []
        for i, (idx, ts) in enumerate(zip(state_indices, timestamps)):
            regime = self.regime_map_.get(idx, MarketRegime.UNKNOWN)
            prob = float(state_probs[i, idx])
            if isinstance(ts, (int, np.integer)):
                ts = datetime.utcfromtimestamp(ts)
            elif isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            elif not isinstance(ts, datetime):
                ts = datetime.utcfromtimestamp(0)
            states.append(RegimeState(regime=regime, probability=prob, timestamp=ts))

        return states

    def predict_current(self, recent_returns: np.ndarray) -> RegimeState:
        """
        Predict the current (most recent) market regime.

        Parameters
        ----------
        recent_returns : np.ndarray of shape (window_size,)
            — Recent return observations

        Returns
        -------
        RegimeState — current regime, its probability, and timestamp
        """
        if not self.fitted_:
            raise RuntimeError("Detector must be fitted before predict_current().")

        data = np.asarray(recent_returns, dtype=np.float64).ravel()
        state_probs = self._get_state_posteriors(data)
        current_probs = state_probs[-1]
        current_idx = int(np.argmax(current_probs))
        current_regime = self.regime_map_.get(current_idx, MarketRegime.UNKNOWN)

        return RegimeState(
            regime=current_regime,
            probability=float(current_probs[current_idx]),
            timestamp=datetime.now(),
        )

    def _get_state_posteriors(self, data: np.ndarray) -> np.ndarray:
        """Compute P(state | observations) using forward-backward."""
        log_emit = self.model_._compute_emit_logprob(data.reshape(-1, 1))
        log_alpha = self.model_._forward(data.reshape(-1, 1), log_emit)
        log_beta = self.model_._backward(data.reshape(-1, 1), log_emit)
        ll = float(np.logaddexp.reduce(log_alpha[-1]))
        log_gamma = log_alpha + log_beta - ll
        return np.exp(log_gamma)

    def get_regime_history(self, returns: pd.Series) -> pd.DataFrame:
        """
        Return a DataFrame with regime labels and probabilities over time.

        Parameters
        ----------
        returns : pd.Series — return series with datetime index

        Returns
        -------
        pd.DataFrame with columns: timestamp, regime, probability
        """
        states = self.predict(returns)
        records = [
            {"timestamp": s.timestamp, "regime": s.regime.value, "probability": s.probability}
            for s in states
        ]
        df = pd.DataFrame(records)
        if df is not None and len(df) > 0:
            df.set_index("timestamp", inplace=True)
        return df

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return the estimated state transition matrix A.

        Returns
        -------
        np.ndarray of shape (n_states, n_states)
            A[i, j] = P(state_j | state_i)
        """
        if not self.fitted_:
            raise RuntimeError("Detector must be fitted first.")
        return self.model_.transmat_.copy()

    def get_expected_duration(self, regime: MarketRegime) -> float:
        """
        Estimate the expected duration (in time steps) for a given regime.

        Uses the diagonal of the transition matrix:
        E[duration] = 1 / (1 - A[i, i])  for state i.

        Parameters
        ----------
        regime : MarketRegime

        Returns
        -------
        float — expected number of steps in this regime
        """
        if not self.fitted_:
            raise RuntimeError("Detector must be fitted first.")
        # Find which state index corresponds to this regime
        idx = None
        for k, v in self.regime_map_.items():
            if v == regime:
                idx = k
                break
        if idx is None:
            return 0.0
        stay_prob = self.model_.transmat_[idx, idx]
        if stay_prob >= 1.0 - 1e-12:
            return float("inf")
        return float(1.0 / (1.0 - stay_prob))

    def get_regime_statistics(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute per-regime statistics (mean return, volatility, count).

        Parameters
        ----------
        returns : pd.Series

        Returns
        -------
        pd.DataFrame with index = regime name, columns = [count, mean, std, min, max]
        """
        states = self.predict(returns)
        regime_map = {i: s.regime.value for i, s in enumerate(states)}
        df = pd.DataFrame({"regime": [regime_map[i] for i in range(len(returns))], "return": returns.values})
        stats = df.groupby("regime")["return"].agg(["count", "mean", "std", "min", "max"])
        return stats

    @property
    def means_(self) -> np.ndarray | None:
        """HMM state means (μ) after fitting."""
        return self.model_.means_ if self.model_ else None

    @property
    def covars_(self) -> np.ndarray | None:
        """HMM state covariance matrices (Σ) after fitting."""
        return self.model_.covars_ if self.model_ else None


# ===========================================================================
# RegimeSwitchingStrategy — 状态切换策略
# ===========================================================================


class RegimeSwitchingStrategy:
    """
    Regime-Switching Trading Strategy / 状态切换交易策略.

    Adjusts position size and strategy selection based on detected market regime.

    Regime Rules / 状态规则
    -----------------------
    - Bull      : Trend-following — full long position
    - Bear      : Mean-reversion or flat — reduced / short bias
    - Sideways  : Range-bound oscillation strategy
    - HighVol   : Reduce position size, widen stop-loss
    - LowVol    : Normal / slightly increased position

    Parameters
    ----------
    detector      : HMMRegimeDetector — fitted regime detector
    base_strategy : callable          — function(returns) → base signal (-1/0/1)
    """

    def __init__(
        self,
        detector: HMMRegimeDetector,
        base_strategy: Callable[[pd.Series], pd.Series] | None = None,
    ):
        self.detector = detector
        self.base_strategy = base_strategy or (lambda r: np.sign(r))

        # Position scales per regime
        self.position_scales: dict[MarketRegime, float] = (
            detector._DEFAULT_POSITION_SCALES.copy()
        )

        # Regime-specific stop-loss multipliers (relative to base)
        self.stop_loss_multipliers: dict[MarketRegime, float] = {
            MarketRegime.HIGH_VOL: 2.0,
            MarketRegime.LOW_VOL: 0.5,
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 1.5,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.UNKNOWN: 1.0,
        }

    def get_position_size(self, regime: MarketRegime) -> float:
        """
        Return the position scale factor for the given regime.

        Parameters
        ----------
        regime : MarketRegime

        Returns
        -------
        float — position scale in (-1, 1); negative = short
        """
        return self.position_scales.get(regime, 0.0)

    def generate_signals(
        self,
        returns: pd.Series,
        price: pd.Series | None = None,
    ) -> pd.Series:
        """
        Generate regime-adjusted trading signals.

        Parameters
        ----------
        returns : pd.Series — return series (datetime index recommended)
        price    : pd.Series — optional price series for additional context

        Returns
        -------
        pd.Series — adjusted signals in {-1, 0, 1} with same index as returns
        """
        if not self.detector.fitted_:
            raise RuntimeError("Detector must be fitted first.")

        regime_states = self.detector.predict(returns)
        regime_seq = pd.Series(
            [s.regime for s in regime_states],
            index=returns.index if hasattr(returns, "index") else None,
        )

        # Base signals
        base_signals = self.base_strategy(returns)
        if not isinstance(base_signals, pd.Series):
            base_signals = pd.Series(base_signals, index=returns.index)

        # Regime-adjusted signals
        adjusted = pd.Series(0.0, index=returns.index)
        for regime in MarketRegime:
            mask = regime_seq == regime
            if not mask.any():
                continue
            scale = self.get_position_size(regime)
            # For bear regime: bias toward short or flat
            if regime == MarketRegime.BEAR:
                # Multiply base signal by scale (scale is negative => short bias)
                adjusted[mask] = np.clip(base_signals[mask] * scale, -1.0, 0.5)
            else:
                adjusted[mask] = np.clip(base_signals[mask] * scale, -1.0, 1.0)

        return adjusted

    def get_stop_loss_multiplier(self, regime: MarketRegime) -> float:
        """Return the stop-loss width multiplier for the given regime."""
        return self.stop_loss_multipliers.get(regime, 1.0)

    def describe_regimes(self) -> pd.DataFrame:
        """
        Return a table summarizing each regime's properties.

        Returns
        -------
        pd.DataFrame with columns: regime, position_scale, stop_loss_mult, expected_duration
        """
        rows = []
        for regime in [r for r in MarketRegime if r != MarketRegime.UNKNOWN]:
            rows.append({
                "regime": regime.value,
                "position_scale": self.get_position_size(regime),
                "stop_loss_mult": self.get_stop_loss_multiplier(regime),
                "expected_duration": (
                    self.detector.get_expected_duration(regime)
                    if self.detector.fitted_
                    else None
                ),
            })
        return pd.DataFrame(rows).set_index("regime")
