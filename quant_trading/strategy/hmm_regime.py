"""
Hidden Markov Model for Market Regime Detection
=================================================

A pure NumPy implementation of Gaussian HMM with Baum-Welch training
for detecting bull / bear / neutral market regimes.

Absorbed from:
  - D:/Hive/Data/trading_repos/RegimeSwitchingMomentumStrategy/   (regime_detection.py, hmm_model.py)
  - D:/Hive/Data/trading_repos/AI-Powered-Energy-Algorithmic-Trading-Integrating-Hidden-Markov-Models-with-Neural-Networks/ (alpha.py)

No hmmlearn dependency — all algorithms implemented from scratch.

Classes
-------
GaussianHMM
    Gaussian-emitting Hidden Markov Model with Baum-Welch EM training.
MarketRegimeDetector
    Detects bull / bear / neutral regimes using HMM on log-returns.
RegimeAwareStrategy
    Base strategy that adapts position sizing and signal generation by regime.

数学公式 / Mathematical Formulas
--------------------------------
Forward Algorithm (前向算法):
    α_t(i) = P(o_1, ..., o_t, q_t = s_i | λ)
    α_1(i) = π_i * N(o_1 | μ_i, Σ_i)
    α_t(i) = [Σ_j α_{t-1}(j) * a_{ji}] * N(o_t | μ_i, Σ_i)

Backward Algorithm (后向算法):
    β_T(i) = 1
    β_t(i) = Σ_j a_{ij} * N(o_{t+1} | μ_j, Σ_j) * β_{t+1}(j)

Viterbi Decoding (Viterbi 解码):
    δ_t(i) = max_{q_1,...,q_{t-1}} P(q_1,...,q_{t-1}, q_t = s_i, o_1,...,o_t | λ)
    ψ_t(i) = argmax_j [δ_{t-1}(j) * a_{ji}]
    q_T* = argmax_i δ_T(i)
    q_t* = ψ_{t+1}(q_{t+1}*)

Baum-Welch EM ( Baum-Welch 期望最大化):
    γ_t(i) = α_t(i) * β_t(i) / Σ_j α_t(j) * β_t(j)        (state posterior)
    ξ_t(i,j) = α_t(i) * a_{ij} * N(o_{t+1}|μ_j,Σ_j) * β_{t+1}(j) / P(O|λ)

    π_i  <- γ_1(i)
    a_{ij} <- Σ_t ξ_t(i,j) / Σ_t γ_t(i)
    μ_i  <- Σ_t γ_t(i) * o_t / Σ_t γ_t(i)
    Σ_i  <- Σ_t γ_t(i) * (o_t-μ_i)(o_t-μ_i)^T / Σ_t γ_t(i)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# GaussianHMM                                                                 #
# --------------------------------------------------------------------------- #


class GaussianHMM:
    """
    Gaussian-emitting Hidden Markov Model — pure NumPy implementation.

    Parameters
    ----------
    n_states : int, default 3
        Number of hidden states (K).
    n_features : int, default 1
        Dimensionality d of each observation o_t ∈ R^d.
    max_iter : int, default 200
        Maximum Baum-Welch iterations.
    tol : float, default 1e-4
        Convergence tolerance on log-likelihood improvement.
    verbose : bool, default False
        Print per-iteration log-likelihood.

    Attributes
    ----------
    pi : np.ndarray (n_states,)
        Initial state probabilities π_i = P(q_1 = s_i | λ).
    A : np.ndarray (n_states, n_states)
        State transition matrix a_{ij} = P(q_{t+1}=s_j | q_t=s_i, λ).
    mu : np.ndarray (n_states, n_features)
        Gaussian means μ_i for each state.
    sigma : np.ndarray (n_states, n_features, n_features)
        Gaussian covariances Σ_i for each state.
    log_likelihood_ : float
        Final log-likelihood after fitting.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> np.random.seed(42)
    >>> model = GaussianHMM(n_states=2, n_features=1, max_iter=100)
    >>> X = np.concatenate([np.random.randn(100, 1) * 0.1 + 0.0,
    ...                     np.random.randn(100, 1) * 0.3 + 2.0])
    >>> model.fit(X)
    >>> states = model.predict(X)
    >>> samples = model.sample(n_samples=10)
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 1,
        max_iter: int = 200,
        tol: float = 1e-4,
        verbose: False = False,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")

        self.n_states = n_states
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Parameters (initialised in fit or _init_params)
        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

        self.log_likelihood_: float = -np.inf
        self.n_iter_: int = 0

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _init_params(self, X: np.ndarray) -> None:
        """Initialise pi, A, mu, sigma using k-means on X."""
        n, d = X.shape
        assert d == self.n_features

        # Random pi (normalize)
        self.pi = np.random.rand(self.n_states)
        self.pi /= self.pi.sum()

        # Random A (row-stochastic)
        self.A = np.random.rand(self.n_states, self.n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # k-means++ init for mu and sigma
        mu = np.zeros((self.n_states, d))
        # Choose first centre uniformly at random
        idx = np.random.randint(n)
        mu[0] = X[idx]

        for k in range(1, self.n_states):
            # Squared distances from chosen centres
            dists = np.sum((X[:, None, :] - mu[:k][None, :, :]) ** 2, axis=2)
            min_dists = np.min(dists, axis=1)
            # Probability proportional to squared distance
            probs = min_dists / (min_dists.sum() + 1e-12)
            idx = np.random.choice(n, p=probs)
            mu[k] = X[idx]

        #sigma = spherical covariance init
        global_var = X.var()
        sigma = np.zeros((self.n_states, d, d))
        for k in range(self.n_states):
            sigma[k] = np.eye(d) * (global_var + 1e-6)

        self.mu = mu
        self.sigma = sigma

    @staticmethod
    def _gaussian_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Compute N(x | mu, sigma) for each row of x.

        Parameters
        ----------
        x : np.ndarray (T, d)
        mu : np.ndarray (d,)
        sigma : np.ndarray (d, d)

        Returns
        -------
        np.ndarray (T,) — pdf value for each row.
        """
        d = x.shape[1]
        diff = x - mu[None, :]                       # (T, d)
        # Use Cholesky for numerical stability
        try:
            L = np.linalg.cholesky(sigma)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            # Solve L @ diff_L = diff.T  =>  diff_L = L^{-1} @ diff.T
            diff_L = np.linalg.solve(L, diff.T)  # (d, T)
            mahala = np.sum(diff_L**2, axis=0)                          # (T,)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            sigma_inv = np.linalg.pinv(sigma)
            log_det = np.linalg.slogdet(sigma)[1]
            mahala = np.sum(diff @ sigma_inv * diff, axis=1)             # (T,)

        norm_const = -0.5 * d * np.log(2 * np.pi) - 0.5 * log_det
        return np.exp(norm_const - 0.5 * mahala)                         # (T,)

    def _compute_alpha(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm.

        Returns
        -------
        alpha : np.ndarray (T, K) — α_t(i)
        log_ll : float — log P(X | λ)
        """
        T, d = X.shape
        K = self.n_states
        alpha = np.zeros((T, K))

        # Emission probabilities B_t(i) = N(x_t | μ_i, Σ_i)
        B = np.zeros((T, K))
        for k in range(K):
            B[:, k] = self._gaussian_pdf(X, self.mu[k], self.sigma[k])

        # Initialise
        alpha[0] = self.pi * B[0]
        sum_alpha0 = alpha[0].sum()
        if sum_alpha0 == 0:
            alpha[0] = 1e-12
            B[0] = 1e-12
            sum_alpha0 = 1e-12
        alpha[0] /= sum_alpha0

        # Scale to avoid underflow (scaled α)
        c = np.zeros(T)
        c[0] = alpha[0].sum()

        # Forward pass with scaling
        for t in range(1, T):
            alpha[t] = B[t] * (alpha[t - 1] @ self.A)   # (K,) = (K,) * (K,)
            sum_alpha = alpha[t].sum()
            if sum_alpha == 0:
                alpha[t] = 1e-300
                sum_alpha = 1e-300
            alpha[t] /= sum_alpha
            c[t] = sum_alpha

        log_ll = np.sum(np.log(c + 1e-300))
        return alpha, log_ll

    def _compute_beta(self, X: np.ndarray) -> np.ndarray:
        """Backward algorithm (scaled).

        Returns
        -------
        beta : np.ndarray (T, K) — β_t(i)
        """
        T, d = X.shape
        K = self.n_states
        beta = np.zeros((T, K))

        # Emission probabilities
        B = np.zeros((T, K))
        for k in range(K):
            B[:, k] = self._gaussian_pdf(X, self.mu[k], self.sigma[k])

        beta[-1] = 1.0
        for t in reversed(range(T - 1)):
            beta[t] = (beta[t + 1] * B[t + 1]) @ self.A.T  # (K,)
            beta[t] /= beta[t].sum() + 1e-300

        return beta

    def _baum_welch_step(self, X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One Baum-Welch EM iteration.

        Returns
        -------
        log_ll : float
        pi_new, A_new, mu_new, sigma_new : updated parameters
        """
        T, d = X.shape
        K = self.n_states

        alpha, log_ll = self._compute_alpha(X)
        beta = self._compute_beta(X)

        # State posterior γ_t(i) = P(q_t = s_i | X, λ)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # Xi_t(i,j) = P(q_t=s_i, q_{t+1}=s_j | X, λ)
        xi = np.zeros((T - 1, K, K))
        B = np.zeros((T, K))
        for k in range(K):
            B[:, k] = self._gaussian_pdf(X, self.mu[k], self.sigma[k])

        for t in range(T - 1):
            denom = (alpha[t][:, None] * self.A * (B[t + 1][None, :] * beta[t + 1][None, :])).sum()
            if denom < 1e-300:
                xi[t] = 1e-300
            else:
                xi[t] = alpha[t][:, None] * self.A * (B[t + 1][None, :] * beta[t + 1][None, :]) / denom

        # Update parameters
        pi_new = gamma[0]
        pi_new /= pi_new.sum() + 1e-300

        A_new = xi.sum(axis=0)                      # (K, K)
        A_new /= A_new.sum(axis=1, keepdims=True) + 1e-300

        mu_new = (gamma[:, :, None] * X[:, None, :]).sum(axis=0)   # (K, d)
        mu_new /= gamma.sum(axis=0)[:, None] + 1e-300

        sigma_new = np.zeros((K, d, d))
        for k in range(K):
            diff = X - mu_new[k]                     # (T, d)
            sigma_new[k] = (gamma[:, k:k+1] * diff).T @ diff / (gamma[:, k].sum() + 1e-300)
            # Ensure positive definite
            sigma_new[k] = 0.5 * (sigma_new[k] + sigma_new[k].T)
            sigma_new[k] += np.eye(d) * 1e-6

        return log_ll, pi_new, A_new, mu_new, sigma_new

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit the HMM to observations X using Baum-Welch EM.

        Parameters
        ----------
        X : np.ndarray (T, d)
            Observation sequence. T = number of time steps, d = n_features.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._init_params(X)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            log_ll, pi_new, A_new, mu_new, sigma_new = self._baum_welch_step(X)

            self.pi = pi_new
            self.A = A_new
            self.mu = mu_new
            self.sigma = sigma_new

            if self.verbose:
                print(f"  iteration {iteration + 1}: log-likelihood = {log_ll:.4f}")

            if log_ll - prev_ll < self.tol:
                break
            prev_ll = log_ll

        self.log_likelihood_ = prev_ll
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Find the most likely state sequence via Viterbi decoding.

        Parameters
        ----------
        X : np.ndarray (T, d)
            Observation sequence.

        Returns
        -------
        states : np.ndarray (T,)
            Most likely state index for each time step.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape
        K = self.n_states

        B = np.zeros((T, K))
        for k in range(K):
            B[:, k] = self._gaussian_pdf(X, self.mu[k], self.sigma[k])

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)

        delta[0] = np.log(self.pi + 1e-300) + np.log(B[0] + 1e-300)
        for t in range(1, T):
            for j in range(K):
                trans_probs = delta[t - 1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = np.max(trans_probs) + np.log(B[t, j] + 1e-300)

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities P(q_t | X, λ) for each t.

        Parameters
        ----------
        X : np.ndarray (T, d)
            Observation sequence.

        Returns
        -------
        gamma : np.ndarray (T, K)
            Posterior probability of each state at each time step.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        alpha, _ = self._compute_alpha(X)
        beta = self._compute_beta(X)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma

    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sample observation sequence and hidden state sequence.

        Parameters
        ----------
        n_samples : int
            Number of time steps T to generate.
        random_state : int, optional
            Random seed.

        Returns
        -------
        X_sample : np.ndarray (T, d)
            Generated observations.
        states : np.ndarray (T,)
            Generated hidden state indices.
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random

        X_sample = np.zeros((n_samples, self.n_features))
        states = np.zeros(n_samples, dtype=int)

        # Sample initial state
        states[0] = rng.choice(self.n_states, p=self.pi)
        X_sample[0] = rng.multivariate_normal(self.mu[states[0]], self.sigma[states[0]])

        for t in range(1, n_samples):
            states[t] = rng.choice(self.n_states, p=self.A[states[t - 1]])
            X_sample[t] = rng.multivariate_normal(self.mu[states[t]], self.sigma[states[t]])

        return X_sample, states

    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood log P(X | λ).

        Parameters
        ----------
        X : np.ndarray (T, d)
            Observation sequence.

        Returns
        -------
        log_ll : float
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, log_ll = self._compute_alpha(X)
        return log_ll


# --------------------------------------------------------------------------- #
# MarketRegimeDetector                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class MarketRegimeDetectorConfig:
    """Configuration for MarketRegimeDetector."""
    n_states: int = 3
    """Number of hidden regimes (default 3: bull, neutral, bear)."""
    lookback: int = 60
    """Rolling window size for feature computation."""
    return_window: int = 1
    """Window for computing log returns."""
    volatility_window: int = 20
    """Window for computing rolling volatility."""
    max_iter: int = 200
    """Max Baum-Welch iterations per fit."""
    tol: float = 1e-4
    """Convergence tolerance."""


class MarketRegimeDetector:
    """
    Market regime detector using Gaussian HMM on log-returns and volatility.

    This class wraps GaussianHMM to provide a market-regime-specific interface.
    It computes features (log-returns, rolling volatility) from OHLCV price data,
    fits a Gaussian HMM, and maps hidden states to economic regime labels.

    Regime Labels (mapped by mean return of each state):
        BULL   — high positive mean return
        NEUTRAL — near-zero mean return
        BEAR   — high negative mean return

    Parameters
    ----------
    config : MarketRegimeDetectorConfig, optional
        Detector configuration.
    verbose : bool, default False
        Print Baum-Welch progress.

    Examples
    --------
    >>> import numpy as np
    >>> detector = MarketRegimeDetector()
    >>> # prices: array of close prices
    >>> features = detector.prepare_features(prices)
    >>> detector.fit(features)
    >>> regimes = detector.predict(features)
    >>> proba = detector.predict_proba(features)
    >>> regime_labels = detector.map_states_to_regimes(regimes)
    """

    BULL: str = "BULL"
    NEUTRAL: str = "NEUTRAL"
    BEAR: str = "BEAR"

    def __init__(
        self,
        config: Optional[MarketRegimeDetectorConfig] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config or MarketRegimeDetectorConfig()
        self.verbose = verbose
        self._hmm = GaussianHMM(
            n_states=self.config.n_states,
            n_features=2,          # [return, volatility]
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            verbose=self.verbose,
        )
        self._state_means: Optional[np.ndarray] = None
        self._state_to_regime: Optional[dict] = None

    @staticmethod
    def compute_log_returns(prices: np.ndarray, window: int = 1) -> np.ndarray:
        """
        Compute multi-period log returns.

        Parameters
        ----------
        prices : np.ndarray (T,)
            Close prices.
        window : int, default 1
            Lookback period for return calculation.

        Returns
        -------
        log_returns : np.ndarray (T - window,)
        """
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim == 2:
            prices = prices.flatten()
        return np.log(prices[window:] / prices[:-window])

    @staticmethod
    def compute_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Compute rolling standard deviation of returns (realised volatility).

        Parameters
        ----------
        returns : np.ndarray (T,)
            Return series.
        window : int, default 20
            Rolling window size.

        Returns
        -------
        volatility : np.ndarray (T - window + 1,)
            Rolling volatility aligned to end of each window.
        """
        returns = np.asarray(returns, dtype=np.float64)
        T = len(returns)
        if window > T:
            return np.array([returns.std()])
        vol = np.array([returns[i - window + 1:i + 1].std() for i in range(window - 1, T)])
        return vol

    def prepare_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Prepare feature matrix [log_return, volatility] for each time step.

        Parameters
        ----------
        prices : np.ndarray (T,) or (T, 1)
            Close price series.

        Returns
        -------
        features : np.ndarray (N, 2)
            Feature matrix where N = T - max(lookback, volatility_window).
            Returns empty array if insufficient data.
        """
        prices = np.asarray(prices, dtype=np.float64).flatten()

        ret_window = max(self.config.return_window, 1)
        vol_window = self.config.volatility_window

        log_ret = self.compute_log_returns(prices, window=ret_window)  # (T_ret,)

        # Pad returns to compute volatility with rolling window
        # vol[i] corresponds to end of window ending at i
        vol_full = self.compute_volatility(log_ret, window=vol_window)  # (T_ret - vol_window + 1,)

        # Align: features start where both are available
        n_feat = len(vol_full)
        if n_feat <= 0:
            # Return empty array instead of raising for numerical stability
            return np.array([]).reshape(0, 2)

        log_ret_aligned = log_ret[-n_feat:]
        features = np.column_stack([log_ret_aligned, vol_full])
        return features

    def fit(self, prices: np.ndarray) -> "MarketRegimeDetector":
        """
        Fit the HMM on price data.

        Parameters
        ----------
        prices : np.ndarray (T,)
            Close price series.

        Returns
        -------
        self
        """
        features = self.prepare_features(prices)
        self._hmm.fit(features)
        self._derive_regime_mapping(features)
        return self

    def _derive_regime_mapping(self, features: np.ndarray) -> None:
        """Map each HMM state to BULL / NEUTRAL / BEAR based on mean return."""
        states = self._hmm.predict(features)           # (N,)
        self._state_means = np.array([
            features[states == k, 0].mean() for k in range(self.config.n_states)
        ])
        # Sort states by mean return descending
        order = np.argsort(self._state_means)[::-1]   # best return first
        regime_labels = [self.BULL, self.NEUTRAL, self.BEAR]
        self._state_to_regime = {
            order[i]: regime_labels[i] for i in range(min(self.config.n_states, len(regime_labels)))
        }

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime sequence.

        Parameters
        ----------
        prices : np.ndarray (T,)

        Returns
        -------
        regimes : np.ndarray (N,)
            Regime label for each time step (BULL=0, NEUTRAL=1, BEAR=2).
        """
        features = self.prepare_features(prices)
        states = self._hmm.predict(features)
        return np.array([self._state_to_regime[s] for s in states])

    def predict_proba(self, prices: np.ndarray) -> np.ndarray:
        """
        Return posterior regime probabilities.

        Parameters
        ----------
        prices : np.ndarray (T,)

        Returns
        -------
        proba : np.ndarray (N, 3)
            Posterior probability of [BULL, NEUTRAL, BEAR] for each step.
        """
        features = self.prepare_features(prices)
        gamma = self._hmm.predict_proba(features)      # (N, K)
        # Remap columns from state order to [BULL, NEUTRAL, BEAR] order
        bull_idx = [s for s, r in self._state_to_regime.items() if r == self.BULL]
        neutral_idx = [s for s, r in self._state_to_regime.items() if r == self.NEUTRAL]
        bear_idx = [s for s, r in self._state_to_regime.items() if r == self.BEAR]

        proba = np.zeros((len(features), 3))
        if bull_idx:
            proba[:, 0] = gamma[:, bull_idx[0]]
        if neutral_idx:
            proba[:, 1] = gamma[:, neutral_idx[0]]
        if bear_idx:
            proba[:, 2] = gamma[:, bear_idx[0]]

        return proba

    def current_regime(self, prices: np.ndarray) -> str:
        """
        Return the most recent regime label.

        Parameters
        ----------
        prices : np.ndarray (T,)

        Returns
        -------
        str
            Most recent regime: BULL | NEUTRAL | BEAR
        """
        features = self.prepare_features(prices)
        states = self._hmm.predict(features)
        return self._state_to_regime[states[-1]]

    @property
    def hmm_model(self) -> GaussianHMM:
        """Return the underlying GaussianHMM instance."""
        return self._hmm


# --------------------------------------------------------------------------- #
# RegimeAwareStrategy                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class RegimeAwareStrategyParams:
    """Parameters for RegimeAwareStrategy."""
    momentum_threshold: float = 0.01
    """Return threshold for generating signals within each regime."""
    bull_position_boost: float = 1.2
    """Position size multiplier when in BULL regime."""
    bear_position_scale: float = 0.5
    """Position size multiplier when in BEAR regime."""
    neutral_position_scale: float = 0.8
    """Position size multiplier when in NEUTRAL regime."""
    lookback: int = 60
    """Lookback window for regime detection."""


class RegimeAwareStrategy:
    """
    Strategy that adapts its position sizing and signal generation to the
    current detected market regime (BULL / NEUTRAL / BEAR).

    This class implements a simple regime-aware momentum strategy:
      - BULL  : confirm momentum with higher conviction; larger position sizes.
      - BEAR  : require stronger momentum signals; reduce position sizes.
      - NEUTRAL : use SMA crossover for direction; moderate position sizes.

    The regime is detected online using a rolling Gaussian HMM fitted to
    recent log-returns and realised volatility.

    Parameters
    ----------
    params : RegimeAwareStrategyParams, optional
        Strategy parameters.
    detector : MarketRegimeDetector, optional
        Pre-configured regime detector. If None, a default one is created.
    verbose : bool, default False
        Print regime transitions.

    Methods
    -------
    generate_signal(prices: np.ndarray) -> int
        Returns 1 (long), -1 (short), or 0 (hold).
    get_position_size(signal: int, prices: np.ndarray) -> float
        Returns scaled position size based on current regime.
    update_detector(prices: np.ndarray) -> None
        Retrain the HMM with the latest price window.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))
    >>> strategy = RegimeAwareStrategy()
    >>> sig = strategy.generate_signal(prices)
    >>> pos = strategy.get_position_size(sig, prices)
    """

    LONG: int = 1
    SHORT: int = -1
    HOLD: int = 0

    def __init__(
        self,
        params: Optional[RegimeAwareStrategyParams] = None,
        detector: Optional[MarketRegimeDetector] = None,
        verbose: bool = False,
    ) -> None:
        self.params = params or RegimeAwareStrategyParams()
        self.detector = detector or MarketRegimeDetector(
            config=MarketRegimeDetectorConfig(
                lookback=self.params.lookback,
                volatility_window=20,
            ),
            verbose=False,
        )
        self.verbose = verbose
        self._last_regime: Optional[str] = None

    def _price_to_features(self, prices: np.ndarray) -> np.ndarray:
        """Prepare features from price array, using rolling window of lookback."""
        prices = np.asarray(prices, dtype=np.float64).flatten()
        if len(prices) < self.params.lookback:
            raise ValueError(
                f"Need at least lookback={self.params.lookback} prices, got {len(prices)}"
            )
        return prices[-self.params.lookback:]

    def generate_signal(self, prices: np.ndarray) -> int:
        """
        Generate a trading signal based on momentum and current regime.

        Signal logic:
          BULL   : buy if daily_return > momentum_threshold, else hold.
          BEAR   : buy only if daily_return > 1.5 * momentum_threshold.
          NEUTRAL: buy if SMA(20) > SMA(50), else sell/hold.

        Parameters
        ----------
        prices : np.ndarray (T,)
            Recent close prices (at least lookback values).

        Returns
        -------
        signal : int
            1 (LONG), -1 (SHORT), or 0 (HOLD).
        """
        win = self._price_to_features(prices)
        self.detector.fit(win)
        regime = self.detector.current_regime(win)

        if self.verbose and regime != self._last_regime:
            print(f"[RegimeAwareStrategy] Regime switched: {self._last_regime} -> {regime}")
        self._last_regime = regime

        ret_window = self.detector.config.return_window
        daily_return = np.log(win[-1] / win[-1 - ret_window])

        sma_20 = win[-20:].mean() if len(win) >= 20 else win.mean()
        sma_50 = win.mean() if len(win) >= 50 else win.mean()

        if regime == MarketRegimeDetector.BULL:
            return self.LONG if daily_return > self.params.momentum_threshold else self.HOLD
        elif regime == MarketRegimeDetector.BEAR:
            return self.LONG if daily_return > 1.5 * self.params.momentum_threshold else self.HOLD
        else:  # NEUTRAL
            if sma_20 > sma_50:
                return self.LONG
            elif sma_20 < sma_50:
                return self.SHORT
            return self.HOLD

    def get_position_size(self, signal: int, prices: np.ndarray) -> float:
        """
        Return a regime-scaled position size.

        The base position size is |signal| (0 or 1), multiplied by:
          BULL    -> bull_position_boost
          BEAR    -> bear_position_scale
          NEUTRAL -> neutral_position_scale

        Parameters
        ----------
        signal : int
            Output of generate_signal.
        prices : np.ndarray (T,)
            Recent close prices.

        Returns
        -------
        float
            Scaled position size in [0, 1].
        """
        win = self._price_to_features(prices)
        self.detector.fit(win)
        regime = self.detector.current_regime(win)

        regime_multiplier = {
            MarketRegimeDetector.BULL: self.params.bull_position_boost,
            MarketRegimeDetector.NEUTRAL: self.params.neutral_position_scale,
            MarketRegimeDetector.BEAR: self.params.bear_position_scale,
        }.get(regime, 1.0)

        return abs(signal) * regime_multiplier

    def update_detector(self, prices: np.ndarray) -> None:
        """
        Retrain the HMM with the latest rolling window of prices.

        Parameters
        ----------
        prices : np.ndarray (T,)
            Full price series.
        """
        prices = np.asarray(prices, dtype=np.float64).flatten()
        if len(prices) < self.params.lookback:
            warnings.warn(
                f"update_detector: only {len(prices)} prices, need {self.params.lookback}. "
                "Skipping retrain."
            )
            return
        win = prices[-self.params.lookback:]
        self.detector.fit(win)
