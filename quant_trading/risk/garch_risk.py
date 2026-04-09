"""
GARCH Volatility Models for Risk Management.

Implements GARCH family models in pure NumPy + scipy.optimize.
No arch/garch third-party libraries used.

GARCH(1,1):     h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
EGARCH(1,1):    log(h_t) = omega + alpha * |r_{t-1}|/sqrt(h_{t-1}) + gamma * r_{t-1}/sqrt(h_{t-1}) + beta * log(h_{t-1})
GJR-GARCH(1,1): h_t = omega + (alpha + gamma * I_{t-1<0}) * r_{t-1}^2 + beta * h_{t-1}

References:
    Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
    Nelson, D.B. (1991). Conditional heteroskedasticity in asset returns.
    Glosten, L., Jagannathan, R., & Runkle, D. (1993). On the relation between expected return and volatility.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple, Dict, Any, List


class GARCHModel:
    """
    GARCH(1,1) Volatility Model in pure NumPy.

    The GARCH(1,1) model estimates time-varying variance:
        h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}

    where:
        omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (stationarity)

    Log-likelihood (Gaussian):
        LL = sum(-0.5 * log(2*pi) - 0.5*log(h_t) - 0.5*r_t^2/h_t)
           = sum(-log(h_t) - r_t^2/h_t) + const  (dropping constants)

    Example:
        >>> returns = np.random.randn(1000) * 0.02  # 2% daily volatility simulated
        >>> model = GARCHModel()
        >>> model.fit(returns)
        >>> vol = model.predict_volatility()
        >>> forecasts = model.forecast(horizon=10)
    """

    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.

        Args:
            p: GARCH lag order (number of lagged squared returns)
            q: ARCH lag order (number of lagged conditional variances)
        """
        if p != 1 or q != 1:
            raise NotImplementedError("Only GARCH(1,1) is implemented in pure NumPy")
        self.p = p
        self.q = q
        self.params: Optional[np.ndarray] = None
        self.omega: Optional[float] = None
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.residuals: Optional[np.ndarray] = None
        self.conditional_vol: Optional[np.ndarray] = None
        self.log_likelihood: Optional[float] = None
        self._fitted = False

    def _garch_loglik(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood of GARCH(1,1) model.

        Args:
            params: [omega, alpha, beta]
            returns: asset returns array

        Returns:
            Negative log-likelihood (for minimization)
        """
        omega, alpha, beta = params
        n = len(returns)

        # Initialize conditional variance at unconditional variance
        mean_var = np.var(returns)
        h = np.full(n, mean_var)

        # Recursive conditional variance computation
        for t in range(1, n):
            h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]

        # Avoid numerical issues
        h = np.maximum(h, 1e-10)

        # Log-likelihood: sum(-log(h) - r^2/h)
        ll = -np.log(h) - (returns ** 2) / h

        return -np.sum(ll)  # negative for minimization

    def _garch_variance(self, returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
        """
        Compute conditional variances given parameters.

        Args:
            returns: returns array
            omega, alpha, beta: model parameters

        Returns:
            Array of conditional variances
        """
        n = len(returns)
        mean_var = np.var(returns)
        h = np.full(n, mean_var)

        for t in range(1, n):
            h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]

        return np.maximum(h, 1e-10)

    def fit(self, returns: np.ndarray, init_params: Optional[np.ndarray] = None) -> "GARCHModel":
        """
        Fit GARCH(1,1) model to returns series.

        Args:
            returns: 1D array of asset returns
            init_params: Optional initial parameter guesses [omega, alpha, beta]

        Returns:
            self (for method chaining)
        """
        returns = np.asarray(returns).flatten()
        if len(returns) < 20:
            raise ValueError("Need at least 20 observations to fit GARCH model")

        # Initial parameters: omega, alpha, beta
        # Start with reasonable defaults: low persistence
        if init_params is None:
            init_params = np.array([np.var(returns) * 0.05, 0.08, 0.90])

        # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]
        constraints = {"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]}  # alpha + beta < 1

        result = minimize(
            self._garch_loglik,
            init_params,
            args=(returns,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8}
        )

        if not result.success:
            import warnings
            warnings.warn(f"GARCH optimization did not converge: {result.message}")

        self.omega, self.alpha, self.beta = result.x
        self.params = result.x
        self.log_likelihood = -result.fun
        self.conditional_vol = self._garch_variance(returns, self.omega, self.alpha, self.beta)
        self.residuals = returns / np.sqrt(self.conditional_vol)
        self._fitted = True

        return self

    def predict_volatility(self, horizon: int = 1) -> np.ndarray:
        """
        Predict volatility for next horizon periods.

        Args:
            horizon: number of periods to forecast

        Returns:
            Array of forecasted volatilities (annualized if returns are daily)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        # Unconditional variance (long-run)
        last_var = self.conditional_vol[-1]
        pers = self.alpha + self.beta
        unc_var = self.omega / (1 - pers)

        # GARCH forecasts converge to unconditional variance
        h = np.zeros(horizon)
        h[0] = self.omega + (self.alpha + self.beta) * last_var
        for t in range(1, horizon):
            h[t] = self.omega + (self.alpha + self.beta) * h[t - 1]

        return np.sqrt(np.maximum(h, 1e-10))

    def forecast(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        """
        Forecast volatility and variance for future periods.

        Args:
            horizon: number of periods to forecast

        Returns:
            Dictionary with 'variance' and 'volatility' arrays
        """
        var = self.predict_volatility(horizon) ** 2
        return {
            "variance": var,
            "volatility": np.sqrt(var)
        }

    @property
    def persistence(self) -> float:
        """GARCH(1,1) persistence parameter (alpha + beta)."""
        if self.params is None:
            raise RuntimeError("Model must be fitted first")
        return self.alpha + self.beta

    @property
    def half_life(self) -> float:
        """Approximate half-life of volatility shocks (in periods)."""
        pers = self.persistence
        if pers >= 1:
            return np.inf
        return np.log(0.5) / np.log(pers)


class EGARCHModel:
    """
    Exponential GARCH (EGARCH) Model for asymmetric volatility.

    The EGARCH(1,1) model:
        log(h_t) = omega + alpha * |r_{t-1}|/sqrt(h_{t-1}) + gamma * r_{t-1}/sqrt(h_{t-1}) + beta * log(h_{t-1})

    This captures the "leverage effect" where negative shocks have
    larger impact on volatility than positive shocks of same magnitude.

    Example:
        >>> model = EGARCHModel()
        >>> model.fit(returns)
        >>> vol = model.predict_volatility()
    """

    def __init__(self):
        """Initialize EGARCH(1,1) model."""
        self.params: Optional[np.ndarray] = None
        self.omega: Optional[float] = None
        self.alpha: Optional[float] = None
        self.gamma: Optional[float] = None
        self.beta: Optional[float] = None
        self.conditional_vol: Optional[np.ndarray] = None
        self.log_likelihood: Optional[float] = None
        self._fitted = False

    def _egarch_loglik(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood of EGARCH(1,1) model.

        Args:
            params: [omega, alpha, gamma, beta]
            returns: asset returns array

        Returns:
            Negative log-likelihood
        """
        omega, alpha, gamma, beta = params
        n = len(returns)

        # Initialize log-variance at unconditional value
        log_h = np.full(n, np.log(np.var(returns) + 1e-10))

        # Clipping for numerical stability
        LOG_H_MIN = -20  # ~4.5e-9 variance floor
        LOG_H_MAX = 10   # ~22000 variance ceiling

        for t in range(1, n):
            # Compute z with clipping to avoid overflow
            exp_half = np.exp(np.clip(log_h[t - 1] / 2, LOG_H_MIN / 2, LOG_H_MAX / 2))
            z = returns[t - 1] / (exp_half + 1e-100)
            log_h[t] = omega + alpha * (np.abs(z) - np.sqrt(2 / np.pi)) + gamma * z + beta * log_h[t - 1]
            log_h[t] = np.clip(log_h[t], LOG_H_MIN, LOG_H_MAX)

        h = np.exp(log_h)
        h = np.maximum(h, 1e-10)

        ll = -0.5 * (np.log(2 * np.pi) + log_h + returns ** 2 / h)
        return -np.sum(ll)

    def _egarch_variance(self, returns: np.ndarray, omega: float, alpha: float,
                         gamma: float, beta: float) -> np.ndarray:
        """Compute conditional variances given parameters."""
        n = len(returns)
        log_h = np.full(n, np.log(np.var(returns) + 1e-10))

        LOG_H_MIN, LOG_H_MAX = -20, 10

        for t in range(1, n):
            exp_half = np.exp(np.clip(log_h[t - 1] / 2, LOG_H_MIN / 2, LOG_H_MAX / 2))
            z = returns[t - 1] / (exp_half + 1e-100)
            log_h[t] = omega + alpha * (np.abs(z) - np.sqrt(2 / np.pi)) + gamma * z + beta * log_h[t - 1]
            log_h[t] = np.clip(log_h[t], LOG_H_MIN, LOG_H_MAX)

        return np.exp(log_h)

    def fit(self, returns: np.ndarray, init_params: Optional[np.ndarray] = None) -> "EGARCHModel":
        """
        Fit EGARCH(1,1) model to returns series.

        Args:
            returns: 1D array of asset returns
            init_params: Optional initial [omega, alpha, gamma, beta]

        Returns:
            self
        """
        returns = np.asarray(returns).flatten()
        if len(returns) < 20:
            raise ValueError("Need at least 20 observations")

        if init_params is None:
            # Typical EGARCH params: asymmetry (gamma < 0)
            init_params = np.array([-0.1, 0.06, -0.02, 0.93])

        # Bounds: omega unconstrained, alpha > 0, gamma in [-1, 1], beta in (0, 1)
        bounds = [(-10, 10), (1e-8, 1), (-1, 1), (1e-8, 0.999)]
        constraints = {"type": "ineq", "fun": lambda x: 0.999 - x[3]}  # beta < 1

        result = minimize(
            self._egarch_loglik,
            init_params,
            args=(returns,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if not result.success:
            import warnings
            warnings.warn(f"EGARCH optimization did not converge: {result.message}")

        self.omega, self.alpha, self.gamma, self.beta = result.x
        self.params = result.x
        self.log_likelihood = -result.fun
        self.conditional_vol = np.sqrt(self._egarch_variance(returns, self.omega, self.alpha, self.gamma, self.beta))
        self._fitted = True

        return self

    def predict_volatility(self, horizon: int = 1) -> np.ndarray:
        """Forecast volatility for next horizon periods."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        last_log_h = np.log(self.conditional_vol[-1] ** 2)
        log_h = np.zeros(horizon)

        log_h[0] = self.omega + (self.alpha + self.gamma + self.beta) * last_log_h
        for t in range(1, horizon):
            log_h[t] = self.omega + (self.alpha + self.gamma + self.beta) * log_h[t - 1]

        return np.sqrt(np.maximum(np.exp(log_h), 1e-10))

    def forecast(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        """Forecast variance and volatility."""
        vol = self.predict_volatility(horizon)
        return {"variance": vol ** 2, "volatility": vol}

    @property
    def persistence(self) -> float:
        """EGARCH persistence parameter."""
        if self.params is None:
            raise RuntimeError("Model must be fitted first")
        return self.beta


class GJRGARCHModel:
    """
    GJR-GARCH (Glosten-Jagannathan-Runkle GARCH) Model for leverage effects.

    The GJR-GARCH(1,1) model:
        h_t = omega + (alpha + gamma * I_{t-1<0}) * r_{t-1}^2 + beta * h_{t-1}

    where I_{t-1<0} is an indicator that equals 1 when r_{t-1} < 0 (negative shock).

    This captures asymmetric volatility where negative returns
    increase volatility more than positive returns of same magnitude.

    Example:
        >>> model = GJRGARCHModel()
        >>> model.fit(returns)
        >>> vol = model.predict_volatility()
    """

    def __init__(self):
        """Initialize GJR-GARCH(1,1) model."""
        self.params: Optional[np.ndarray] = None
        self.omega: Optional[float] = None
        self.alpha: Optional[float] = None
        self.gamma: Optional[float] = None
        self.beta: Optional[float] = None
        self.conditional_vol: Optional[np.ndarray] = None
        self.log_likelihood: Optional[float] = None
        self._fitted = False

    def _gjr_garch_loglik(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute negative log-likelihood of GJR-GARCH(1,1) model.

        Args:
            params: [omega, alpha, gamma, beta]
            returns: asset returns array

        Returns:
            Negative log-likelihood
        """
        omega, alpha, gamma, beta = params
        n = len(returns)

        mean_var = np.var(returns)
        h = np.full(n, mean_var)

        for t in range(1, n):
            indicator = 1.0 if returns[t - 1] < 0 else 0.0
            h[t] = omega + (alpha + gamma * indicator) * returns[t - 1] ** 2 + beta * h[t - 1]

        h = np.maximum(h, 1e-10)
        ll = -np.log(h) - (returns ** 2) / h

        return -np.sum(ll)

    def _gjr_garch_variance(self, returns: np.ndarray, omega: float, alpha: float,
                            gamma: float, beta: float) -> np.ndarray:
        """Compute conditional variances given parameters."""
        n = len(returns)
        mean_var = np.var(returns)
        h = np.full(n, mean_var)

        for t in range(1, n):
            indicator = 1.0 if returns[t - 1] < 0 else 0.0
            h[t] = omega + (alpha + gamma * indicator) * returns[t - 1] ** 2 + beta * h[t - 1]

        return np.maximum(h, 1e-10)

    def fit(self, returns: np.ndarray, init_params: Optional[np.ndarray] = None) -> "GJRGARCHModel":
        """
        Fit GJR-GARCH(1,1) model to returns series.

        Args:
            returns: 1D array of asset returns
            init_params: Optional initial [omega, alpha, gamma, beta]

        Returns:
            self
        """
        returns = np.asarray(returns).flatten()
        if len(returns) < 20:
            raise ValueError("Need at least 20 observations")

        if init_params is None:
            # gamma > 0 indicates leverage effect (negative shocks increase vol)
            init_params = np.array([np.var(returns) * 0.05, 0.06, 0.04, 0.90])

        # Constraints: omega > 0, alpha >= 0, gamma >= 0, beta >= 0, alpha + gamma + beta < 1
        bounds = [(1e-8, None), (1e-8, 0.999), (0, 0.999), (1e-8, 0.999)]
        constraints = {"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2] - x[3]}

        result = minimize(
            self._gjr_garch_loglik,
            init_params,
            args=(returns,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if not result.success:
            import warnings
            warnings.warn(f"GJR-GARCH optimization did not converge: {result.message}")

        self.omega, self.alpha, self.gamma, self.beta = result.x
        self.params = result.x
        self.log_likelihood = -result.fun
        self.conditional_vol = np.sqrt(self._gjr_garch_variance(returns, self.omega, self.alpha, self.gamma, self.beta))
        self._fitted = True

        return self

    def predict_volatility(self, horizon: int = 1) -> np.ndarray:
        """Forecast volatility for next horizon periods."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        last_var = self.conditional_vol[-1] ** 2
        pers = self.alpha + self.gamma + self.beta

        # For forecasting, we use average leverage (assuming 50% negative shocks)
        h = np.zeros(horizon)
        h[0] = self.omega + (self.alpha + 0.5 * self.gamma + self.beta) * last_var
        for t in range(1, horizon):
            h[t] = self.omega + (self.alpha + 0.5 * self.gamma + self.beta) * h[t - 1]

        return np.sqrt(np.maximum(h, 1e-10))

    def forecast(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        """Forecast variance and volatility."""
        vol = self.predict_volatility(horizon)
        return {"variance": vol ** 2, "volatility": vol}

    @property
    def persistence(self) -> float:
        """GJR-GARCH persistence (alpha + gamma/2 + beta for average leverage)."""
        if self.params is None:
            raise RuntimeError("Model must be fitted first")
        return self.alpha + self.gamma / 2 + self.beta


class VolatilityForecaster:
    """
    Volatility Forecaster using fitted GARCH family models.

    Provides unified interface to forecast volatility from any
    supported GARCH model type.

    Example:
        >>> forecaster = VolatilityForecaster(model_type="garch")
        >>> forecaster.fit(returns)
        >>> vol_forecast = forecaster.forecast(horizon=10)
    """

    SUPPORTED_MODELS = {"garch", "egarch", "gjr_garch"}

    def __init__(self, model_type: str = "garch"):
        """
        Initialize forecaster with specified model type.

        Args:
            model_type: One of "garch", "egarch", "gjr_garch"
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Choose from {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.model: Optional[Any] = None
        self._fitted = False

    def fit(self, returns: np.ndarray) -> "VolatilityForecaster":
        """
        Fit the volatility model to returns.

        Args:
            returns: 1D array of asset returns

        Returns:
            self
        """
        if self.model_type == "garch":
            self.model = GARCHModel().fit(returns)
        elif self.model_type == "egarch":
            self.model = EGARCHModel().fit(returns)
        elif self.model_type == "gjr_garch":
            self.model = GJRGARCHModel().fit(returns)

        self._fitted = True
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility for future periods.

        Args:
            horizon: number of periods to forecast

        Returns:
            Array of forecasted volatilities
        """
        if not self._fitted:
            raise RuntimeError("Forecaster must be fitted before forecasting")

        return self.model.predict_volatility(horizon)

    def predict_variance(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast variance for future periods.

        Args:
            horizon: number of periods to forecast

        Returns:
            Array of forecasted variances
        """
        if not self._fitted:
            raise RuntimeError("Forecaster must be fitted before forecasting")

        return self.model.forecast(horizon)["variance"]

    def rolling_forecast(self, returns: np.ndarray, window: int, horizon: int = 1) -> np.ndarray:
        """
        Compute rolling volatility forecasts.

        Args:
            returns: Full returns series
            window: Size of rolling window for fitting
            horizon: Forecast horizon

        Returns:
            Array of forecasted volatilities
        """
        n = len(returns)
        forecasts = np.full(n, np.nan)

        for i in range(window, n):
            window_returns = returns[i - window:i]
            if self.model_type == "garch":
                model = GARCHModel().fit(window_returns)
            elif self.model_type == "egarch":
                model = EGARCHModel().fit(window_returns)
            elif self.model_type == "gjr_garch":
                model = GJRGARCHModel().fit(window_returns)

            forecasts[i] = model.predict_volatility(horizon)[0]

        return forecasts


class RiskMetrics:
    """
    Risk Metrics Calculator using GARCH volatility forecasts.

    Computes Value at Risk (VaR), Conditional VaR (CVaR),
    and Expected Shortfall (ES) using GARCH-based volatility.

    VaR_alpha: The loss threshold that is exceeded with probability (1-alpha)
    CVaR_alpha: Expected loss given that VaR_alpha is exceeded
    ES_alpha: Same as CVaR (expected value of losses beyond VaR)

    Example:
        >>> returns = np.random.randn(1000) * 0.02
        >>> risk = RiskMetrics()
        >>> risk.fit_garch(returns)
        >>> var_95 = risk.compute_var(0.95)  # 1-day 95% VaR
        >>> cvar_95 = risk.compute_cvar(0.95)
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize RiskMetrics calculator.

        Args:
            confidence_level: Confidence level for VaR/ES (default 0.95)
        """
        self.confidence_level = confidence_level
        self.returns: Optional[np.ndarray] = None
        self.garch_model: Optional[GARCHModel] = None
        self.conditional_vol: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self._fitted = False

    def fit_garch(self, returns: np.ndarray, p: int = 1, q: int = 1) -> "RiskMetrics":
        """
        Fit GARCH(1,1) model to returns.

        Args:
            returns: 1D array of asset returns
            p, q: GARCH orders (only p=q=1 fully implemented)

        Returns:
            self
        """
        self.returns = np.asarray(returns).flatten()
        self.garch_model = GARCHModel(p=p, q=q)
        self.garch_model.fit(self.returns)
        self.conditional_vol = self.garch_model.conditional_vol
        self.residuals = self.garch_model.residuals
        self._fitted = True
        return self

    def compute_var(self, confidence_level: Optional[float] = None,
                     horizon: int = 1, method: str = "gaussian") -> float:
        """
        Compute Value at Risk (VaR).

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            horizon: Time horizon in days
            method: "gaussian" for parametric, "historical" for historical simulation

        Returns:
            VaR as a positive number representing potential loss
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        if method == "gaussian":
            # Parametric VaR using GARCH volatility
            vol = self.garch_model.predict_volatility(horizon)[0]
            z = np.abs(np.random.randn(100000))  # Use simulation for exact quantile
            z_quantile = np.percentile(z, confidence_level * 100)
            var = z_quantile * vol * np.sqrt(horizon)
        elif method == "historical":
            # Historical simulation using recent residuals
            recent_resid = self.residuals[-252:] if len(self.residuals) > 252 else self.residuals
            vol_current = self.conditional_vol[-1]
            simulated_returns = recent_resid * np.sqrt(vol_current)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        else:
            raise ValueError(f"Unknown method: {method}")

        return var

    def compute_cvar(self, confidence_level: Optional[float] = None,
                     horizon: int = 1, method: str = "gaussian") -> float:
        """
        Compute Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that VaR is exceeded.

        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            method: "gaussian" or "historical"

        Returns:
            CVaR as a positive number
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        if method == "gaussian":
            # Parametric CVaR using GARCH volatility
            vol = self.garch_model.predict_volatility(horizon)[0]
            # For Gaussian, CVaR = vol * n(z_alpha) / (1 - alpha)
            from scipy.stats import norm
            alpha = 1 - confidence_level
            z_alpha = norm.ppf(alpha)
            cvar = vol * np.sqrt(horizon) * norm.pdf(z_alpha) / alpha
        elif method == "historical":
            recent_resid = self.residuals[-252:] if len(self.residuals) > 252 else self.residuals
            vol_current = self.conditional_vol[-1]
            simulated_returns = recent_resid * np.sqrt(vol_current)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            tail_losses = simulated_returns[simulated_returns <= -var]
            cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        else:
            raise ValueError(f"Unknown method: {method}")

        return cvar

    def compute_es(self, confidence_level: Optional[float] = None,
                    horizon: int = 1, n_simulations: int = 100000) -> float:
        """
        Compute Expected Shortfall (ES) via Monte Carlo simulation.

        ES is the average loss conditional on exceeding VaR.

        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            n_simulations: Number of Monte Carlo paths

        Returns:
            Expected Shortfall
        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        vol = self.garch_model.predict_volatility(horizon)[0]
        alpha = 1 - confidence_level

        # Simulate returns using GARCH volatility
        simulated_returns = np.random.randn(n_simulations) * vol * np.sqrt(horizon)

        # Compute VaR and ES
        var = -np.percentile(simulated_returns, alpha * 100)
        tail_losses = simulated_returns[simulated_returns <= -var]
        es = -np.mean(tail_losses) if len(tail_losses) > 0 else var

        return es

    def rolling_var(self, returns: np.ndarray, window: int = 252,
                    confidence_level: float = 0.95, horizon: int = 1) -> np.ndarray:
        """
        Compute rolling VaR using expanding or fixed window.

        Args:
            returns: Full returns series
            window: Rolling window size
            confidence_level: VaR confidence level
            horizon: VaR horizon

        Returns:
            Array of rolling VaR estimates
        """
        n = len(returns)
        var_series = np.full(n, np.nan)

        for i in range(window, n):
            window_returns = returns[i - window:i]
            try:
                model = GARCHModel().fit(window_returns)
                vol = model.predict_volatility(horizon)[0]
                from scipy.stats import norm
                alpha = 1 - confidence_level
                z = norm.ppf(alpha)
                var_series[i] = -z * vol * np.sqrt(horizon)
            except Exception:
                continue

        return var_series

    def summary(self, horizon: int = 1) -> Dict[str, Any]:
        """
        Return comprehensive risk summary.

        Args:
            horizon: Risk horizon in days

        Returns:
            Dictionary with VaR, CVaR, ES at multiple confidence levels
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        levels = [0.90, 0.95, 0.99]
        summary = {}

        for cl in levels:
            var = self.compute_var(cl, horizon, method="gaussian")
            cvar = self.compute_cvar(cl, horizon, method="gaussian")
            summary[f"VaR_{int(cl*100)}"] = var
            summary[f"CVaR_{int(cl*100)}"] = cvar

        summary["GARCH_params"] = {
            "omega": self.garch_model.omega,
            "alpha": self.garch_model.alpha,
            "beta": self.garch_model.beta,
            "persistence": self.garch_model.persistence,
            "half_life": self.garch_model.half_life
        }
        summary["log_likelihood"] = self.garch_model.log_likelihood

        return summary


# Convenience functions for quick use
def fit_garch11(returns: np.ndarray) -> GARCHModel:
    """
    Fit GARCH(1,1) model to returns (convenience function).

    Args:
        returns: 1D array of asset returns

    Returns:
        Fitted GARCHModel
    """
    return GARCHModel().fit(returns)


def forecast_volatility(returns: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Forecast volatility using GARCH(1,1).

    Args:
        returns: Asset returns
        horizon: Forecast horizon

    Returns:
        Forecasted volatility array
    """
    return GARCHModel().fit(returns).predict_volatility(horizon)


def compute_garch_var(returns: np.ndarray, confidence_level: float = 0.95,
                       horizon: int = 1) -> float:
    """
    Compute VaR using GARCH(1,1) volatility.

    Args:
        returns: Asset returns
        confidence_level: VaR confidence level
        horizon: Time horizon

    Returns:
        VaR estimate
    """
    risk = RiskMetrics(confidence_level)
    risk.fit_garch(returns)
    return risk.compute_var(confidence_level, horizon)


__all__ = [
    "GARCHModel",
    "EGARCHModel",
    "GJRGARCHModel",
    "VolatilityForecaster",
    "RiskMetrics",
    "fit_garch11",
    "forecast_volatility",
    "compute_garch_var",
]
