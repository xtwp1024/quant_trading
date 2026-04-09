"""AgentQuant: Gemini 2.5 Flash-driven autonomous quantitative research agent.

Absorbed from D:/Hive/Data/trading_repos/AgentQuant/
- Gemini 2.5 Flash powered autonomous research
- Walk-Forward validation framework
- Market regime detection (Bull/Bear/Crisis)
- Ablation experiment framework

Classes:
    ResearchTask   -- Research task descriptor
    ResearchResult -- Research output with backtest & regime info
    RegimeDetector -- Market regime detection (Bull/Bear/Crisis/Unknown)
    WalkForwardValidator -- Walk-forward rolling window validator
    AgentQuant     -- Main autonomous research agent

Bilingual docstrings (English primary, Chinese comments).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ResearchTask",
    "ResearchResult",
    "RegimeDetector",
    "WalkForwardValidator",
    "AgentQuant",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResearchTask:
    """Research task descriptor.

    Attributes:
        symbol: Ticker symbol, e.g. "SPY".
        hypothesis: Free-text research hypothesis to validate.
        regime: Market regime at creation time. One of
            'bull' | 'bear' | 'crisis' | 'unknown'.
        completed_at: Timestamp when research finished, or None if pending.
    """
    symbol: str
    hypothesis: str
    regime: str = "unknown"
    completed_at: Optional[datetime] = None


@dataclass
class ResearchResult:
    """Output of a completed research task.

    Attributes:
        task: Original ResearchTask that was executed.
        data_analysis: LLM-generated textual analysis of the data.
        backtest_results: Dictionary of backtest metrics
            (e.g. sharpe, return, max_drawdown, win_rate).
        regime: Detected market regime during research.
        verdict: Research conclusion. One of
            'accept' | 'reject' | 'needs_more_data'.
        confidence: Confidence score in [0.0, 1.0].
        walk_forward_score: Walk-forward validation score,
            or None if walk-forward was not run.
    """
    task: ResearchTask
    data_analysis: str
    backtest_results: dict
    regime: str
    verdict: str
    confidence: float
    walk_forward_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Regime detection  -- Bull / Bear / Crisis / Unknown
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Market regime detector.

    Classifies the current market state using VIX level and
    63-day price momentum:

    - Crisis  : VIX > 30  AND  momentum_63d < -10 %
    - Bear   : VIX > 20  AND  momentum_63d < -5 %
    - Bull   : VIX <= 20 AND  momentum_63d > +5 %
    - Unknown: all other cases

    Uses the same heuristic as the original AgentQuant regime.py.
    """

    def __init__(self, vix_ticker: str = "VIX") -> None:
        """
        Args:
            vix_ticker: Ticker symbol for the VIX index (default "VIX").
        """
        self.vix_ticker = vix_ticker

    def detect(self, price_series: pd.Series) -> str:
        """Detect current market regime from a price series.

        Args:
            price_series: pd.Series of close prices, indexed by date.

        Returns:
            One of 'crisis', 'bear', 'bull', 'unknown'.
        """
        if price_series.empty:
            return "unknown"

        # Attempt to fetch VIX from the series if present
        # (For standalone use without external data feed)
        vix_val: float = 20.0  # neutral default
        if isinstance(price_series, pd.DataFrame):
            if self.vix_ticker in price_series.columns:
                vix_val = float(price_series[self.vix_ticker].iloc[-1])
            elif "vix_close" in price_series.columns:
                vix_val = float(price_series["vix_close"].iloc[-1])
        elif hasattr(price_series, "name") and price_series.name == self.vix_ticker:
            vix_val = float(price_series.iloc[-1])

        # 63-day (approx. 3-month) momentum
        if len(price_series) < 63:
            mom63d = 0.0
        else:
            mom63d = (price_series.iloc[-1] / price_series.iloc[-63]) - 1.0

        # Rule-based classification (same as original AgentQuant)
        if vix_val > 30:
            if mom63d < -0.10:
                return "crisis"
            else:
                return "bear"
        elif vix_val > 20:
            if mom63d > 0.05:
                return "bull"
            elif mom63d < -0.05:
                return "bear"
            else:
                return "unknown"
        else:  # VIX <= 20
            if mom63d > 0.05:
                return "bull"
            else:
                return "unknown"

    def detect_from_features(self, features_df: pd.DataFrame) -> str:
        """Detect regime from a pre-computed features DataFrame.

        Args:
            features_df: DataFrame with at least 'vix_close' and
                'momentum_63d' columns (same schema as AgentQuant engine).

        Returns:
            One of 'crisis', 'bear', 'bull', 'unknown'.
        """
        if features_df.empty:
            return "unknown"

        latest = features_df.iloc[-1]
        vix_val = float(latest.get("vix_close", 20.0))
        mom63d = float(latest.get("momentum_63d", 0.0))

        if vix_val > 30:
            if mom63d < -0.10:
                return "crisis"
            else:
                return "bear"
        elif vix_val > 20:
            if mom63d > 0.05:
                return "bull"
            elif mom63d < -0.05:
                return "bear"
            else:
                return "unknown"
        else:
            if mom63d > 0.05:
                return "bull"
            else:
                return "unknown"


# ---------------------------------------------------------------------------
# Walk-Forward validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """Walk-Forward rolling window validator.

    Splits the available price history into rolling train/test windows
    and computes:

    - **PBO** (Probability of Backtest Overfitting):
      fraction of windows where test Sharpe < train Sharpe.
    - **mean_train_sharpe**: average train-window Sharpe.
    - **mean_test_sharpe**:  average test-window Sharpe.
    - **degradation**:       mean (train_sharpe - test_sharpe).

    A PBO close to 50 % indicates the strategy is likely overfit;
    a low PBO with positive degradation suggests genuine edge.

    References:
        - Bailey, D. H., & Lopez de Prado, M. (2014).
          "The Deflated Sharpe Ratio".
        - Walk-Forward analysis as used in original AgentQuant.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        warmup_window: int = 252,
    ) -> None:
        """
        Args:
            train_window: Number of trading days for each train slice.
            test_window:  Number of trading days for each test slice.
            warmup_window: Number of warmup days prepended for indicator
                calculation (e.g. 252 for SMA).
        """
        self.train_window = train_window
        self.test_window = test_window
        self.warmup_window = warmup_window

    def _sharpe(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio from a return series."""
        if returns.empty or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * math.sqrt(periods_per_year)

    def validate(
        self,
        strategy_func: Callable[[pd.DataFrame], pd.Series],
        data: pd.DataFrame,
        price_col: str = "close",
    ) -> dict:
        """Run walk-forward validation.

        Args:
            strategy_func: Callable that takes a price DataFrame
                and returns a pd.Series of daily returns (0 for no position).
            data: DataFrame with at least a 'close' (or price_col) column,
                indexed by date. Must have enough rows for
                warmup + train + test windows.
        Returns:
            Dictionary with keys:
                pbo, mean_train_sharpe, mean_test_sharpe,
                degradation, n_windows, results (list of per-window dicts).
        """
        if price_col not in data.columns:
            raise ValueError(f"Column '{price_col}' not found in data")

        close = data[price_col]
        total_len = self.warmup_window + self.train_window + self.test_window

        if len(data) < total_len:
            # Not enough data – return empty/infinite degradation
            return {
                "pbo": 1.0,
                "mean_train_sharpe": 0.0,
                "mean_test_sharpe": 0.0,
                "degradation": 0.0,
                "n_windows": 0,
                "results": [],
            }

        results: list[dict] = []
        train_sharpes: list[float] = []
        test_sharpes: list[float] = []

        # Slide the window forward by test_window each step
        n_steps = (len(data) - total_len) // self.test_window + 1

        for step in range(n_steps):
            warmup_start = step * self.test_window
            train_start  = warmup_start + self.warmup_window
            train_end    = train_start + self.train_window
            test_start   = train_end
            test_end     = test_start + self.test_window

            if test_end > len(data):
                break

            # Full warmup-augmented slice for indicator computation
            full_slice = data.iloc[warmup_start:test_end]

            # Strategy returns over the full slice
            try:
                strat_returns = strategy_func(full_slice)
            except Exception:
                continue

            if not isinstance(strat_returns, pd.Series) or strat_returns.empty:
                continue

            # Restrict to train and test windows
            train_ret = strat_returns.iloc[
                (train_start - warmup_start): (train_end - warmup_start)
            ]
            test_ret  = strat_returns.iloc[
                (test_start - warmup_start): (test_end - warmup_start)
            ]

            if len(train_ret) < 20 or len(test_ret) < 5:
                continue

            train_sharpe = self._sharpe(train_ret)
            test_sharpe  = self._sharpe(test_ret)

            train_sharpes.append(train_sharpe)
            test_sharpes.append(test_sharpe)

            results.append({
                "step": step,
                "train_start": data.index[train_start] if train_start < len(data) else None,
                "test_start":  data.index[test_start]  if test_start  < len(data) else None,
                "train_sharpe": train_sharpe,
                "test_sharpe":  test_sharpe,
            })

        n_windows = len(train_sharpes)
        if n_windows == 0:
            return {
                "pbo": 1.0,
                "mean_train_sharpe": 0.0,
                "mean_test_sharpe": 0.0,
                "degradation": 0.0,
                "n_windows": 0,
                "results": [],
            }

        # PBO: fraction of windows where test < train
        pbo = sum(1 for tr, te in zip(train_sharpes, test_sharpes) if te < tr) / n_windows

        mean_train = float(np.mean(train_sharpes))
        mean_test  = float(np.mean(test_sharpes))
        degradation = float(np.mean([tr - te for tr, te in zip(train_sharpes, test_sharpes)]))

        return {
            "pbo": pbo,
            "mean_train_sharpe": mean_train,
            "mean_test_sharpe":  mean_test,
            "degradation": degradation,
            "n_windows": n_windows,
            "results": results,
        }


# ---------------------------------------------------------------------------
# Gemini REST client helper
# ---------------------------------------------------------------------------

def _gemini_rest_call(
    model: str,
    prompt: str,
    api_key: str,
    temperature: float = 0.1,
    timeout: float = 30.0,
) -> str:
    """Call Gemini via the Vertex AI REST API (model must be prefixed with
    ``gemini-`` and accessed through the ``generateContent`` endpoint).

    Falls back to a minimal no-op response if the call fails.
    """
    import urllib.request

    url = (
        f"https://gateway.ai.cloud.google.com/v1beta2/projects/"
        f"default-project/locations/us-central1/publishers/google/models/"
        f"{model}:generateContent?alt=sjson"
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 2048,
        },
    })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        # Parse candidate text
        for cand in raw.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    return part["text"]
        return str(raw)
    except Exception as exc:
        return f"[Gemini API error: {exc}]"


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class AgentQuant:
    """Gemini-driven autonomous quantitative research agent.

    Workflow:
        1. Perceive market data (price series, features).
        2. Generate research hypothesis (LLM or provided).
        3. Analyse data & run backtests.
        4. Walk-forward validation.
        5. Output research result with verdict & confidence.

    Args:
        model_name: Gemini model name, e.g. "gemini-2.0-flash".
        api_key: Google Cloud / AI Studio API key.
            Falls back to the GOOGLE_API_KEY environment variable.
        llm_client: Optional pre-constructed LLM client (e.g. langchain).
            If provided, api_key is ignored and the client is used directly.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str = "",
        llm_client: Any = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.llm_client = llm_client
        self._regime_detector = RegimeDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(
        self,
        symbol: str,
        hypothesis: Optional[str] = None,
        price_data: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
    ) -> ResearchResult:
        """Execute a complete research pipeline for *symbol*.

        Args:
            symbol: Ticker symbol, e.g. "SPY".
            hypothesis: Optional pre-defined hypothesis to test.
                If None, one will be generated from the data.
            price_data: DataFrame with at least a 'close' column.
            features: Optional pre-computed features DataFrame.

        Returns:
            ResearchResult with full analysis and verdict.
        """
        # Detect regime
        if features is not None and not features.empty:
            regime = self._regime_detector.detect_from_features(features)
        elif price_data is not None and not price_data.empty:
            regime = self._regime_detector.detect(price_data["close"])
        else:
            regime = "unknown"

        # Create task
        task = ResearchTask(symbol=symbol, hypothesis=hypothesis or "", regime=regime)

        # Generate hypothesis if not provided
        if not task.hypothesis:
            task.hypothesis = self.generate_hypothesis(symbol, {
                "price_data": price_data,
                "features": features,
                "regime": regime,
            })

        # Analyze data
        data_analysis = self._analyze_data(symbol, task.hypothesis, price_data, features)

        # Backtest (simplified – uses price data directly)
        backtest_results = self._run_backtest(symbol, task.hypothesis, price_data)

        # Walk-forward validation
        if price_data is not None and len(price_data) >= 600:
            wf_results = self._run_walk_forward(symbol, price_data)
            walk_forward_score = wf_results.get("mean_test_sharpe", 0.0)
        else:
            walk_forward_score = None
            wf_results = {}

        # Determine verdict
        verdict, confidence = self._compute_verdict(
            backtest_results, wf_results, data_analysis
        )

        task.completed_at = datetime.now()

        return ResearchResult(
            task=task,
            data_analysis=data_analysis,
            backtest_results=backtest_results,
            regime=regime,
            verdict=verdict,
            confidence=confidence,
            walk_forward_score=walk_forward_score,
        )

    def generate_hypothesis(self, symbol: str, market_data: dict) -> str:
        """Use the LLM to generate a research hypothesis from market data.

        Args:
            symbol: Ticker symbol.
            market_data: Dict that may contain 'price_data', 'features',
                and 'regime' keys.

        Returns:
            Free-text hypothesis suitable for backtesting.
        """
        regime  = market_data.get("regime", "unknown")
        features = market_data.get("features")
        price_df = market_data.get("price_data")

        # Build summary string
        summary_parts = [f"Symbol: {symbol}", f"Regime: {regime}"]
        if price_df is not None and not price_df.empty:
            close = price_df["close"]
            summary_parts.append(f"Latest close: {close.iloc[-1]:.2f}")
            if len(close) >= 252:
                annual_ret = (close.iloc[-1] / close.iloc[-252] - 1) * 100
                summary_parts.append(f"1Y return: {annual_ret:.1f}%")
            if len(close) >= 21:
                mom21 = (close.iloc[-1] / close.iloc[-21] - 1) * 100
                summary_parts.append(f"1M momentum: {mom21:.1f}%")
        if features is not None and not features.empty:
            latest = features.iloc[-1]
            for col in ["rsi", "macd", "bb_position"]:
                if col in latest.index:
                    summary_parts.append(f"{col}: {float(latest[col]):.3f}")

        summary = "\n".join(summary_parts)

        prompt = (
            "You are a quantitative researcher. Given the market data below, "
            "generate ONE specific, testable trading hypothesis.\n"
            "Your response must be a single sentence (under 120 characters) "
            "describing the hypothesis, e.g.:\n"
            '"SPY will mean-revert when RSI < 30 and VIX > 25"\n\n'
            f"{summary}\n\n"
            "Hypothesis:"
        )

        if self.llm_client is not None:
            try:
                response = self.llm_client.invoke(prompt)
                return getattr(response, "content", str(response)).strip()
            except Exception:
                pass

        # REST fallback
        if self.api_key:
            result = _gemini_rest_call(self.model_name, prompt, self.api_key)
            # Strip any markdown formatting
            result = re.sub(r"^```.*$", "", result, flags=re.MULTILINE).strip()
            return result.strip()

        return f"{symbol} exhibits short-term mean-reversion on RSI extremes"

    def run_ablation(
        self,
        base_strategy: Callable[[pd.DataFrame], pd.Series],
        remove_components: list[str],
        data: pd.DataFrame,
    ) -> dict:
        """Ablation study: evaluate strategy performance with components removed.

        Args:
            base_strategy: Full strategy function returning returns series.
            remove_components: List of component names to separately ablate.
            data: Price DataFrame.

        Returns:
            Dict mapping component name to its marginal contribution
            (full_sharpe - ablation_sharpe).
        """
        validator = WalkForwardValidator()
        full_result = validator.validate(base_strategy, data)
        full_sharpe = full_result["mean_test_sharpe"]

        contributions: dict[str, float] = {}
        for component in remove_components:
            # Placeholder: in a real system, the strategy would accept
            # a config dict disabling specific components.
            # Here we return the difference as a stub.
            try:
                ablation_result = validator.validate(base_strategy, data)
                ablation_sharpe = ablation_result["mean_test_sharpe"]
            except Exception:
                ablation_sharpe = 0.0
            contributions[component] = full_sharpe - ablation_sharpe

        contributions["_full_sharpe"] = full_sharpe
        return contributions

    def get_current_regime(self, symbol: str) -> str:
        """Placeholder: return current regime for *symbol*.

        In a full deployment this would fetch live data and call
        ``_regime_detector.detect()``.
        """
        return "unknown"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_data(
        self,
        symbol: str,
        hypothesis: str,
        price_data: Optional[pd.DataFrame],
        features: Optional[pd.DataFrame],
    ) -> str:
        """Call LLM to produce a textual data analysis."""
        parts = [f"Symbol: {symbol}", f"Hypothesis: {hypothesis}"]

        if price_data is not None and not price_data.empty:
            close = price_data["close"]
            parts.append(
                f"Price stats (last {len(close)} days): "
                f"mean={close.mean():.2f}, std={close.std():.2f}, "
                f"min={close.min():.2f}, max={close.max():.2f}"
            )
        if features is not None and not features.empty:
            latest = features.iloc[-1]
            parts.append(f"Latest features:\n{latest.to_string()}")

        prompt = (
            "You are a quantitative analyst. Review the market data below "
            "and provide a brief textual analysis relevant to the hypothesis.\n\n"
            + "\n".join(parts)
        )

        if self.llm_client is not None:
            try:
                response = self.llm_client.invoke(prompt)
                return getattr(response, "content", str(response)).strip()
            except Exception:
                pass

        if self.api_key:
            return _gemini_rest_call(self.model_name, prompt, self.api_key)

        return "[Analysis unavailable – no LLM client or API key]"

    def _run_backtest(
        self,
        symbol: str,
        hypothesis: str,
        price_data: Optional[pd.DataFrame],
    ) -> dict:
        """Simple event-driven backtest from hypothesis string.

        Parses indicators mentioned in the hypothesis and applies a basic
        strategy. Returns Sharpe, total return, max drawdown, win rate.
        """
        if price_data is None or price_data.empty or "close" not in price_data.columns:
            return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        close = price_data["close"].copy()
        returns = close.pct_change().dropna()

        # Parse simple indicators from hypothesis
        rsi_match = re.search(r"RSI\s*[<>]?\s*(\d+(?:\.\d+)?)", hypothesis, re.I)
        mom_match = re.search(r"momentum.*?(\d+)\s*d", hypothesis, re.I)

        if rsi_match:
            period = 14
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            threshold = float(rsi_match.group(1))
            if "<" in hypothesis:
                signal = (rsi < threshold).astype(int)
            else:
                signal = (rsi > threshold).astype(int)
            strat_ret = signal.shift(1) * returns
        elif mom_match:
            period = int(mom_match.group(1))
            if len(close) > period:
                mom = close / close.shift(period) - 1
                signal = (mom > 0).astype(int)
                strat_ret = signal.shift(1) * returns
            else:
                strat_ret = returns * 0
        else:
            # Default: simple momentum (21-day)
            if len(close) > 21:
                mom = close / close.shift(21) - 1
                signal = (mom > 0).astype(int)
                strat_ret = signal.shift(1) * returns
            else:
                strat_ret = returns * 0

        strat_ret = strat_ret.dropna()

        if strat_ret.empty or strat_ret.std() == 0:
            return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        sharpe = (strat_ret.mean() / strat_ret.std()) * math.sqrt(252)
        total_return = (1 + strat_ret).prod() - 1
        cumulative = (1 + strat_ret).cumprod()
        max_dd = float((cumulative / cumulative.cummax() - 1).min())
        win_rate = float((strat_ret > 0).sum() / len(strat_ret))

        return {
            "sharpe": float(sharpe),
            "total_return": float(total_return),
            "max_drawdown": abs(max_dd),
            "win_rate": win_rate,
        }

    def _run_walk_forward(
        self,
        symbol: str,
        price_data: pd.DataFrame,
    ) -> dict:
        """Run walk-forward validation using momentum strategy as proxy."""
        def momentum_strategy(df: pd.DataFrame) -> pd.Series:
            if "close" not in df.columns:
                return pd.Series(0, index=df.index)
            close = df["close"]
            if len(close) < 21:
                return pd.Series(0.0, index=df.index)
            mom = close / close.shift(21) - 1
            signal = (mom > 0).astype(int)
            ret = close.pct_change().fillna(0)
            return signal.shift(1) * ret

        validator = WalkForwardValidator()
        return validator.validate(momentum_strategy, price_data)

    def _compute_verdict(
        self,
        backtest_results: dict,
        wf_results: dict,
        data_analysis: str,
    ) -> tuple[str, float]:
        """Derive research verdict and confidence from results."""
        sharpe = backtest_results.get("sharpe", 0.0)
        wf_sharpe = wf_results.get("mean_test_sharpe", sharpe)
        pbo = wf_results.get("pbo", 1.0)

        # Confidence based on consistency
        confidence = 0.5
        if pbo < 0.4 and wf_sharpe > 0.5:
            confidence = 0.9
        elif pbo < 0.5 and wf_sharpe > 0.3:
            confidence = 0.7
        elif sharpe > 1.0:
            confidence = 0.6

        # Verdict
        if confidence >= 0.7 and wf_sharpe > 0.5:
            verdict = "accept"
        elif confidence >= 0.5 and wf_sharpe > 0.3:
            verdict = "accept"
        elif "API error" in data_analysis or "unavailable" in data_analysis:
            verdict = "needs_more_data"
        else:
            verdict = "reject"

        return verdict, min(max(confidence, 0.0), 1.0)
