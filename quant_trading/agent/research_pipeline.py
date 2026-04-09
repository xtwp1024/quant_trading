"""Research Pipeline — orchestrates the full AgentQuant research workflow.

Absorbed from D:/Hive/Data/trading_repos/AgentQuant/

Stages:
    1. Data ingestion (yfinance / parquet cache)
    2. Feature engineering (technical indicators)
    3. Regime detection
    4. Hypothesis generation (LLM)
    5. Backtesting
    6. Walk-Forward validation
    7. Ablation study
    8. Result aggregation

Classes:
    ResearchPipeline -- End-to-end pipeline runner.
    PipelineStage    -- Single stage descriptor.
    PipelineReport   -- Aggregated output report.

Bilingual docstrings (English primary, Chinese comments).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from quant_trading.agent.agent_quant import (
    AgentQuant,
    ResearchResult,
    ResearchTask,
    WalkForwardValidator,
)

__all__ = [
    "ResearchPipeline",
    "PipelineStage",
    "PipelineReport",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """Descriptor for a single pipeline stage.

    Attributes:
        name: Unique stage identifier.
        description: Human-readable description.
        fn: Callable that executes the stage.
        required: Whether the pipeline should abort if this stage fails.
    """
    name: str
    description: str
    fn: Callable[..., Any]
    required: bool = True


@dataclass
class PipelineReport:
    """Aggregated output from a full pipeline run.

    Attributes:
        task: The original ResearchTask.
        stages: List of stage results in execution order.
        final_result: Final ResearchResult from AgentQuant.
        regime_detected: Detected market regime.
        walk_forward: Walk-forward validation summary.
        ablation: Ablation study results, or empty dict.
        started_at: Pipeline start timestamp.
        finished_at: Pipeline finish timestamp.
        errors: List of error messages encountered.
    """
    task: ResearchTask
    stages: list[dict[str, Any]] = field(default_factory=list)
    final_result: Optional[ResearchResult] = None
    regime_detected: str = "unknown"
    walk_forward: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def summary(self) -> dict:
        """Return a JSON-serializable summary dict."""
        return {
            "symbol": self.task.symbol,
            "hypothesis": self.task.hypothesis,
            "regime": self.regime_detected,
            "verdict": (
                self.final_result.verdict
                if self.final_result else "unknown"
            ),
            "confidence": (
                self.final_result.confidence
                if self.final_result else 0.0
            ),
            "walk_forward_score": (
                self.final_result.walk_forward_score
                if self.final_result else None
            ),
            "n_stages": len(self.stages),
            "duration_s": self.duration_seconds,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ResearchPipeline:
    """End-to-end quantitative research pipeline.

    Orchestrates data ingestion, feature engineering, regime detection,
    LLM-powered hypothesis generation, backtesting, walk-forward
    validation, and optional ablation studies.

    Example:
        pipeline = ResearchPipeline(symbol="SPY")
        pipeline.add_data_stage(fetch_func=my_fetch)
        pipeline.add_feature_stage(compute_func=my_compute_features)
        pipeline.add_agent_stage(api_key="...")
        report = pipeline.run()
        print(report.summary())

    Parameters:
        symbol: Ticker symbol for the research.
        hypothesis: Optional initial hypothesis to test.
        train_window: Trading days for walk-forward train window (default 252).
        test_window:  Trading days for walk-forward test window  (default 63).
    """

    def __init__(
        self,
        symbol: str,
        hypothesis: Optional[str] = None,
        train_window: int = 252,
        test_window: int = 63,
    ) -> None:
        self.symbol = symbol
        self.hypothesis = hypothesis
        self.train_window = train_window
        self.test_window = test_window
        self._stages: list[PipelineStage] = []
        self._data_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage builders
    # ------------------------------------------------------------------

    def add_data_stage(
        self,
        fetch_func: Callable[[str], pd.DataFrame],
        name: str = "data_ingestion",
        description: str = "Fetch and cache price data",
    ) -> "ResearchPipeline":
        """Add a data-ingestion stage.

        Args:
            fetch_func: Callable accepting a symbol and returning a DataFrame
                with at least a 'close' column indexed by date.
            name: Stage identifier.
            description: Human-readable description.

        Returns:
            self (for chaining).
        """
        def stage_fn() -> pd.DataFrame:
            df = fetch_func(self.symbol)
            self._data_cache["price_data"] = df
            return df

        self._stages.append(
            PipelineStage(name=name, description=description, fn=stage_fn)
        )
        return self

    def add_feature_stage(
        self,
        compute_func: Callable[[pd.DataFrame], pd.DataFrame],
        name: str = "feature_engineering",
        description: str = "Compute technical indicators and features",
    ) -> "ResearchPipeline":
        """Add a feature-engineering stage.

        Args:
            compute_func: Callable accepting a price DataFrame and returning
                a features DataFrame.
            name: Stage identifier.
            description: Human-readable description.

        Returns:
            self (for chaining).
        """
        def stage_fn() -> pd.DataFrame:
            price_data = self._data_cache.get("price_data")
            if price_data is None:
                raise RuntimeError("data_ingestion stage must run before feature_engineering")
            features = compute_func(price_data)
            self._data_cache["features"] = features
            return features

        self._stages.append(
            PipelineStage(name=name, description=description, fn=stage_fn)
        )
        return self

    def add_agent_stage(
        self,
        name: str = "agent_research",
        agent: Optional[AgentQuant] = None,
        model_name: str = "gemini-2.0-flash",
        api_key: str = "",
        llm_client: Any = None,
        description: str = "Run AgentQuant research",
    ) -> "ResearchPipeline":
        """Add the main AgentQuant research stage.

        Args:
            name: Stage identifier.
            agent: Optional pre-constructed AgentQuant instance.
            model_name: Gemini model name (used if agent is None).
            api_key: API key (used if agent is None).
            llm_client: Optional LLM client (used if agent is None).
            description: Human-readable description.

        Returns:
            self (for chaining).
        """
        def stage_fn() -> ResearchResult:
            if agent is None:
                _agent = AgentQuant(
                    model_name=model_name,
                    api_key=api_key or os.environ.get("GOOGLE_API_KEY", ""),
                    llm_client=llm_client,
                )
            else:
                _agent = agent

            price_data = self._data_cache.get("price_data")
            features   = self._data_cache.get("features")

            return _agent.research(
                symbol=self.symbol,
                hypothesis=self.hypothesis,
                price_data=price_data,
                features=features,
            )

        self._stages.append(
            PipelineStage(name=name, description=description, fn=stage_fn)
        )
        return self

    def add_walkforward_stage(
        self,
        name: str = "walk_forward",
        strategy_func: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        description: str = "Walk-forward validation",
    ) -> "ResearchPipeline":
        """Add a walk-forward validation stage.

        Args:
            name: Stage identifier.
            strategy_func: Callable that takes a DataFrame and returns
                a returns series. Uses a default momentum strategy if None.
            description: Human-readable description.

        Returns:
            self (for chaining).
        """
        def default_momentum(df: pd.DataFrame) -> pd.Series:
            if "close" not in df.columns:
                return pd.Series(0.0, index=df.index)
            close = df["close"]
            if len(close) < 21:
                return pd.Series(0.0, index=df.index)
            mom = close / close.shift(21) - 1
            signal = (mom > 0).astype(int)
            ret = close.pct_change().fillna(0)
            return signal.shift(1) * ret

        _strategy_func = strategy_func or default_momentum

        def stage_fn() -> dict:
            price_data = self._data_cache.get("price_data")
            if price_data is None:
                raise RuntimeError("data_ingestion stage must run before walk_forward")
            validator = WalkForwardValidator(
                train_window=self.train_window,
                test_window=self.test_window,
            )
            return validator.validate(_strategy_func, price_data)

        self._stages.append(
            PipelineStage(name=name, description=description, fn=stage_fn)
        )
        return self

    def add_ablation_stage(
        self,
        name: str = "ablation",
        base_strategy: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        remove_components: Optional[list[str]] = None,
        description: str = "Ablation study",
    ) -> "ResearchPipeline":
        """Add an ablation study stage.

        Args:
            name: Stage identifier.
            base_strategy: Strategy function to ablate. Uses default
                momentum if None.
            remove_components: List of component names to test removing.
            description: Human-readable description.

        Returns:
            self (for chaining).
        """
        def default_momentum(df: pd.DataFrame) -> pd.Series:
            if "close" not in df.columns:
                return pd.Series(0.0, index=df.index)
            close = df["close"]
            if len(close) < 21:
                return pd.Series(0.0, index=df.index)
            mom = close / close.shift(21) - 1
            signal = (mom > 0).astype(int)
            ret = close.pct_change().fillna(0)
            return signal.shift(1) * ret

        _base = base_strategy or default_momentum
        _remove = remove_components or []

        def stage_fn() -> dict:
            price_data = self._data_cache.get("price_data")
            if price_data is None:
                raise RuntimeError("data_ingestion stage must run before ablation")
            agent = AgentQuant()
            return agent.run_ablation(_base, _remove, price_data)

        self._stages.append(
            PipelineStage(name=name, description=description, fn=stage_fn)
        )
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        hypothesis: Optional[str] = None,
        abort_on_error: bool = False,
    ) -> PipelineReport:
        """Execute all pipeline stages in order.

        Args:
            hypothesis: Override the hypothesis for this run.
            abort_on_error: If True, raise on the first stage error.

        Returns:
            PipelineReport with results from all stages.
        """
        if hypothesis:
            self.hypothesis = hypothesis

        task = ResearchTask(
            symbol=self.symbol,
            hypothesis=self.hypothesis or "",
            regime="unknown",
        )

        report = PipelineReport(
            task=task,
            started_at=datetime.now(),
        )

        for stage in self._stages:
            stage_started = datetime.now()
            try:
                result = stage.fn()
                duration = (datetime.now() - stage_started).total_seconds()
                report.stages.append({
                    "name": stage.name,
                    "description": stage.description,
                    "status": "success",
                    "duration_s": duration,
                    "result": _safe_serialize(result),
                })

                # Propagate known results into cache
                if isinstance(result, ResearchResult):
                    report.final_result = result
                    report.regime_detected = result.regime
                    if result.walk_forward_score is not None:
                        report.walk_forward = {"score": result.walk_forward_score}

            except Exception as exc:
                duration = (datetime.now() - stage_started).total_seconds()
                error_msg = f"[{stage.name}] {type(exc).__name__}: {exc}"
                report.errors.append(error_msg)
                report.stages.append({
                    "name": stage.name,
                    "description": stage.description,
                    "status": "error",
                    "duration_s": duration,
                    "error": error_msg,
                })
                if stage.required and abort_on_error:
                    raise

        report.finished_at = datetime.now()
        return report


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_serialize(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form."""
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return {"columns": obj.columns.tolist(), "index": obj.index.tolist(), "data": obj.values.tolist()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj
