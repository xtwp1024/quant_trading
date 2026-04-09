"""
AlphaAgent — Multi-agent alpha factor system with ensemble prediction.

Architecture
------------
- CommunicationManager : thread-safe agent间通信协议 (publish/subscribe/direct)
- DataAgent            : fetches and normalises OHLCV + auxiliary data
- PredictionAgent      : LSTM / XGBoost / Prophet ensemble forecasts
- SignalAgent         : computes 101 formulaic alphas, selects best subset
- SentimentAgent      : news-sentiment signal integration
- RiskAgent           : portfolio-level risk overlays
- AlphaAgent          : top-level orchestrator coordinating all agents

Key capabilities
----------------
- 101 formulaic alphas from Kakushadze (2016) via Alpha101
- IC / turnover / half-life factor evaluation via AlphaEvaluator
- Multi-model ensemble: LSTM + XGBoost + Prophet
- Dynamic factor selection (greedy low-correlation subset)
- Asynchronous message-passing between agents
- Optional Prophet integration for time-series decomposition

Requires (all optional): torch, xgboost, prophet, scikit-learn, ta-lib
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

__all__ = [
    # Multi-agent system (existing)
    "CommunicationManager",
    "DataAgent",
    "PredictionAgent",
    "SignalAgent",
    "SentimentAgent",
    "RiskAgent",
    "AlphaAgent",
    "ProphetPredictor",
    # AlphaFactor system (new — absorbed from alpha-agent repo)
    "AlphaFactor",
    "Alpha101Bundle",
    "FactorCategory",
    "MarketRegime",
    # Key alpha factor classes
    "Alpha001", "Alpha002", "Alpha003", "Alpha004",
    "Alpha006", "Alpha007", "Alpha008", "Alpha009", "Alpha010",
    "Alpha012", "Alpha013", "Alpha014", "Alpha015", "Alpha016",
    "Alpha017", "Alpha018", "Alpha019", "Alpha020", "Alpha021",
    "Alpha023", "Alpha024", "Alpha026", "Alpha027", "Alpha028",
    "Alpha029", "Alpha031", "Alpha032", "Alpha033", "Alpha036",
    "Alpha037", "Alpha039", "Alpha040", "Alpha044", "Alpha046",
    "Alpha051", "Alpha056", "Alpha057", "Alpha059", "Alpha060",
    "Alpha071", "Alpha074", "Alpha101",
    # Utility functions
    "neutralize_zscore",
    "neutralize_rank",
    "compute_ic",
    "compute_ir",
    "compute_rolling_ic",
]

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# CommunicationManager — adapted from alpha-agent unified_communication
# ----------------------------------------------------------------------


class Message:
    """Lightweight message object for agent communication."""

    def __init__(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 1,
    ):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type  # REQUEST | RESPONSE | DATA | BROADCAST
        self.content = content
        self.metadata = metadata or {}
        self.priority = priority
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "metadata": self.metadata,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
        }


class CommunicationManager:
    """
    Thread-safe publish/subscribe + direct-message communication bus.

    Agents register themselves; they can then publish to topics or send
    direct messages.  All operations are thread-safe via RLock.
    """

    def __init__(self):
        self._agents: Dict[str, object] = {}
        self._topics: Dict[str, List[object]] = {}
        self._queues: Dict[str, queue.PriorityQueue] = {}
        self._lock = threading.RLock()
        self._running = False
        self._message_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._shutdown.clear()
            self._message_thread = threading.Thread(target=self._process_loop, daemon=True)
            self._message_thread.start()
            logger.info("CommunicationManager started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._shutdown.set()
            if self._message_thread:
                self._message_thread.join(timeout=5.0)
            logger.info("CommunicationManager stopped")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, agent: object) -> bool:
        with self._lock:
            if agent_id in self._agents:
                logger.warning("Agent %s already registered", agent_id)
                return False
            self._agents[agent_id] = agent
            self._queues[agent_id] = queue.PriorityQueue()
            logger.info("Agent registered: %s", agent_id)
            return True

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False
            del self._agents[agent_id]
            del self._queues[agent_id]
            for topic_list in self._topics.values():
                topic_list[:] = [a for a in topic_list if a != agent]
            logger.info("Agent unregistered: %s", agent_id)
            return True

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def create_topic(self, topic: str) -> None:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []

    def subscribe(self, agent_id: str, topic: str) -> bool:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []
            agent = self._agents.get(agent_id)
            if agent and agent not in self._topics[topic]:
                self._topics[topic].append(agent)
                logger.debug("Agent %s subscribed to topic %s", agent_id, topic)
            return True

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        with self._lock:
            if topic not in self._topics:
                return False
            agent = self._agents.get(agent_id)
            if agent in self._topics[topic]:
                self._topics[topic].remove(agent)
            return True

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def publish(self, topic: str, message: Message) -> None:
        """Publish a message to all subscribers of a topic."""
        with self._lock:
            subscribers = list(self._topics.get(topic, []))
        for agent in subscribers:
            agent_id = self._resolve_agent_id(agent)
            if agent_id:
                self._queues[agent_id].put((message.priority, message))
                logger.debug("Published to %s <- %s", agent_id, topic)

    def send_message(self, message: Message) -> bool:
        """Direct message to a specific agent."""
        receiver = message.receiver_id
        with self._lock:
            if receiver not in self._agents:
                logger.warning("Message to unknown agent: %s", receiver)
                return False
        self._queues[receiver].put((message.priority, message))
        logger.debug("Direct message %s -> %s", message.sender_id, receiver)
        return True

    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """Blocking receive for an agent."""
        q = self._queues.get(agent_id)
        if not q:
            return None
        try:
            _, msg = q.get(block=True, timeout=timeout)
            return msg
        except queue.Empty:
            return None

    def receive_all(self, agent_id: str) -> List[Message]:
        """Drain all pending messages for an agent."""
        messages = []
        q = self._queues.get(agent_id)
        if not q:
            return messages
        while True:
            try:
                _, msg = q.get_nowait()
                messages.append(msg)
            except queue.Empty:
                break
        return messages

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_agent_id(self, agent: object) -> Optional[str]:
        for aid, a in self._agents.items():
            if a is agent:
                return aid
        return None

    def _process_loop(self) -> None:
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=0.1)
            with self._lock:
                topics = list(self._topics.keys())
            for topic in topics:
                # Allow other threads to interleave
                if self._shutdown.is_set():
                    break


# ----------------------------------------------------------------------
# Prophet predictor
# ----------------------------------------------------------------------


def _has_prophet() -> bool:
    try:
        from prophet import Prophet  # noqa: F401
        return True
    except ImportError:
        return False


class ProphetPredictor:
    """
    Facebook/Meta Prophet wrapper for time-series trend + seasonality forecasting.

    Uses the additive model (trend + weekly + yearly + optional daily seasonality)
    to forecast future prices or returns.

    Parameters
    ----------
    periods       : forecast horizon in data frequency units (default 5)
    weekly_seasonality  : enable weekly seasonality (default True)
    yearly_seasonality  : enable yearly seasonality (default True)
    daily_seasonality   : enable daily seasonality (default False)
    changepoint_prior_scale : trend flexibility (default 0.05)
    """

    def __init__(
        self,
        periods: int = 5,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ):
        self.periods = periods
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model = None
        self._trained = False
        self._ds_col = "ds"
        self._y_col = "y"

    def train(self, df: pd.DataFrame, target_col: str = "close") -> Dict[str, Any]:
        """
        Train Prophet on df (must have a DatetimeIndex or 'date' column).

        Returns training summary dict.
        """
        if not _has_prophet():
            logger.warning("prophet not installed — ProphetPredictor unavailable")
            return {}

        from prophet import Prophet

        ts = df.copy()
        if isinstance(ts.index, pd.DatetimeIndex):
            ts = ts.reset_index()
        col_date = "date" if "date" in ts.columns else ts.columns[0]
        ts = ts[[col_date, target_col]].rename(columns={col_date: self._ds_col, target_col: self._y_col})
        ts = ts.dropna()

        self._model = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        self._model.fit(ts)
        self._trained = True
        logger.info("Prophet model trained on %d rows", len(ts))
        return {"status": "trained", "n_samples": len(ts)}

    def predict(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate future forecasts.

        If df is None, forecasts `periods` steps ahead from training data.
        Returns DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
        """
        if not self._trained or self._model is None:
            logger.warning("Prophet model not trained — returning empty")
            return pd.DataFrame()

        from prophet import Prophet

        if df is not None:
            # Refit on new data
            ts = df.copy()
            if isinstance(ts.index, pd.DatetimeIndex):
                ts = ts.reset_index()
            col_date = "date" if "date" in ts.columns else ts.columns[0]
            target = "close" if "close" in ts.columns else ts.columns[-1]
            ts = ts[[col_date, target]].rename(columns={col_date: self._ds_col, target: self._y_col})
            ts = ts.dropna()
            self._model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
            )
            self._model.fit(ts)

        future = self._model.make_future_dataframe(periods=self.periods)
        forecast = self._model.predict(future)
        cols = [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper"] if c in forecast.columns]
        return forecast[cols]


# ----------------------------------------------------------------------
# Agent base class
# ----------------------------------------------------------------------


class BaseAgent:
    """Base class shared by all agents."""

    def __init__(self, agent_id: str, comm: Optional[CommunicationManager] = None):
        self.agent_id = agent_id
        self.comm = comm
        if comm:
            comm.register_agent(agent_id, self)

    def send(self, receiver: str, content: Any, msg_type: str = "REQUEST",
             metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.comm:
            return False
        msg = Message(self.agent_id, receiver, msg_type, content, metadata)
        return self.comm.send_message(msg)

    def broadcast(self, topic: str, content: Any, msg_type: str = "DATA",
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.comm:
            return
        msg = Message(self.agent_id, "", msg_type, content, metadata)
        self.comm.publish(topic, msg)

    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        if not self.comm:
            return None
        return self.comm.receive(self.agent_id, timeout)

    def receive_all(self) -> List[Message]:
        if not self.comm:
            return []
        return self.comm.receive_all(self.agent_id)


# ----------------------------------------------------------------------
# DataAgent
# ----------------------------------------------------------------------


class DataAgent(BaseAgent):
    """
    Ingestion and normalisation of OHLCV data.

    Responsibilities
    -----------------
    - Fetch raw OHLCV (interface: data_fetcher callable)
    - Compute derived columns: returns, vwap, adv{d}, cap proxy
    - Publish normalised panel to 'market_data' topic
    - Respond to data requests from other agents
    """

    def __init__(
        self,
        comm: Optional[CommunicationManager] = None,
        data_fetcher: Optional[Callable[[str, int], pd.DataFrame]] = None,
    ):
        super().__init__("DataAgent", comm)
        self.data_fetcher = data_fetcher
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch(
        self,
        ticker: str,
        days: int = 252,
        data_fetcher: Optional[Callable[[str, int], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Fetch and normalise data for a ticker.

        data_fetcher: callable(ticker: str, days: int) -> DataFrame with OHLCV cols.
                      If None, uses the instance-level fetcher.
        """
        fetcher = data_fetcher or self.data_fetcher
        if fetcher:
            raw = fetcher(ticker, days)
        else:
            # Stub: generate synthetic data for testing
            raw = self._synthetic_data(ticker, days)

        df = self._normalise(raw)
        self._cache[ticker] = df
        return df

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard columns exist and add derived fields."""
        df = df.copy()
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Data missing columns: {missing}")

        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
        if "vwap" not in df.columns:
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
        if "cap" not in df.columns:
            df["cap"] = df["close"]  # proxy
        for d in (5, 10, 20, 30, 60, 120):
            col = f"adv{d}"
            if col not in df.columns:
                df[col] = df["volume"].rolling(window=d, min_periods=1).mean()
        return df.dropna()

    def _synthetic_data(self, ticker: str, days: int) -> pd.DataFrame:
        """Generate random walk OHLCV for testing."""
        np.random.seed(hash(ticker) % (2**31))
        dates = pd.bdate_range(end=datetime.now(), periods=days)
        close = 100 * np.exp(np.cumsum(np.random.randn(days) * 0.02))
        high = close * (1 + np.abs(np.random.randn(days) * 0.01))
        low = close * (1 - np.abs(np.random.randn(days) * 0.01))
        open_price = low + np.random.rand(days) * (high - low)
        volume = np.random.randint(1_000_000, 10_000_000, days)
        return pd.DataFrame(
            {"open": open_price, "high": high, "low": low,
             "close": close, "volume": volume},
            index=dates,
        )

    def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data requests from other agents."""
        ticker = content.get("ticker")
        days = content.get("days", 252)
        if not ticker:
            return {"error": "ticker required"}
        df = self.fetch(ticker, days)
        return {"ticker": ticker, "data": df.to_dict()}


# ----------------------------------------------------------------------
# PredictionAgent — LSTM / XGBoost / Prophet ensemble
# ----------------------------------------------------------------------


class PredictionAgent(BaseAgent):
    """
    Multi-model prediction agent wrapping LSTMPredictor, XGBoostPredictor,
    and ProphetPredictor.

    Each model produces a next-period return forecast.  The final prediction
    is a simple weighted average of available model outputs.

    Parameters
    ----------
    comm           : CommunicationManager instance
    lstm_weight    : weight for LSTM prediction (default 0.35)
    xgb_weight     : weight for XGBoost prediction (default 0.35)
    prophet_weight : weight for Prophet prediction (default 0.30)
    seq_len        : LSTM sequence length (default 20)
    """

    def __init__(
        self,
        comm: Optional[CommunicationManager] = None,
        lstm_weight: float = 0.35,
        xgb_weight: float = 0.35,
        prophet_weight: float = 0.30,
        seq_len: int = 20,
    ):
        super().__init__("PredictionAgent", comm)
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.prophet_weight = prophet_weight
        self.seq_len = seq_len

        self._lstm = None
        self._xgb = None
        self._prophet = None
        self._lstm_trained = False
        self._xgb_trained = False
        self._prophet_trained = False
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_lstm(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train LSTM on df."""
        try:
            from quant_trading.factors.lstm_predictor import LSTMPredictor
        except ImportError:
            logger.warning("LSTMPredictor not available")
            return {}

        if feature_cols:
            self._feature_cols = feature_cols
        else:
            ohlcv = {"open", "high", "low", "close", "volume", "returns"}
            self._feature_cols = [c for c in df.columns if c not in ("date", "symbol")]

        self._lstm = LSTMPredictor(seq_len=self.seq_len, feature_cols=self._feature_cols)
        result = self._lstm.train(df)
        self._lstm_trained = self._lstm._trained
        return result

    def train_xgb(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train XGBoost on df."""
        try:
            from quant_trading.factors.lstm_predictor import XGBoostPredictor
        except ImportError:
            logger.warning("XGBoostPredictor not available")
            return {}

        self._xgb = XGBoostPredictor(feature_cols=feature_cols or self._feature_cols)
        result = self._xgb.train(df)
        self._xgb_trained = self._xgb._trained
        return result

    def train_prophet(self, df: pd.DataFrame, target_col: str = "close") -> Dict[str, Any]:
        """Train Prophet on df."""
        if not _has_prophet():
            return {}
        self._prophet = ProphetPredictor()
        return self._prophet.train(df, target_col=target_col)

    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all available models."""
        results = {}
        results["lstm"] = self.train_lstm(df)
        results["xgb"] = self.train_xgb(df)
        results["prophet"] = self.train_prophet(df)
        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_lstm(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if not self._lstm_trained or self._lstm is None:
            return None
        result = self._lstm.predict(df)
        return result["pred"] if "pred" in result.columns else None

    def predict_xgb(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if not self._xgb_trained or self._xgb is None:
            return None
        result = self._xgb.predict(df)
        return result["pred"] if "pred" in result.columns else None

    def predict_prophet(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if not self._prophet_trained or self._prophet is None:
            return None
        forecast = self._prophet.predict(df)
        if "yhat" in forecast.columns:
            # Return pct change from last known close
            last_close = df["close"].iloc[-1]
            future = forecast["yhat"].iloc[-1]
            return pd.Series([(future - last_close) / last_close], index=[df.index[-1]])
        return None

    def predict_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a weighted ensemble prediction.

        Returns DataFrame with columns: lstm_pred, xgb_pred, prophet_pred, ensemble_pred
        and a datetime index.
        """
        tick = df.index[-1]
        lstm_pred = self.predict_lstm(df)
        xgb_pred = self.predict_xgb(df)
        prophet_pred = self.predict_prophet(df)

        row = {"ticker": getattr(df, "ticker", None)}

        w_total = 0.0
        weighted_sum = 0.0

        if lstm_pred is not None and len(lstm_pred) > 0:
            row["lstm_pred"] = float(lstm_pred.iloc[-1])
            w_total += self.lstm_weight
            weighted_sum += self.lstm_weight * float(lstm_pred.iloc[-1])
        else:
            row["lstm_pred"] = None

        if xgb_pred is not None and len(xgb_pred) > 0:
            row["xgb_pred"] = float(xgb_pred.iloc[-1])
            w_total += self.xgb_weight
            weighted_sum += self.xgb_weight * float(xgb_pred.iloc[-1])
        else:
            row["xgb_pred"] = None

        if prophet_pred is not None and len(prophet_pred) > 0:
            row["prophet_pred"] = float(prophet_pred.iloc[-1])
            w_total += self.prophet_weight
            weighted_sum += self.prophet_weight * float(prophet_pred.iloc[-1])
        else:
            row["prophet_pred"] = None

        row["ensemble_pred"] = weighted_sum / w_total if w_total > 0 else 0.0
        row["timestamp"] = datetime.now().isoformat()

        return pd.DataFrame([row], index=[tick])

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction requests."""
        action = content.get("action")
        if action == "train":
            df = pd.DataFrame(content.get("data", []))
            if df.empty:
                return {"error": "no data"}
            return self.train_ensemble(df)
        elif action == "predict":
            df = pd.DataFrame(content.get("data", []))
            if df.empty:
                return {"error": "no data"}
            pred_df = self.predict_ensemble(df)
            return {"predictions": pred_df.to_dict(orient="records")}
        return {"error": f"unknown action: {action}"}


# ----------------------------------------------------------------------
# SignalAgent — alpha computation + dynamic factor selection
# ----------------------------------------------------------------------


class SignalAgent(BaseAgent):
    """
    Compute formulaic alphas and dynamically select the best subset.

    Wraps Alpha101 and AlphaEvaluator from the factors library.
    """

    def __init__(
        self,
        comm: Optional[CommunicationManager] = None,
        min_ic: float = 0.02,
        max_corr: float = 0.7,
        min_half_life: float = 2.0,
    ):
        super().__init__("SignalAgent", comm)
        self.min_ic = min_ic
        self.max_corr = max_corr
        self.min_half_life = min_half_life
        self._alpha101 = None
        self._evaluator = None
        self._selected_factors: List[str] = []

    def _ensure_alpha101(self):
        if self._alpha101 is None:
            try:
                from quant_trading.factors.alpha_101 import Alpha101
                self._alpha101 = Alpha101()
            except ImportError:
                logger.error("Alpha101 not found — install quant_trading.factors.alpha_101")
                raise

    def _ensure_evaluator(self):
        if self._evaluator is None:
            try:
                from quant_trading.factors.alpha_evaluator import AlphaEvaluator
                self._evaluator = AlphaEvaluator
            except ImportError:
                logger.error("AlphaEvaluator not found")
                raise

    def compute_alphas(
        self,
        df: pd.DataFrame,
        alpha_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute all (or selected) formulaic alphas on df.

        df must have OHLCV columns.
        Returns df with new alpha columns added.
        """
        self._ensure_alpha101()
        df = df.copy()
        df = self._alpha101.compute(df, names=alpha_names)
        return df

    def evaluate_factors(
        self,
        df: pd.DataFrame,
        alpha_names: Optional[List[str]] = None,
        forward_periods: List[int] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate alpha quality via IC, turnover, and half-life.

        Returns evaluation report dict keyed by alpha name.
        """
        self._ensure_evaluator()
        if forward_periods is None:
            forward_periods = [1, 5, 10, 20]
        if alpha_names is None:
            # Detect alpha columns (all that start with 'alpha_')
            alpha_names = [c for c in df.columns if c.startswith("alpha_")]
        return self._evaluator.evaluate(df, alpha_names, forward_periods)

    def select_factors(
        self,
        df: pd.DataFrame,
        forward_periods: List[int] = None,
    ) -> List[str]:
        """
        Compute + evaluate + greedily select optimal non-correlated factors.

        Returns list of selected factor names.
        """
        self._ensure_alpha101()
        self._ensure_evaluator()
        if forward_periods is None:
            forward_periods = [1, 5, 10, 20]

        # Compute all alphas
        df_with_alphas = self.compute_alphas(df)
        alpha_names = [c for c in df_with_alphas.columns if c.startswith("alpha_")]

        # Evaluate
        report = self._evaluator.evaluate(df_with_alphas, alpha_names, forward_periods)

        # Select
        selected = self._evaluator.select_optimal_factors(
            report,
            min_ic=self.min_ic,
            max_corr=self.max_corr,
            min_half_life=self.min_half_life,
        )
        self._selected_factors = selected
        return selected

    @property
    def selected_factors(self) -> List[str]:
        return self._selected_factors

    def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle signal requests."""
        action = content.get("action")
        df = pd.DataFrame(content.get("data", []))

        if df.empty:
            return {"error": "no data"}

        if action == "compute":
            names = content.get("alpha_names")
            result = self.compute_alphas(df, alpha_names=names)
            return {"alphas": result.to_dict()}
        elif action == "evaluate":
            names = content.get("alpha_names")
            report = self.evaluate_factors(df, alpha_names=names)
            return {"report": report}
        elif action == "select":
            selected = self.select_factors(df)
            return {"selected_factors": selected}
        return {"error": f"unknown action: {action}"}


# ----------------------------------------------------------------------
# SentimentAgent — thin wrapper around factors.sentiment_agent.SentimentAgent
# ----------------------------------------------------------------------


class SentimentAgent(BaseAgent):
    """
    Sentiment signal agent.

    Wraps quant_trading.factors.sentiment_agent.SentimentAgent and publishes
    sentiment scores to the 'sentiment' topic.
    """

    def __init__(
        self,
        comm: Optional[CommunicationManager] = None,
        api_key: Optional[str] = None,
        news_fetcher: Optional[Callable[[str, int], List[Dict]]] = None,
    ):
        super().__init__("SentimentAgent", comm)
        try:
            from quant_trading.factors.sentiment_agent import SentimentAgent as SAS
            self._agent = SAS(api_key=api_key, news_fetcher=news_fetcher)
        except ImportError:
            logger.warning("SentimentAgent fallback — using built-in")
            self._agent = None

    def analyze(self, ticker: str, force_update: bool = False) -> Dict[str, Any]:
        if self._agent:
            return self._agent.analyze(ticker, force_update=force_update)
        return {"ticker": ticker, "sentiment_score": 0.0, "sentiment_label": "neutral"}

    def analyze_batch(self, tickers: List[str]) -> Dict[str, Any]:
        if self._agent:
            return self._agent.analyze_batch(tickers)
        return {t: {"ticker": t, "sentiment_score": 0.0} for t in tickers}

    def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        ticker = content.get("ticker")
        if not ticker:
            tickers = content.get("tickers", [])
            return self.analyze_batch(tickers)
        return self.analyze(ticker)


# ----------------------------------------------------------------------
# RiskAgent — portfolio-level risk overlays
# ----------------------------------------------------------------------


class RiskAgent(BaseAgent):
    """
    Portfolio risk management agent.

    Responsibilities
    ----------------
    - Compute rolling volatility, drawdown, and VaR estimates
    - Respond to risk limit checks (position size, drawdown budget)
    - Publish risk metrics to 'risk' topic
    """

    def __init__(
        self,
        comm: Optional[CommunicationManager] = None,
        vol_window: int = 20,
        var_confidence: float = 0.95,
    ):
        super().__init__("RiskAgent", comm)
        self.vol_window = vol_window
        self.var_confidence = var_confidence

    def rolling_volatility(self, returns: pd.Series, window: int = None) -> pd.Series:
        window = window or self.vol_window
        return returns.rolling(window, min_periods=window).std() * np.sqrt(252)

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Peak-to-trough drawdown in fraction terms."""
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return float(drawdown.min())

    def value_at_risk(
        self, returns: pd.Series, confidence: float = None
    ) -> float:
        confidence = confidence or self.var_confidence
        return float(returns.quantile(1 - confidence))

    def risk_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a full risk report from a returns series."""
        returns = df["returns"] if "returns" in df.columns else df["close"].pct_change()
        returns = returns.dropna()
        if len(returns) < self.vol_window:
            return {"error": "insufficient data"}

        vol = self.rolling_volatility(returns)
        equity = (1 + returns).cumprod()
        dd = self.max_drawdown(equity)
        var = self.value_at_risk(returns)
        last_vol = float(vol.iloc[-1])

        return {
            "annualised_volatility": round(last_vol, 4),
            "max_drawdown": round(dd, 4),
            "value_at_risk": round(var, 4),
            "current_risk_regime": (
                "high" if last_vol > 0.30
                else "low" if last_vol < 0.10
                else "medium"
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def check_risk_limit(
        self, position_pnl: float, current_drawdown: float, max_drawdown: float = -0.15
    ) -> Dict[str, Any]:
        """Evaluate whether a new position breaches risk limits."""
        breach = current_drawdown < max_drawdown
        return {
            "approved": not breach,
            "current_drawdown": round(current_drawdown, 4),
            "limit": max_drawdown,
            "breach": breach,
            "timestamp": datetime.now().isoformat(),
        }

    def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        action = content.get("action")
        if action == "risk_report":
            df = pd.DataFrame(content.get("data", []))
            if df.empty:
                return {"error": "no data"}
            return self.risk_report(df)
        elif action == "check_limit":
            pnl = content.get("position_pnl", 0.0)
            dd = content.get("current_drawdown", 0.0)
            limit = content.get("max_drawdown", -0.15)
            return self.check_risk_limit(pnl, dd, max_drawdown=limit)
        return {"error": f"unknown action: {action}"}


# ----------------------------------------------------------------------
# AlphaAgent — top-level orchestrator
# ----------------------------------------------------------------------


class AlphaAgent:
    """
    Top-level orchestrator coordinating DataAgent, PredictionAgent,
    SignalAgent, SentimentAgent, and RiskAgent.

    Provides a unified interface for alpha factor generation, model
    prediction, and risk management.

    Usage
    -----
        agent = AlphaAgent()
        agent.start()

        # Fetch data
        agent.fetch_data("AAPL", days=252)

        # Compute alphas and select best subset
        selected = agent.select_factors()

        # Get ensemble predictions
        predictions = agent.predict("AAPL")

        # Get risk report
        risk = agent.risk_report("AAPL")

        agent.stop()

    Parameters
    ----------
    lstm_weight, xgb_weight, prophet_weight : model weights for ensemble
    factor_min_ic        : minimum IC for factor selection
    factor_max_corr      : max correlation between selected factors
    factor_min_half_life : minimum decay half-life (days)
    vol_window           : rolling volatility window for RiskAgent
    var_confidence       : VaR confidence level
    """

    def __init__(
        self,
        lstm_weight: float = 0.35,
        xgb_weight: float = 0.35,
        prophet_weight: float = 0.30,
        factor_min_ic: float = 0.02,
        factor_max_corr: float = 0.7,
        factor_min_half_life: float = 2.0,
        vol_window: int = 20,
        var_confidence: float = 0.95,
    ):
        self.comm = CommunicationManager()

        self.data_agent = DataAgent(comm=self.comm)
        self.prediction_agent = PredictionAgent(
            comm=self.comm,
            lstm_weight=lstm_weight,
            xgb_weight=xgb_weight,
            prophet_weight=prophet_weight,
        )
        self.signal_agent = SignalAgent(
            comm=self.comm,
            min_ic=factor_min_ic,
            max_corr=factor_max_corr,
            min_half_life=factor_min_half_life,
        )
        self.sentiment_agent = SentimentAgent(comm=self.comm)
        self.risk_agent = RiskAgent(comm=self.comm, vol_window=vol_window, var_confidence=var_confidence)

        # Cached data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._selected_factors: List[str] = []
        self._prediction_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the communication manager."""
        self.comm.start()
        logger.info("AlphaAgent started")

    def stop(self) -> None:
        """Stop the communication manager."""
        self.comm.stop()
        logger.info("AlphaAgent stopped")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def fetch_data(
        self,
        ticker: str,
        days: int = 252,
        data_fetcher: Optional[Callable[[str, int], pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Fetch and cache normalised OHLCV data for a ticker.
        """
        if ticker in self._data_cache:
            return self._data_cache[ticker]
        df = self.data_agent.fetch(ticker, days=days, data_fetcher=data_fetcher)
        self._data_cache[ticker] = df
        return df

    def get_data(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._data_cache.get(ticker)

    # ------------------------------------------------------------------
    # Alphas
    # ------------------------------------------------------------------

    def compute_alphas(
        self,
        ticker: Optional[str] = None,
        alpha_names: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Compute formulaic alphas for a cached ticker.

        If ticker is None, computes on all cached tickers.
        """
        if ticker:
            df = self._data_cache.get(ticker)
            if df is None:
                return None
            return self.signal_agent.compute_alphas(df, alpha_names=alpha_names)

        results = {}
        for t, df in self._data_cache.items():
            results[t] = self.signal_agent.compute_alphas(df, alpha_names=alpha_names)
        return pd.concat(results) if results else None

    def select_factors(
        self,
        ticker: Optional[str] = None,
        forward_periods: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Compute alphas, evaluate them, and select optimal subset.

        Returns list of selected factor names.
        """
        df_with_alphas = self.compute_alphas(ticker=ticker)
        if df_with_alphas is None:
            return []

        if forward_periods is None:
            forward_periods = [1, 5, 10, 20]

        # Use first ticker if ticker is None
        if ticker is None:
            ticker = next(iter(self._data_cache.keys()), None)
        if ticker is None:
            return []

        df = df_with_alphas if isinstance(df_with_alphas, pd.DataFrame) else df_with_alphas.get(ticker)
        if df is None:
            return []

        selected = self.signal_agent.select_factors(df, forward_periods=forward_periods)
        self._selected_factors = selected
        return selected

    @property
    def selected_factors(self) -> List[str]:
        return self._selected_factors

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def train_models(self, ticker: str) -> Dict[str, Any]:
        """Train all prediction models on a ticker's cached data."""
        df = self._data_cache.get(ticker)
        if df is None:
            return {"error": f"no data for {ticker}"}
        return self.prediction_agent.train_ensemble(df)

    def predict(
        self,
        ticker: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Generate ensemble predictions.

        If ticker is None, predicts for all cached tickers.
        """
        if ticker:
            if use_cache and ticker in self._prediction_cache:
                return self._prediction_cache[ticker]
            df = self._data_cache.get(ticker)
            if df is None:
                return None
            pred = self.prediction_agent.predict_ensemble(df)
            self._prediction_cache[ticker] = pred
            return pred

        results = {}
        for t, df in self._data_cache.items():
            results[t] = self.prediction_agent.predict_ensemble(df)
        return pd.concat(results) if results else None

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    def risk_report(self, ticker: str) -> Dict[str, Any]:
        """Generate risk report for a cached ticker."""
        df = self._data_cache.get(ticker)
        if df is None:
            return {"error": f"no data for {ticker}"}
        return self.risk_agent.risk_report(df)

    def check_risk_limit(
        self,
        ticker: str,
        position_pnl: float = 0.0,
        max_drawdown: float = -0.15,
    ) -> Dict[str, Any]:
        """Check whether a position breaches risk limits."""
        report = self.risk_report(ticker)
        if "error" in report:
            return report
        dd = report.get("max_drawdown", 0.0)
        return self.risk_agent.check_risk_limit(position_pnl, dd, max_drawdown=max_drawdown)

    # ------------------------------------------------------------------
    # Sentiment
    # ------------------------------------------------------------------

    def sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get sentiment score for a ticker."""
        return self.sentiment_agent.analyze(ticker)

    def sentiment_batch(self, tickers: List[str]) -> Dict[str, Any]:
        """Get sentiment scores for multiple tickers."""
        return self.sentiment_agent.analyze_batch(tickers)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        ticker: str,
        days: int = 252,
        forward_periods: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full alpha pipeline for a ticker.

        Steps
        -----
        1. Fetch & cache data
        2. Compute all 101 alphas
        3. Evaluate & select best factors
        4. Train ensemble prediction models
        5. Generate ensemble prediction
        6. Generate risk report
        7. Fetch sentiment score

        Returns a dict with all results.
        """
        if forward_periods is None:
            forward_periods = [1, 5, 10, 20]

        result = {"ticker": ticker, "timestamp": datetime.now().isoformat()}

        # 1. Data
        df = self.fetch_data(ticker, days=days)
        result["n_bars"] = len(df)

        # 2. Alphas
        df_alphas = self.compute_alphas(ticker=ticker)
        if df_alphas is not None:
            result["n_alphas"] = len([c for c in df_alphas.columns if c.startswith("alpha_")])

        # 3. Factor selection
        selected = self.select_factors(ticker=ticker, forward_periods=forward_periods)
        result["selected_factors"] = selected

        # 4. Train models
        train_result = self.train_models(ticker)
        result["training"] = train_result

        # 5. Predictions
        pred = self.predict(ticker=ticker)
        if pred is not None:
            result["predictions"] = pred.to_dict(orient="records")

        # 6. Risk
        result["risk"] = self.risk_report(ticker)

        # 7. Sentiment
        result["sentiment"] = self.sentiment(ticker)

        return result


# =============================================================================
# AlphaFactor System — 101 Alpha Factors with Alpha101Bundle
# Absorbed from D:/Hive/Data/trading_repos/alpha-agent/
# Kakushadze "101 Formulaic Alphas" (2016) — pure NumPy + pandas, no Talib
# =============================================================================

from typing import Union, Callable, Dict, List, Optional, Tuple
from enum import Enum


# -----------------------------------------------------------------------
# Enums — factor category and market regime
# -----------------------------------------------------------------------

class FactorCategory(Enum):
    """因子类别枚举 / Factor category enumeration."""
    PRICE_RETURN = "price_return"       # 价格收益类
    PRICE_MOMENTUM = "price_momentum"   # 价格动量类
    PRICE_REVERSAL = "price_reversal"   # 价格反转类
    VOLUME_TURNOVER = "volume_turnover" # 成交量周转类
    VOLUME_RATIO = "volume_ratio"       # 成交量比率类
    VOLATILITY = "volatility"           # 波动率类
    MICROSTRUCTURE = "microstructure"   # 市场微观结构类
    CROSS_SECTIONAL = "cross_sectional" # 横截面类


class MarketRegime(Enum):
    """市场状态枚举 / Market regime enumeration."""
    TRENDING_UP = "trending_up"     # 强势上涨
    TRENDING_DOWN = "trending_down" # 强势下跌
    MEAN_REVERTING = "mean_reverting"  # 均值回归
    HIGH_VOLATILITY = "high_volatility" # 高波动
    LOW_VOLATILITY = "low_volatility"   # 低波动
    NEUTRAL = "neutral"            # 中性


# -----------------------------------------------------------------------
# Helper operators (NumPy + pandas, no Talib)
# -----------------------------------------------------------------------

def _rank(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Cross-sectional percentile rank (0–1). / 横截面百分位排名."""
    if isinstance(x, pd.DataFrame):
        return x.rank(axis=1, pct=True)
    return x.rank(pct=True)


def _ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Time-series rank: rank of current value vs. last d values. / 时序排名."""
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: pd.Series(arr).rank(pct=True).iloc[-1], raw=False
    )


def _correlation(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling Pearson correlation over d periods. / 滚动皮尔逊相关."""
    return x.rolling(window=d, min_periods=d).corr(y)


def _covariance(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling covariance over d periods. / 滚动协方差."""
    return x.rolling(window=d, min_periods=d).cov(y)


def _decay_linear(x: pd.Series, d: int) -> pd.Series:
    """
    Linearly-weighted moving average over d periods.
    Most recent weight = d, oldest = 1.
    / 线性加权移动平均，最近权重最大.
    """
    result = pd.Series(index=x.index, dtype=np.float64)
    values = x.values
    n = len(values)
    for i in range(d - 1, n):
        window = values[i - d + 1:i + 1]
        weights = np.arange(1, d + 1, dtype=float)
        result.iloc[i] = np.dot(window, weights) / weights.sum()
    return result


def _delta(x: Union[pd.Series, pd.DataFrame], d: int) -> Union[pd.Series, pd.DataFrame]:
    """d-period difference. / d期差分."""
    return x.diff(d)


def _ts_argmax(x: pd.Series, d: int) -> pd.Series:
    """Index (0-based) of max in last d periods. / 最近d期最大值索引."""
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: np.argmax(arr), raw=True
    )


def _ts_argmin(x: pd.Series, d: int) -> pd.Series:
    """Index (0-based) of min in last d periods. / 最近d期最小值索引."""
    return x.rolling(window=d, min_periods=d).apply(
        lambda arr: np.argmin(arr), raw=True
    )


def _ts_max(x: pd.Series, d: int) -> pd.Series:
    """Rolling maximum over d periods. / 滚动最大值."""
    return x.rolling(window=d, min_periods=d).max()


def _ts_min(x: pd.Series, d: int) -> pd.Series:
    """Rolling minimum over d periods. / 滚动最小值."""
    return x.rolling(window=d, min_periods=d).min()


def _delay(x: pd.Series, d: int) -> pd.Series:
    """Value d periods ago. / d期前数值."""
    return x.shift(d)


def _signedpower(x: pd.Series, a: float) -> pd.Series:
    """x^{sign(x) * a}. / 有符号幂."""
    return np.sign(x) * (np.abs(x) ** a)


def _adv(d: int, df: pd.DataFrame) -> pd.Series:
    """Average daily volume over last d days. / d日平均成交量."""
    return df[f"adv{d}"]


def _stddev(x: pd.Series, d: int) -> pd.Series:
    """Rolling standard deviation. / 滚动标准差."""
    return x.rolling(window=d, min_periods=d).std()


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 包含所有 alpha 计算所需的列。
    Ensures df contains all columns required for alpha computation.
    """
    df = df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change()
    if "vwap" not in df.columns:
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum()
    if "cap" not in df.columns:
        df["cap"] = df["close"]
    for d in (5, 10, 20, 30, 60, 120):
        col = f"adv{d}"
        if col not in df.columns:
            df[col] = df["volume"].rolling(window=d, min_periods=1).mean()
    return df


# -----------------------------------------------------------------------
# Factor neutralization utilities
# -----------------------------------------------------------------------

def neutralize_zscore(
    factor: pd.Series,
    controls: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Z-score 中性化因子值（按市值等控制变量回归残差）。
    Z-score neutralization (regress out market-cap and other controls).

    Parameters
    ----------
    factor : pd.Series
        原始因子值 / Raw factor values.
    controls : pd.DataFrame, optional
        控制变量（如市值对数、所属行业虚拟变量）/ Control variables.

    Returns
    -------
    pd.Series : z-score 中性化后的因子值 / Neutralized factor.
    """
    mu = factor.mean()
    sigma = factor.std()
    neutral = (factor - mu) / (sigma + 1e-10)

    if controls is not None and not controls.empty:
        # Orthogonalize against controls using simple linear regression
        import numpy as np
        X = controls.fillna(0).values
        # Add intercept
        X = np.column_stack([np.ones(X.shape[0]), X])
        y = factor.fillna(0).values
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residual = y - X @ beta
            neutral = pd.Series(residual, index=factor.index)
            neutral = (neutral - neutral.mean()) / (neutral.std() + 1e-10)
        except Exception:
            pass  # Fallback to plain z-score

    return neutral


def neutralize_rank(
    factor: pd.Series,
    percentiles: bool = True,
) -> pd.Series:
    """
    横截面排名中性化因子（转换为百分位排名）。
    Cross-sectional rank neutralization (converts to percentile rank).

    Parameters
    ----------
    factor : pd.Series
        原始因子值 / Raw factor values.
    percentiles : bool
        若为 True，返回 0–1 百分位；若为 False，返回排名整数。

    Returns
    -------
    pd.Series : 排名中性化后的因子 / Rank-neutralized factor.
    """
    ranked = factor.rank(pct=percentiles, method="average")
    return ranked


def compute_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Information Coefficient — 因子与未来收益的秩相关。
    IC = rank_corr(factor, forward_return)

    Parameters
    ----------
    factor : pd.Series
        因子值序列 / Factor values.
    forward_returns : pd.Series
        未来收益序列 / Forward returns.
    method : str
        'spearman' (默认，秩相关) 或 'pearson' (线性相关).

    Returns
    -------
    float : IC 值，范围 [-1, 1] / IC value in [-1, 1].
    """
    # Align by index
    align_idx = factor.index.intersection(forward_returns.index)
    f = factor.loc[align_idx].values
    r = forward_returns.loc[align_idx].values
    mask = np.isfinite(f) & np.isfinite(r)
    f, r = f[mask], r[mask]
    if len(f) < 10:
        return 0.0
    if method == "spearman":
        from scipy.stats import spearmanr
        ic, _ = spearmanr(f, r)
    else:
        from scipy.stats import pearsonr
        ic, _ = pearsonr(f, r)
    return float(ic) if not np.isnan(ic) else 0.0


def compute_ir(
    ic_series: Union[List[float], pd.Series],
    periods_per_year: int = 252,
) -> float:
    """
    Information Ratio — IC 序列均值除以标准差。
    IR = mean(IC) / std(IC)，衡量 IC 的稳定性。

    Parameters
    ----------
    ic_series : list or pd.Series
        IC 时间序列 / Time series of IC values.
    periods_per_year : int
        年化周期数（默认252交易日）/ Periods per year for annualization.

    Returns
    -------
    float : IR 值 / IR value.
    """
    if isinstance(ic_series, list):
        ic_series = pd.Series(ic_series)
    ic_arr = np.array(ic_series.dropna())
    if len(ic_arr) < 2:
        return 0.0
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr)
    if std_ic < 1e-10:
        return 0.0
    return float(mean_ic / std_ic)


def compute_rolling_ic(
    factor: pd.Series,
    forward_returns: pd.Series,
    window: int = 20,
    method: str = "spearman",
) -> pd.Series:
    """
    滚动 IC（过去 window 期 IC 均值）。
    Rolling IC — mean IC over a rolling window.

    Parameters
    ----------
    factor : pd.Series
    forward_returns : pd.Series
    window : int
        滚动窗口大小（默认20）/ Rolling window size.
    method : str
        'spearman' or 'pearson'.

    Returns
    -------
    pd.Series : 滚动 IC 序列 / Rolling IC series.
    """
    ic_ts = pd.Series(index=factor.index, dtype=np.float64)
    for t in range(window, len(factor)):
        ic_val = compute_ic(
            factor.iloc[t - window:t],
            forward_returns.iloc[t - window:t],
            method=method,
        )
        ic_ts.iloc[t] = ic_val
    return ic_ts


# -----------------------------------------------------------------------
# AlphaFactor base class
# -----------------------------------------------------------------------

class AlphaFactor:
    """
    Alpha 因子基类 — 所有 101 因子的抽象基类。
    Alpha Factor Base Class — abstract base for all 101 formulaic alphas.

    Attributes
    ----------
    name : str
        因子名称 / Factor name.
    category : FactorCategory
        因子类别 / Factor category.
    formula : str
        因子公式描述（Kakushadze 2016 原始公式）/ Original formula string.
    description_zh : str
        中文描述 / Chinese description.
    description_en : str
        英文描述 / English description.

    Methods
    -------
    compute(df) -> pd.Series
        在单个 asset/stock 的时间序列上计算因子值。
        Compute factor on a single asset's time-series.

    neutralize(factor_value) -> pd.Series
        对因子值进行中性化处理（zscore 或 rank）。
        Apply neutralization to factor values.

    Examples
    --------
    >>> af = Alpha001()
    >>> factor_values = af.compute(df)  # 单标的 / single asset
    >>> neutralized = af.neutralize(factor_values)  # z-score neutralize
    """

    # 类级别的因子元数据 — 子类必须覆盖
    name: str = ""
    category: FactorCategory = FactorCategory.PRICE_RETURN
    formula: str = ""
    description_zh: str = ""
    description_en: str = ""

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        计算因子值（子类必须实现）。
        Compute factor values (must be implemented by subclass).

        Parameters
        ----------
        df : pd.DataFrame
            必须包含 OHLCV 列 + adv* 列的数据框。
            DataFrame with OHLCV columns + adv* columns.

        Returns
        -------
        pd.Series : 因子值序列 / Factor value series.
        """
        raise NotImplementedError("Subclass must implement compute()")

    def neutralize(
        self,
        factor: pd.Series,
        method: str = "zscore",
        controls: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        中性化因子值。
        Neutralize factor values.

        Parameters
        ----------
        factor : pd.Series
            原始因子值 / Raw factor values.
        method : str
            'zscore' (default) 或 'rank' / Neutralization method.
        controls : pd.DataFrame, optional
            zscore 中性化时的控制变量 / Control variables for z-score.

        Returns
        -------
        pd.Series : 中性化后的因子值 / Neutralized factor values.
        """
        if method == "zscore":
            return neutralize_zscore(factor, controls=controls)
        elif method == "rank":
            return neutralize_rank(factor)
        else:
            return factor

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.category.value}]>"


# -----------------------------------------------------------------------
# Price-based factors — Returns, Momentum, Reversal
# -----------------------------------------------------------------------

class Alpha001(AlphaFactor):
    """
    Alpha001: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
    Category: PRICE_REVERSAL — 逆性策略，捕捉短期反转信号。
    公式: 当收益率负时用收益波动率，否则用收盘价；取5日窗口最大值的排名。
    Description: Contrarian factor — when returns are negative, use volatility; when positive, use price level.
      Selects stocks with extreme recent deviations as potential reversal candidates.
    """
    name = "alpha_001"
    category = FactorCategory.PRICE_REVERSAL
    formula = "rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5"
    description_zh = "价格反转因子：收益率负时取波动率，否则取收盘价；5日最极端值排名"
    description_en = "Contrarian: use volatility when returns < 0 else price; rank of most extreme in 5-day window"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        r = df["returns"]
        cond = r < 0
        signed_power = np.where(cond, r.rolling(20).std(), df["close"])
        signed_power = _signedpower(pd.Series(signed_power, index=df.index), 2.0)
        return _rank(_ts_argmax(signed_power, 5)) - 0.5


class Alpha004(AlphaFactor):
    """
    Alpha004: -1 * rank(((rank((1/close)) * rank(volume)) / ((1-open/close) * rank((close-open)))))
    Category: PRICE_MOMENTUM — 价格动量与成交量结合。
    Description: Short-term reversal combined with volume — negative rank product signals overvalued.
    """
    name = "alpha_004"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * rank(((rank((1/close)) * rank(volume)) / ((1-open/close) * rank((close-open)))))"
    description_zh = "价格动量因子：价格倒数排名与成交量排名的比值，捕捉短期反转"
    description_en = "Price momentum: rank product of inverse price and volume, capturing short-term reversal"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        rank_close_inv = _rank(1 / df["close"])
        rank_vol = _rank(df["volume"])
        denom = (1 - df["open"] / df["close"]) * _rank(df["close"] - df["open"])
        return -_rank((rank_close_inv * rank_vol) / denom)


class Alpha008(AlphaFactor):
    """
    Alpha008: -1 * Ts_ArgMax(rank(close), 5)
    Category: PRICE_MOMENTUM — 5日价格排名最大值（顶部信号）。
    Description: Short-term momentum — identifies when close is at its 5-day high relative rank.
    """
    name = "alpha_008"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * Ts_ArgMax(rank(close), 5)"
    description_zh = "动量因子：5日收盘价排名最大值，捕捉短期顶部"
    description_en = "Momentum: 5-day rank argmax of close, identifies short-term peaks"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_ts_argmax(_rank(df["close"]), 5)


class Alpha009(AlphaFactor):
    """
    Alpha009: rank(delta(((close * 0.5) + (vwap * 0.5)), 20)) * -1
    Category: PRICE_MOMENTUM — 20日加权价格变化率。
    Description: 20-day momentum of blended price (close + vwap) — negative means downward pressure.
    """
    name = "alpha_009"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "rank(delta(((close * 0.5) + (vwap * 0.5)), 20)) * -1"
    description_zh = "动量因子：20日(收盘价+VWAP)/2的变化率，加权价格动量"
    description_en = "Momentum: 20-day change in (close+vwap)/2 blend, weighted price momentum"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        half_price = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return _rank(_delta(half_price, 20)) * -1


class Alpha010(AlphaFactor):
    """
    Alpha010: -1 * rank(((rank(0) < rank((close - vwap))) ? 1 : 0))
    Category: PRICE_REVERSAL — 价格对VWAP的偏离信号。
    Description: Relative value signal — close above vwap indicates overvaluation (negative signal).
    """
    name = "alpha_010"
    category = FactorCategory.PRICE_REVERSAL
    formula = "-1 * rank(((rank(0) < rank((close - vwap))) ? 1 : 0))"
    description_zh = "均值回归因子：收盘价相对VWAP偏离，捕捉定价偏差"
    description_en = "Mean reversion: close vs vwap deviation, captures pricing inefficiency"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        cond = _rank(pd.Series(0.0, index=df.index)) < _rank(df["close"] - df["vwap"])
        return -_rank(pd.Series(np.where(cond, 1.0, 0.0), index=df.index))


class Alpha017(AlphaFactor):
    """
    Alpha017: -1 * (rank(ts_rank(close, 30)) * -1 * rank((close / open)))
    Category: PRICE_MOMENTUM — 30日时序排名与收益率的组合动量。
    Description: Long-term time-series rank combined with short-term return direction.
    """
    name = "alpha_017"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * (rank(ts_rank(close, 30)) * -1 * rank((close / open)))"
    description_zh = "长期动量因子：30日时序排名结合当日收益率方向"
    description_en = "Long-term momentum: 30-day time-series rank combined with same-day return direction"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_ts_rank(df["close"], 30)) * _rank(df["close"] / df["open"])


class Alpha018(AlphaFactor):
    """
    Alpha018: -1 * rank((stddev(abs((close - open)), 5) + (close - open)) + correlation(open, volume, 10))
    Category: PRICE_MOMENTUM — 5日价格振幅与成交量相关性结合。
    Description: Combines intraday price range (volatility proxy) with open-volume correlation.
    """
    name = "alpha_018"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * rank((stddev(abs((close - open)), 5) + (close - open)) + correlation(open, volume, 10))"
    description_zh = "综合动量因子：5日价格振幅结合开价-成交量相关性"
    description_en = "Composite momentum: 5-day price range combined with open-volume correlation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        abs_ret = np.abs(df["close"] - df["open"])
        return -(_rank(_stddev(abs_ret, 5) + (df["close"] - df["open"])) +
                  _correlation(df["open"], df["volume"], 10))


class Alpha019(AlphaFactor):
    """
    Alpha019: -1 * sign(((close - delay(close, 7)) - delta(close, 7))) * (rank((close - vwap))))
    Category: PRICE_MOMENTUM — 7日价格动量与VWAP偏离的方向性信号。
    Description: Directional signal combining 7-day return momentum with relative value (close-vwap).
    """
    name = "alpha_019"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * sign(((close - delay(close, 7)) - delta(close, 7))) * (rank((close - vwap))))"
    description_zh = "方向动量因子：7日收益方向结合VWAP偏离度"
    description_en = "Directional momentum: 7-day return direction combined with vwap deviation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        close_d7 = _delay(df["close"], 7)
        delta_c7 = _delta(df["close"], 7)
        return -np.sign((df["close"] - close_d7) - delta_c7) * _rank(df["close"] - df["vwap"])


class Alpha021(AlphaFactor):
    """
    Alpha021: -1 * (rank((vwap - close)) / rank((vwap + close)))
    Category: PRICE_REVERSAL — VWAP与收盘价的相对位置。
    Description: Relative value: negative when close > vwap (overvalued), positive when close < vwap.
    """
    name = "alpha_021"
    category = FactorCategory.PRICE_REVERSAL
    formula = "-1 * (rank((vwap - close)) / rank((vwap + close)))"
    description_zh = "均值回归因子：VWAP相对收盘价的偏离，捕捉定价偏差"
    description_en = "Mean reversion: vwap vs close relative position, captures pricing deviation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank((df["vwap"] - df["close"]) / (df["vwap"] + df["close"]))


class Alpha023(AlphaFactor):
    """
    Alpha023: -1 * rank(((delta(((close * 0.5) + (vwap * 0.5)), 20)) * -1))
    Category: PRICE_MOMENTUM — 20日加权价格变化。
    Description: 20-day change in blended price — negative means downward.
    """
    name = "alpha_023"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * rank(((delta(((close * 0.5) + (vwap * 0.5)), 20)) * -1))"
    description_zh = "动量因子：20日加权价格变化率"
    description_en = "Momentum: 20-day change in (close+vwap)/2 blend"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        half_price = (df["close"] * 0.5) + (df["vwap"] * 0.5)
        return -_rank(_delta(half_price, 20))


class Alpha024(AlphaFactor):
    """
    Alpha024: -1 * rank(((rank((1/close))) * rank((volume / adv(20)))))
    Category: PRICE_RETURN — 成交量调整的价格倒数排名。
    Description: Volume-adjusted inverse price rank — cheap stocks with high volume.
    """
    name = "alpha_024"
    category = FactorCategory.PRICE_RETURN
    formula = "-1 * rank(((rank((1/close))) * rank((volume / adv(20)))))"
    description_zh = "价格收益因子：成交量调整的价格倒数排名，捕捉价值效应"
    description_en = "Price return: volume-adjusted inverse price rank, captures value effect"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_rank(1 / df["close"]) * _rank(df["volume"] / _adv(20, df)))


class Alpha027(AlphaFactor):
    """
    Alpha027: (rank((close - ts_max(close, 5)))) * -1 * (rank((close / ts_min(close, 5)))))
    Category: PRICE_MOMENTUM — 相对5日高低点的动量信号。
    Description: Mean reversion to 5-day range — positions between min and max.
    """
    name = "alpha_027"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "(rank((close - ts_max(close, 5)))) * -1 * (rank((close / ts_min(close, 5)))))"
    description_zh = "区间动量因子：收盘价相对5日高低点的位置"
    description_en = "Range momentum: close position within 5-day high-low range"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return _rank(df["close"] - _ts_max(df["close"], 5)) * _rank(df["close"] / _ts_min(df["close"], 5))


class Alpha032(AlphaFactor):
    """
    Alpha032: -1 * (rank((vwap - close)) / rank((vwap + close)))
    Category: PRICE_REVERSAL — VWAP标准化偏离（与Alpha021等价）。
    Description: Normalized VWAP deviation — same as Alpha021, captures relative value.
    """
    name = "alpha_032"
    category = FactorCategory.PRICE_REVERSAL
    formula = "-1 * (rank((vwap - close)) / rank((vwap + close)))"
    description_zh = "VWAP偏离因子：标准化VWAP偏离度"
    description_en = "VWAP deviation: normalized VWAP deviation, relative value signal"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank((df["vwap"] - df["close"]) / (df["vwap"] + df["close"]))


class Alpha039(AlphaFactor):
    """
    Alpha039: -1 * (rank((ts_max(close, 5) - close)) * (rank((close / ts_min(close, 5)))))
    Category: PRICE_MOMENTUM — 5日最高-收盘与收盘-最低的排名乘积。
    Description: Combines distance from 5-day high and ratio to 5-day low.
    """
    name = "alpha_039"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * (rank((ts_max(close, 5) - close)) * (rank((close / ts_min(close, 5)))))"
    description_zh = "双端动量因子：5日高-低距离与低-高比值的组合"
    description_en = "Dual momentum: combines 5-day high-low distance and low-high ratio"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_ts_max(df["close"], 5) - df["close"]) * _rank(df["close"] / _ts_min(df["close"], 5))


class Alpha044(AlphaFactor):
    """
    Alpha044: -1 * rank(correlation(low, volume, 5)) * (delta(close, 5) * -1)
    Category: PRICE_MOMENTUM — 低价-成交量相关性与5日价格变化的方向性组合。
    Description: Combines low-volume correlation (liquidity signal) with 5-day return direction.
    """
    name = "alpha_044"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * rank(correlation(low, volume, 5)) * (delta(close, 5) * -1)"
    description_zh = "流动性动量因子：低价-成交量相关性结合5日价格方向"
    description_en = "Liquidity momentum: low-volume correlation combined with 5-day price direction"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_correlation(df["low"], df["volume"], 5)) * _delta(df["close"], 5)


class Alpha046(AlphaFactor):
    """
    Alpha046: -1 * ((rank((stddev(close, 20)))) < rank((correlation(close, volume, 5))))
    Category: PRICE_RETURN — 20日波动率排名与5日量价相关性的比较。
    Description: Signals when price-volume correlation exceeds volatility rank.
    """
    name = "alpha_046"
    category = FactorCategory.PRICE_RETURN
    formula = "-1 * ((rank((stddev(close, 20)))) < rank((correlation(close, volume, 5))))"
    description_zh = "波动率比较因子：20日波动率排名与5日量价相关性比较"
    description_en = "Volatility comparison: 20-day vol rank vs 5-day price-volume correlation rank"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -((_rank(_stddev(df["close"], 20)) < _rank(_correlation(df["close"], df["volume"], 5)))).astype(float)


class Alpha060(AlphaFactor):
    """
    Alpha060: -1 * ((rank(delta((((close * 0.35) + (vwap * 0.65))), 5))) * -1 * (rank(correlation(vwap, volume, 5))))
    Category: PRICE_MOMENTUM — 加权价格5日变化与VWAP-成交量相关性的组合。
    Description: Weighted price delta combined with VWAP-volume correlation — composite signal.
    """
    name = "alpha_060"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * ((rank(delta((((close * 0.35) + (vwap * 0.65))), 5))) * -1 * (rank(correlation(vwap, volume, 5))))"
    description_zh = "复合动量因子：加权价格5日变化与VWAP-成交量相关性组合"
    description_en = "Composite momentum: weighted price 5-day delta combined with VWAP-volume correlation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        weighted_price = (df["close"] * 0.35) + (df["vwap"] * 0.65)
        return -_rank(_delta(weighted_price, 5)) * _rank(_correlation(df["vwap"], df["volume"], 5))


class Alpha071(AlphaFactor):
    """
    Alpha071: -1 * ((rank((ts_max(close, 5) - close))) * -1 * (rank((close / open))))
    Category: PRICE_MOMENTUM — 5日高点回落与收益率方向组合。
    Description: Pullback from 5-day high combined with return direction.
    """
    name = "alpha_071"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * ((rank((ts_max(close, 5) - close))) * -1 * (rank((close / open))))"
    description_zh = "回落动量因子：5日高点回落程度与收益率方向"
    description_en = "Pullback momentum: pullback from 5-day high combined with return direction"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_ts_max(df["close"], 5) - df["close"]) * _rank(df["close"] / df["open"])


class Alpha074(AlphaFactor):
    """
    Alpha074: -1 * ((rank(delta(close, 5))) * -1 * (rank((ts_max(close, 5) - close))))
    Category: PRICE_MOMENTUM — 5日价格变化与5日高点回落信号。
    Description: 5-day return combined with distance from 5-day high.
    """
    name = "alpha_074"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * ((rank(delta(close, 5))) * -1 * (rank((ts_max(close, 5) - close))))"
    description_zh = "双信号动量因子：5日价格变化与高点回落"
    description_en = "Dual momentum: 5-day return combined with distance from 5-day high"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_delta(df["close"], 5)) * _rank(_ts_max(df["close"], 5) - df["close"])


class Alpha101(AlphaFactor):
    """
    Alpha101: (close - open) / (high - low)
    Category: PRICE_MOMENTUM — 日内价格范围标准化收益率（振幅因子）。
    Description: Normalized intraday return by high-low range — amplitude.
    """
    name = "alpha_101"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "(close - open) / (high - low)"
    description_zh = "日内振幅因子：标准化日内收益率，高波动的低估股票"
    description_en = "Intraday amplitude: (close-open)/(high-low), high-vol low-price stocks"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)


# -----------------------------------------------------------------------
# Volume-based factors — Turnover, Volume-ratio
# -----------------------------------------------------------------------

class Alpha002(AlphaFactor):
    """
    Alpha002: -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)
    Category: VOLUME_TURNOVER — 成交量变化率排名与收益率排名的相关性。
    Description: Volume momentum vs price return — negative correlation means volume leads price.
    """
    name = "alpha_002"
    category = FactorCategory.VOLUME_TURNOVER
    formula = "-1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)"
    description_zh = "成交量动量因子：成交量变化率与收益率排名的相关性"
    description_en = "Volume momentum: log-volume change rank vs return rank correlation over 6 days"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        vol = df["volume"]
        rank_delta_vol = _rank(_delta(np.log(vol), 2))
        rank_ret = _rank((df["close"] - df["open"]) / df["open"])
        return -_correlation(rank_delta_vol, rank_ret, 6)


class Alpha003(AlphaFactor):
    """
    Alpha003: -1 * correlation(rank(open), rank(volume), 10)
    Category: VOLUME_RATIO — 开盘价排名与成交量排名的相关性。
    Description: Open price rank vs volume rank — captures opening auction dynamics.
    """
    name = "alpha_003"
    category = FactorCategory.VOLUME_RATIO
    formula = "-1 * correlation(rank(open), rank(volume), 10)"
    description_zh = "开盘量价因子：开盘价与成交量的排名相关性"
    description_en = "Open Auction: open price rank vs volume rank correlation over 10 days"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_correlation(_rank(df["open"]), _rank(df["volume"]), 10)


class Alpha007(AlphaFactor):
    """
    Alpha007: -1 * (rank((1/close)) * rank((volume / adv(20))))
    Category: VOLUME_RATIO — 成交量相对平均水平的价值调整排名。
    Description: Volume ratio rank multiplied by inverse price rank — volume effect in cheap stocks.
    """
    name = "alpha_007"
    category = FactorCategory.VOLUME_RATIO
    formula = "-1 * (rank((1/close)) * rank((volume / adv(20))))"
    description_zh = "成交量比率因子：相对平均成交量的价值调整因子"
    description_en = "Volume ratio: volume relative to 20-day ADV multiplied by inverse price rank"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(1 / df["close"]) * _rank(df["volume"] / _adv(20, df))


class Alpha012(AlphaFactor):
    """
    Alpha012: -1 * correlation(rank((open - (sum(vwap, 10) / 10))), rank(volume), 5)
    Category: VOLUME_TURNOVER — 开盘价对10日VWAP均值的偏离与成交量的关系。
    Description: Opening price deviation from 10-day VWAP mean vs volume — auction pressure signal.
    """
    name = "alpha_012"
    category = FactorCategory.VOLUME_TURNOVER
    formula = "-1 * correlation(rank((open - (sum(vwap, 10) / 10))), rank(volume), 5)"
    description_zh = "开盘偏离因子：开盘价对10日VWAP均值的偏离与成交量关系"
    description_en = "Opening deviation: open vs 10-day VWAP mean deviation correlated with volume"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        avg_vwap_10 = df["vwap"].rolling(10).mean()
        return -_correlation(_rank(df["open"] - avg_vwap_10), _rank(df["volume"]), 5)


class Alpha015(AlphaFactor):
    """
    Alpha015: -1 * rank(covariance(rank(close), rank(volume), 5))
    Category: VOLUME_RATIO — 收盘价与成交量的5日滚动协方差。
    Description: 5-day rolling covariance between price and volume ranks.
    """
    name = "alpha_015"
    category = FactorCategory.VOLUME_RATIO
    formula = "-1 * rank(covariance(rank(close), rank(volume), 5))"
    description_zh = "量价协方差因子：收盘价与成交量的5日滚动协方差"
    description_en = "Price-volume covariance: 5-day rolling covariance of price and volume ranks"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_covariance(_rank(df["close"]), _rank(df["volume"]), 5))


class Alpha028(AlphaFactor):
    """
    Alpha028: -1 * correlation(rank((close - ts_max(close, 5))), rank((volume / adv(20))), 5)
    Category: VOLUME_TURNOVER — 5日高点回落与成交量相对水平的滚动相关。
    Description: Correlation between pullback from 5-day high and volume ratio — reversal with volume.
    """
    name = "alpha_028"
    category = FactorCategory.VOLUME_TURNOVER
    formula = "-1 * correlation(rank((close - ts_max(close, 5))), rank((volume / adv(20))), 5)"
    description_zh = "回落成交量因子：5日高点回落与成交量相对水平的滚动相关"
    description_en = "Pullback volume: correlation of 5-day high pullback with relative volume level"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_correlation(
            _rank(df["close"] - _ts_max(df["close"], 5)),
            _rank(df["volume"] / _adv(20, df)),
            5
        )


class Alpha036(AlphaFactor):
    """
    Alpha036: -1 * (rank((covariance(rank(close), rank(volume), 5))))
    Category: VOLUME_RATIO — 收盘价与成交量的协方差排名（Alpha015变体）。
    Description: Covariance rank variant of Alpha015 — same signal family.
    """
    name = "alpha_036"
    category = FactorCategory.VOLUME_RATIO
    formula = "-1 * (rank((covariance(rank(close), rank(volume), 5))))"
    description_zh = "量价协方差排名因子：收盘价与成交量排名的协方差"
    description_en = "Price-volume covariance rank: covariance of price and volume ranks"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_covariance(_rank(df["close"]), _rank(df["volume"]), 5))


class Alpha051(AlphaFactor):
    """
    Alpha051: -1 * (rank((stddev(returns, 20)))) * correlation(close, volume, 5)
    Category: VOLatility × VOLUME — 20日收益波动率排名与5日量价相关性的乘积。
    Description: Combines volatility rank with liquidity (volume correlation).
    """
    name = "alpha_051"
    category = FactorCategory.VOLATILITY
    formula = "-1 * (rank((stddev(returns, 20)))) * correlation(close, volume, 5)"
    description_zh = "波动率-流动性因子：20日收益波动率排名与5日量价相关性"
    description_en = "Volatility-liquidity: 20-day return volatility rank times 5-day price-volume correlation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_stddev(df["returns"], 20)) * _correlation(df["close"], df["volume"], 5)


# -----------------------------------------------------------------------
# Volatility factors — Realized vol, GARCH-style
# -----------------------------------------------------------------------

class Alpha040(AlphaFactor):
    """
    Alpha040: -1 * ((rank((stddev(high, 10)))) * -1 * correlation(high, volume, 10))
    Category: VOLATILITY — 10日最高价波动率与10日量-最高价相关性。
    Description: High price volatility combined with high-volume correlation — strong trend signal.
    """
    name = "alpha_040"
    category = FactorCategory.VOLATILITY
    formula = "-1 * ((rank((stddev(high, 10)))) * -1 * correlation(high, volume, 10))"
    description_zh = "波动率因子：10日最高价波动率与量-最高价相关性"
    description_en = "Volatility: 10-day high price volatility combined with high-volume correlation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_stddev(df["high"], 10)) * _correlation(df["high"], df["volume"], 10)


class Alpha056(AlphaFactor):
    """
    Alpha056: -1 * ((rank(correlation(rank((high - low)), rank((volume / adv(20))), 5))) * -1)
    Category: VOLATILITY — 日内波动幅度排名与成交量相对水平排名的5日相关性。
    Description: Intraday range vs volume ratio — captures volatility-volume relationship.
    """
    name = "alpha_056"
    category = FactorCategory.VOLATILITY
    formula = "-1 * ((rank(correlation(rank((high - low)), rank((volume / adv(20))), 5))) * -1)"
    description_zh = "波动率-成交量因子：日内波幅与成交量相对水平的相关性"
    description_en = "Volatility-volume: intraday range vs relative volume level correlation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_correlation(
            _rank(df["high"] - df["low"]),
            _rank(df["volume"] / _adv(20, df)),
            5
        ))


# -----------------------------------------------------------------------
# Microstructure factors — Bid-ask spread, order flow imbalance
# -----------------------------------------------------------------------

class Alpha016(AlphaFactor):
    """
    Alpha016: -1 * ((rank(correlation((high), rank(volume), 5))) * -1)
    Category: MICROSTRUCTURE — 最高价与成交量排名的5日滚动相关。
    Description: Price-volume microstructure — correlation of high with volume rank.
    """
    name = "alpha_016"
    category = FactorCategory.MICROSTRUCTURE
    formula = "-1 * ((rank(correlation((high), rank(volume), 5))) * -1)"
    description_zh = "微观结构因子：最高价与成交量排名的相关性"
    description_en = "Microstructure: high price vs volume rank correlation over 5 days"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_correlation(df["high"], _rank(df["volume"]), 5))


class Alpha026(AlphaFactor):
    """
    Alpha026: -1 * ((rank(correlation((close), (volume), 5))) * -1 * rank((close - open))))
    Category: MICROSTRUCTURE — 量价相关性与价格变化方向的组合。
    Description: Combines 5-day price-volume correlation with intraday return direction.
    """
    name = "alpha_026"
    category = FactorCategory.MICROSTRUCTURE
    formula = "-1 * ((rank(correlation((close), (volume), 5))) * -1 * rank((close - open))))"
    description_zh = "微观结构因子：量价相关性与日内收益率方向组合"
    description_en = "Microstructure: 5-day price-volume correlation combined with intraday return"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_correlation(df["close"], df["volume"], 5)) * _rank(df["close"] - df["open"])


class Alpha037(AlphaFactor):
    """
    Alpha037: -1 * ((rank(correlation(rank(close), rank(volume), 5))) * -1) * delta(close, 5)
    Category: MICROSTRUCTURE — 量价排名相关性变化与5日价格变化的组合。
    Description: Price-volume rank correlation change combined with 5-day return.
    """
    name = "alpha_037"
    category = FactorCategory.MICROSTRUCTURE
    formula = "-1 * ((rank(correlation(rank(close), rank(volume), 5))) * -1) * delta(close, 5)"
    description_zh = "微观结构动量因子：量价排名相关性变化结合5日价格动量"
    description_en = "Microstructure momentum: price-volume rank correlation change combined with 5-day return"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_correlation(_rank(df["close"]), _rank(df["volume"]), 5)) * _delta(df["close"], 5)


class Alpha059(AlphaFactor):
    """
    Alpha059: -1 * ((rank(correlation(((high * 0.9) + (close * 0.1)), volume, 5))) * -1)
    Category: MICROSTRUCTURE — 加权价格（高+收盘）与成交量的相关性。
    Description: High-weighted price vs volume correlation — aggressive buying pressure signal.
    """
    name = "alpha_059"
    category = FactorCategory.MICROSTRUCTURE
    formula = "-1 * ((rank(correlation(((high * 0.9) + (close * 0.1)), volume, 5))) * -1)"
    description_zh = "订单流因子：加权价格与成交量相关性，捕捉积极买入压力"
    description_en = "Order flow: high-weighted price vs volume correlation, captures aggressive buying"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        weighted = (df["high"] * 0.9) + (df["close"] * 0.1)
        return -_rank(_correlation(weighted, df["volume"], 5))


# -----------------------------------------------------------------------
# Cross-sectional rank factors
# -----------------------------------------------------------------------

class Alpha006(AlphaFactor):
    """
    Alpha006: -1 * correlation(rank(open), rank(volume), 10)
    Category: CROSS_SECTIONAL — 开盘价与成交量的横截面排名相关性。
    Description: Cross-sectional rank correlation of open and volume — market-wide auction signal.
    """
    name = "alpha_006"
    category = FactorCategory.CROSS_SECTIONAL
    formula = "-1 * correlation(rank(open), rank(volume), 10)"
    description_zh = "横截面因子：开盘价与成交量的横截面排名相关性"
    description_en = "Cross-sectional: open vs volume rank correlation across market"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_correlation(_rank(df["open"]), _rank(df["volume"]), 10)


class Alpha013(AlphaFactor):
    """
    Alpha013: -1 * ((rank((open - vwap))) * -1 * rank((open - close)))
    Category: CROSS_SECTIONAL — 开盘价对VWAP偏离与开盘-收盘价的排名组合。
    Description: Opening auction mispricing combined with intraday direction.
    """
    name = "alpha_013"
    category = FactorCategory.CROSS_SECTIONAL
    formula = "-1 * ((rank((open - vwap))) * -1 * rank((open - close)))"
    description_zh = "横截面因子：开盘价对VWAP偏离与开盘-收盘偏离组合"
    description_en = "Cross-sectional: open vs VWAP deviation combined with open-close spread"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(df["open"] - df["vwap"]) * _rank(df["open"] - df["close"])


class Alpha020(AlphaFactor):
    """
    Alpha020: -1 * rank((open - (sum(vwap, 10) / 10)))) * rank((abs((close - open))))
    Category: CROSS_SECTIONAL — 开盘价对10日VWAP均值的偏离幅度。
    Description: Opening deviation from 10-day VWAP mean — overnight gap signal.
    """
    name = "alpha_020"
    category = FactorCategory.CROSS_SECTIONAL
    formula = "-1 * rank((open - (sum(vwap, 10) / 10)))) * rank((abs((close - open))))"
    description_zh = "横截面因子：开盘价对10日VWAP均值的偏离与日内振幅"
    description_en = "Cross-sectional: open deviation from 10-day VWAP mean combined with intraday range"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        avg_vwap_10 = df["vwap"].rolling(10).mean()
        return -_rank(df["open"] - avg_vwap_10) * _rank(np.abs(df["close"] - df["open"]))


class Alpha033(AlphaFactor):
    """
    Alpha033: -1 * ((rank(correlation((vwap), (volume), 5))) * delta(close, 5))
    Category: CROSS_SECTIONAL — VWAP-成交量相关性与5日价格变化。
    Description: VWAP-volume correlation with 5-day price change — composite cross-sectional signal.
    """
    name = "alpha_033"
    category = FactorCategory.CROSS_SECTIONAL
    formula = "-1 * ((rank(correlation((vwap), (volume), 5))) * delta(close, 5))"
    description_zh = "横截面复合因子：VWAP-成交量相关性与5日价格变化"
    description_en = "Cross-sectional composite: VWAP-volume correlation combined with 5-day price change"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_correlation(df["vwap"], df["volume"], 5) * _delta(df["close"], 5)


# -----------------------------------------------------------------------
# Additional high-impact alphas (not in top 25 but highly effective)
# -----------------------------------------------------------------------

class Alpha014(AlphaFactor):
    """
    Alpha014: -1 * rank(((((high - low) / (sum(high, 5) / 5)) * exp(-1 * ((close - open) / close)))))
    Category: VOLATILITY — 5日高低比乘以收益率指数衰减。
    Description: Relative high-low range times exponential decay of return — volatility signal.
    """
    name = "alpha_014"
    category = FactorCategory.VOLATILITY
    formula = "-1 * rank(((((high - low) / (sum(high, 5) / 5)) * exp(-1 * ((close - open) / close)))))"
    description_zh = "波动率因子：相对高低比乘以收益率指数衰减"
    description_en = "Volatility: relative high-low ratio times exponential decay of return"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        hl_range = df["high"] - df["low"]
        avg_high_5 = df["high"].rolling(5).mean()
        price_ret = (df["close"] - df["open"]) / df["close"]
        return -_rank((hl_range / avg_high_5) * np.exp(-1 * price_ret))


class Alpha029(AlphaFactor):
    """
    Alpha029: ((rank((close - ts_min(close, 5)))) * -1) * (rank((close - open))))
    Category: PRICE_REVERSAL — 5日低点反弹与日内收益率组合。
    Description: Bounce from 5-day low combined with intraday return direction.
    """
    name = "alpha_029"
    category = FactorCategory.PRICE_REVERSAL
    formula = "((rank((close - ts_min(close, 5)))) * -1) * (rank((close - open))))"
    description_zh = "反弹因子：5日低点反弹与日内收益率"
    description_en = "Reversal: bounce from 5-day low combined with intraday return"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(df["close"] - _ts_min(df["close"], 5)) * _rank(df["close"] - df["open"])


class Alpha031(AlphaFactor):
    """
    Alpha031: -1 * ((rank(ts_rank(close, 10)) * -1) * rank((close / open)))
    Category: PRICE_MOMENTUM — 10日时序排名与收益率方向。
    Description: Short-term time-series rank with intraday momentum.
    """
    name = "alpha_031"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * ((rank(ts_rank(close, 10)) * -1) * rank((close / open)))"
    description_zh = "短期动量因子：10日时序排名与日内收益率"
    description_en = "Short-term momentum: 10-day time-series rank with intraday momentum"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -_rank(_ts_rank(df["close"], 10)) * _rank(df["close"] / df["open"])


class Alpha057(AlphaFactor):
    """
    Alpha057: -1 * ((rank((ts_max(close, 5)) - rank(ts_min(close, 5)))) * -1 * correlation(close, volume, 5)
    Category: PRICE_MOMENTUM — 5日高低排名的差距与量价相关性的组合。
    Description: 5-day high-low range rank combined with volume correlation.
    """
    name = "alpha_057"
    category = FactorCategory.PRICE_MOMENTUM
    formula = "-1 * ((rank((ts_max(close, 5)) - rank(ts_min(close, 5)))) * -1 * correlation(close, volume, 5)"
    description_zh = "动量因子：5日高低排名差距与量价相关性"
    description_en = "Momentum: 5-day high-low range rank combined with price-volume correlation"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        df = _ensure_ohlcv(df)
        return -(_rank(_ts_max(df["close"], 5)) - _rank(_ts_min(df["close"], 5))) * _correlation(df["close"], df["volume"], 5)


# -----------------------------------------------------------------------
# Alpha101Bundle — 101-factor bundle with regime-adaptive weights
# -----------------------------------------------------------------------

# Registry of all implemented AlphaFactor subclasses
_ALPHA_FACTOR_REGISTRY: Dict[str, type] = {
    "alpha_001": Alpha001,
    "alpha_002": Alpha002,
    "alpha_003": Alpha003,
    "alpha_004": Alpha004,
    "alpha_006": Alpha006,
    "alpha_007": Alpha007,
    "alpha_008": Alpha008,
    "alpha_009": Alpha009,
    "alpha_010": Alpha010,
    "alpha_012": Alpha012,
    "alpha_013": Alpha013,
    "alpha_014": Alpha014,
    "alpha_015": Alpha015,
    "alpha_016": Alpha016,
    "alpha_017": Alpha017,
    "alpha_018": Alpha018,
    "alpha_019": Alpha019,
    "alpha_020": Alpha020,
    "alpha_021": Alpha021,
    "alpha_023": Alpha023,
    "alpha_024": Alpha024,
    "alpha_026": Alpha026,
    "alpha_027": Alpha027,
    "alpha_028": Alpha028,
    "alpha_029": Alpha029,
    "alpha_031": Alpha031,
    "alpha_032": Alpha032,
    "alpha_033": Alpha033,
    "alpha_036": Alpha036,
    "alpha_037": Alpha037,
    "alpha_039": Alpha039,
    "alpha_040": Alpha040,
    "alpha_044": Alpha044,
    "alpha_046": Alpha046,
    "alpha_051": Alpha051,
    "alpha_056": Alpha056,
    "alpha_057": Alpha057,
    "alpha_059": Alpha059,
    "alpha_060": Alpha060,
    "alpha_071": Alpha071,
    "alpha_074": Alpha074,
    "alpha_101": Alpha101,
}


# Regime-specific factor weight multipliers
# Higher weight = factor is more predictive in that regime
_REGIME_WEIGHTS: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.TRENDING_UP: {
        "alpha_008": 1.5, "alpha_017": 1.5, "alpha_019": 1.4,
        "alpha_023": 1.3, "alpha_031": 1.3, "alpha_071": 1.2,
        "alpha_040": 0.8, "alpha_002": 0.9, "alpha_101": 1.0,
    },
    MarketRegime.TRENDING_DOWN: {
        "alpha_001": 1.5, "alpha_010": 1.4, "alpha_021": 1.4,
        "alpha_029": 1.3, "alpha_032": 1.3, "alpha_039": 1.2,
        "alpha_040": 0.8, "alpha_059": 0.9, "alpha_074": 1.1,
    },
    MarketRegime.MEAN_REVERTING: {
        "alpha_001": 1.5, "alpha_004": 1.4, "alpha_010": 1.4,
        "alpha_021": 1.3, "alpha_029": 1.3, "alpha_032": 1.3,
        "alpha_051": 1.2, "alpha_056": 1.1, "alpha_024": 1.1,
    },
    MarketRegime.HIGH_VOLATILITY: {
        "alpha_014": 1.6, "alpha_040": 1.5, "alpha_051": 1.5,
        "alpha_056": 1.4, "alpha_036": 1.3, "alpha_044": 1.2,
        "alpha_017": 0.8, "alpha_008": 0.8, "alpha_060": 1.0,
    },
    MarketRegime.LOW_VOLATILITY: {
        "alpha_002": 1.4, "alpha_003": 1.3, "alpha_006": 1.3,
        "alpha_007": 1.2, "alpha_012": 1.2, "alpha_015": 1.2,
        "alpha_026": 1.1, "alpha_033": 1.1, "alpha_028": 1.1,
    },
    MarketRegime.NEUTRAL: {
        f"alpha_{i:03d}": 1.0 for i in range(1, 102)
    },
}


def _detect_regime(
    returns: pd.Series,
    volatility: Optional[pd.Series] = None,
    window: int = 20,
) -> MarketRegime:
    """
    根据近期收益和波动率检测市场状态。
    Detect market regime based on recent returns and volatility.

    Parameters
    ----------
    returns : pd.Series
        收益率序列 / Return series.
    volatility : pd.Series, optional
        波动率序列（如已计算）/ Volatility series (pre-computed).
    window : int
        检测窗口（默认20）/ Detection window.

    Returns
    -------
    MarketRegime : 检测到的市场状态 / Detected market regime.
    """
    recent = returns.iloc[-window:] if len(returns) >= window else returns
    mean_ret = recent.mean()
    if volatility is None:
        vol = recent.std()
    else:
        vol = volatility.iloc[-1] if len(volatility) >= 1 else recent.std()

    # Annualized vol threshold
    ann_vol = vol * np.sqrt(252) if vol is not None else 0.0
    vol_threshold_high = 0.25  # 25% annualized vol = high volatility
    vol_threshold_low = 0.10   # 10% annualized vol = low volatility

    if ann_vol >= vol_threshold_high:
        return MarketRegime.HIGH_VOLATILITY
    if ann_vol <= vol_threshold_low:
        return MarketRegime.LOW_VOLATILITY

    # Trend detection: use 5-day return sign consistency
    sign_changes = (np.sign(recent) != np.sign(mean_ret)).sum()
    if mean_ret > 0 and sign_changes <= window * 0.3:
        return MarketRegime.TRENDING_UP
    if mean_ret < 0 and sign_changes <= window * 0.3:
        return MarketRegime.TRENDING_DOWN

    return MarketRegime.NEUTRAL


class Alpha101Bundle:
    """
    Alpha101Bundle — 101因子Bundle，支持自适应市场状态权重。
    101-factor bundle with regime-adaptive factor weights.

    Features / 功能
    ---------------
    - Compute all (or a subset of) 101 alpha factors from a DataFrame
    - Regime-adaptive factor weighting (trending / mean-reverting / high-vol / ...)
    - IC / IR evaluation of factor predictive power
    - Factor neutralization (zscore / rank)
    - Top-K factor selection by IC

    Usage / 使用示例
    ---------------
    >>> bundle = Alpha101Bundle()
    >>> df = bundle.compute(df)                     # compute all implemented alphas
    >>> df = bundle.compute(df, names=["alpha_001", "alpha_002", "alpha_101"])
    >>> regime = bundle.detect_regime(df["returns"])
    >>> bundle.set_regime(regime)
    >>> weights = bundle.get_factor_weights()        # regime-adaptive weights
    >>> ic_dict = bundle.evaluate_ic(df, forward_returns)

    Attributes
    ----------
    factors : Dict[str, AlphaFactor]
        因子名称到因子实例的映射 / Factor name to instance mapping.
    factor_weights : Dict[str, float]
        当前市场状态下的因子权重 / Current regime factor weights.
    current_regime : MarketRegime
        当前检测到的市场状态 / Currently detected market regime.
    """

    def __init__(
        self,
        names: Optional[List[str]] = None,
        neutralization: str = "rank",
    ):
        """
        初始化 Alpha101Bundle.
        Initialize Alpha101Bundle.

        Parameters
        ----------
        names : list of str, optional
            要计算的因子名称列表（默认：所有已实现因子）。
            List of factor names to compute (default: all implemented).
        neutralization : str
            'rank' (横截面百分位排名) 或 'zscore' (Z-score中性化).
            Neutralization method: 'rank' (percentile) or 'zscore' (z-score).
        """
        # Build factor instances for requested names (or all)
        if names is None:
            names = list(_ALPHA_FACTOR_REGISTRY.keys())

        self.factors: Dict[str, AlphaFactor] = {}
        for name in names:
            if name in _ALPHA_FACTOR_REGISTRY:
                self.factors[name] = _ALPHA_FACTOR_REGISTRY[name]()

        self.neutralization = neutralization
        self.current_regime: MarketRegime = MarketRegime.NEUTRAL
        self._factor_weights: Dict[str, float] = {n: 1.0 for n in self.factors}
        # IC history for adaptive weighting
        self._ic_history: Dict[str, List[float]] = {n: [] for n in self.factors}

    # ------------------------------------------------------------------
    # Regime detection & weight adaptation
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None,
        window: int = 20,
    ) -> MarketRegime:
        """
        检测市场状态。
        Detect current market regime.

        Parameters
        ----------
        returns : pd.Series
            收益率序列 / Return series.
        volatility : pd.Series, optional
            波动率序列 / Volatility series.
        window : int
            检测窗口（默认20）/ Detection window.

        Returns
        -------
        MarketRegime : 检测到的市场状态 / Detected regime.
        """
        self.current_regime = _detect_regime(returns, volatility, window)
        return self.current_regime

    def set_regime(self, regime: MarketRegime) -> None:
        """
        手动设置市场状态。
        Manually set market regime.

        Parameters
        ----------
        regime : MarketRegime
            市场状态 / Market regime.
        """
        self.current_regime = regime
        self._apply_regime_weights()

    def _apply_regime_weights(self) -> None:
        """根据当前市场状态更新因子权重。"""
        base = _REGIME_WEIGHTS.get(self.current_regime, _REGIME_WEIGHTS[MarketRegime.NEUTRAL])
        for name in self.factors:
            self._factor_weights[name] = base.get(name, 1.0)

    def get_factor_weights(self) -> Dict[str, float]:
        """
        获取当前市场状态下的因子权重。
        Get factor weights for current regime.

        Returns
        -------
        Dict[str, float] : 因子名称到权重值的映射 / Factor name to weight mapping.
        """
        return self._factor_weights.copy()

    # ------------------------------------------------------------------
    # Factor computation
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        names: Optional[List[str]] = None,
        neutralized: bool = True,
    ) -> pd.DataFrame:
        """
        计算 alpha 因子值（可指定子集）。
        Compute alpha factor values (can specify a subset).

        Parameters
        ----------
        df : pd.DataFrame
            包含 OHLCV + volume 列的数据框。
            DataFrame with OHLCV + volume columns.
        names : list of str, optional
            要计算的因子名称列表（默认：所有已注册因子）。
            List of factor names to compute (default: all registered).
        neutralized : bool
            是否对因子值进行中性化（默认True）。

        Returns
        -------
        pd.DataFrame : 添加了 alpha_* 列的数据框。
        DataFrame with additional alpha_* columns.
        """
        df = _ensure_ohlcv(df)
        names_to_compute = names or list(self.factors.keys())

        for name in names_to_compute:
            if name not in self.factors:
                continue
            factor = self.factors[name]
            values = factor.compute(df)
            if neutralized:
                values = factor.neutralize(values, method=self.neutralization)
            df[name] = values

        return df

    def compute_composite(
        self,
        df: pd.DataFrame,
        top_k: int = 10,
        ic_threshold: float = 0.02,
        neutralized: bool = True,
    ) -> pd.Series:
        """
        计算加权复合因子（取 IC 最高的 top_k 因子加权求和）。
        Compute weighted composite alpha (top-K by IC, weighted sum).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV 数据框 / OHLCV DataFrame.
        top_k : int
            选取的 top K 因子数量（默认10）/ Number of top factors to use.
        ic_threshold : float
            IC 最低阈值（低于此则不使用）/ Minimum IC threshold.
        neutralized : bool
            是否中性化（默认True）/ Whether to neutralize.

        Returns
        -------
        pd.Series : 复合因子值 / Composite factor values.
        """
        # First compute all registered factors
        df = self.compute(df, neutralized=neutralized)

        # Evaluate IC for each factor if forward returns are available
        if "forward_return" in df.columns:
            ic_scores: Dict[str, float] = {}
            for name in self.factors:
                if name in df.columns:
                    ic = compute_ic(df[name], df["forward_return"])
                    ic_scores[name] = abs(ic)
            # Sort by IC
            sorted_factors = sorted(ic_scores, key=lambda x: ic_scores[x], reverse=True)
            selected = [f for f in sorted_factors if ic_scores[f] >= ic_threshold][:top_k]
        else:
            # Fallback: use equal weights for all factors sorted alphabetically
            selected = sorted(self.factors.keys())[:top_k]

        # Compute weighted composite
        composite = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for name in selected:
            w = self._factor_weights.get(name, 1.0)
            composite += w * df[name].fillna(0)
            total_weight += w

        if total_weight > 0:
            composite /= total_weight

        return composite

    # ------------------------------------------------------------------
    # IC / IR evaluation
    # ------------------------------------------------------------------

    def evaluate_ic(
        self,
        df: pd.DataFrame,
        forward_returns: Union[str, pd.Series],
        window: int = 20,
    ) -> Dict[str, Dict[str, float]]:
        """
        计算每个因子的 IC 和 IR 指标。
        Evaluate IC and IR for each factor.

        Parameters
        ----------
        df : pd.DataFrame
            包含因子值的数据框（由 compute() 返回）/ DataFrame with factor values.
        forward_returns : str or pd.Series
            未来收益列名或序列 / Column name or series of forward returns.
        window : int
            滚动 IC 窗口（默认20）/ Rolling IC window.

        Returns
        -------
        Dict[str, Dict[str, float]] : 每个因子的 IC 均值、IC标准差、IR 值。
        Per-factor IC mean, IC std, and IR.
        """
        if isinstance(forward_returns, str):
            if forward_returns not in df.columns:
                raise ValueError(f"Column '{forward_returns}' not found in df")
            fwd = df[forward_returns]
        else:
            fwd = forward_returns

        results = {}
        for name in self.factors:
            if name not in df.columns:
                continue
            ic_ts = compute_rolling_ic(df[name], fwd, window=window)
            ic_arr = ic_ts.dropna()
            if len(ic_arr) < 2:
                results[name] = {"ic_mean": 0.0, "ic_std": 0.0, "ir": 0.0}
                continue

            ic_mean = float(ic_arr.mean())
            ic_std = float(ic_arr.std())
            ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

            # Update IC history for adaptive weighting
            if len(fwd) > 0:
                latest_ic = compute_ic(df[name], fwd)
                history = self._ic_history[name]
                history.append(latest_ic)
                # Keep only last 60 observations
                if len(history) > 60:
                    history.pop(0)

            results[name] = {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
            }

        return results

    def adaptive_weights(
        self,
        ic_results: Dict[str, Dict[str, float]],
        decay: float = 0.95,
    ) -> Dict[str, float]:
        """
        根据 IC 结果更新自适应因子权重（指数移动平均加权）。
        Update adaptive factor weights based on IC results (EMA weighting).

        Parameters
        ----------
        ic_results : Dict[str, Dict[str, float]]
            evaluate_ic() 返回的 IC 结果 / IC results from evaluate_ic().
        decay : float
            历史 IC 衰减因子（默认0.95）/ Historical IC decay factor.

        Returns
        -------
        Dict[str, float] : 更新后的因子权重 / Updated factor weights.
        """
        for name, stats in ic_results.items():
            if name not in self._factor_weights:
                continue
            # Use IC mean (absolute) as weight signal
            ic_signal = abs(stats["ic_mean"])
            # Blend with historical IC EMA
            history = self._ic_history.get(name, [])
            if history:
                hist_mean = np.mean(history)
                blended_ic = decay * hist_mean + (1 - decay) * ic_signal
            else:
                blended_ic = ic_signal

            # Normalize: base weight 0.5 + 2 * blended_ic (capped at 2.0)
            new_weight = min(2.0, max(0.1, 0.5 + 2.0 * blended_ic))
            self._factor_weights[name] = new_weight

        return self._factor_weights.copy()

    # ------------------------------------------------------------------
    # Factor selection
    # ------------------------------------------------------------------

    def select_top_k(
        self,
        ic_results: Dict[str, Dict[str, float]],
        k: int = 20,
        min_ir: float = 0.1,
        max_correlation: float = 0.7,
    ) -> List[str]:
        """
        选择 IC 最高、互相低相关的 top-K 因子。
        Select top-K factors by IC with low inter-factor correlation.

        Parameters
        ----------
        ic_results : Dict[str, Dict[str, float]]
            IC evaluation results / IC evaluation results.
        k : int
            选取的因子数量（默认20）/ Number of factors to select.
        min_ir : float
            IR 最低阈值（默认0.1）/ Minimum IR threshold.
        max_correlation : float
            因子间最大相关系数（默认0.7）/ Maximum inter-factor correlation.

        Returns
        -------
        List[str] : 选中的因子名称列表 / Selected factor names.
        """
        # Filter by IR threshold
        viable = [
            name for name, stats in ic_results.items()
            if stats["ir"] >= min_ir
        ]
        # Sort by IR descending
        viable.sort(key=lambda x: ic_results[x]["ir"], reverse=True)

        selected = []
        for candidate in viable:
            if len(selected) >= k:
                break
            # Check correlation with already selected factors
            is_correlated = False
            for sel in selected:
                corr = np.corrcoef(
                    self._ic_history.get(candidate, [0] * 10),
                    self._ic_history.get(sel, [0] * 10)
                )[0, 1]
                if abs(corr) > max_correlation:
                    is_correlated = True
                    break
            if not is_correlated:
                selected.append(candidate)

        return selected

    def __repr__(self) -> str:
        n = len(self.factors)
        regime = self.current_regime.value
        return f"<Alpha101Bundle [{n} factors, regime={regime}]>"
