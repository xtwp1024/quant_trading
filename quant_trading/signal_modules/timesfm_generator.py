"""TimesFM 时间序列预测信号生成器

基于 Google TimesFM 2.5 foundation model 生成交易信号。

Usage
-----
```python
import pandas as pd
from quant_trading.signal_modules.timesfm_generator import TimesFMGenerator

df = pd.read_csv("BTC_USDT_1h.csv")
gen = TimesFMGenerator(
    symbol="BTC/USDT",
    horizon=24,          # 预测24个周期
    context_len=256,     # 用256个点作为上下文
    buy_threshold=0.02,  # 预测涨幅>2%买入
    sell_threshold=-0.02,
)
signals = gen.generate(df)
```
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal_modules.types import Signal, SignalType, SignalDirection


class TimesFMModel:
    """TimesFM 模型加载器（延迟加载，避免启动时下载模型）"""

    _instance: Optional[Any] = None
    _model: Optional[Any] = None

    @classmethod
    def get_model(cls, backend: str = "torch") -> Any:
        """单例获取 TimesFM 模型"""
        if cls._instance is None:
            import timesfm

            if backend == "torch":
                cls._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    "google/timesfm-2.5-200m-pytorch"
                )
            else:
                raise NotImplementedError("Only torch backend is supported")
            cls._instance = True
        return cls._model


@dataclass
class TimesFMGenerator:
    """TimesFM 时间序列预测信号生成器

    基于 TimesFM 预测未来价格走势，生成趋势跟踪信号。

    Signal Logic
    -----------
    - BUY:  预测收益 > buy_threshold
    - SELL: 预测收益 < sell_threshold（负数）
    - HOLD: 其他情况

    TimesFM 预测 horizon 个时间步后的价格变化，
    signal_strength = min(|predicted_return| / threshold, 1.0)
    """

    symbol: str = "UNKNOWN"
    horizon: int = 24          # 预测步数（对应数据的时间粒度）
    context_len: int = 256     # 用多少个历史点做上下文（max 1024）
    buy_threshold: float = 0.02    # 预测涨幅 > 2% 买入
    sell_threshold: float = -0.02  # 预测跌幅 < -2% 卖出
    use_quantile: bool = True      # 使用分位数预测（更稳健）
    quantile_weights: tuple = (0.1, 0.2, 0.4, 0.2, 0.1)  # P10-P90加权
    min_context_points: int = 50   # 最少需要的上下文点数
    backend: str = "torch"         # torch 或jax
    model: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled: bool = False

    def _ensure_model(self) -> Any:
        """延迟加载模型"""
        if self.model is None:
            self.model = TimesFMModel.get_model(backend=self.backend)
        return self.model

    def _compile(self) -> None:
        """编译模型（首次调用加速）"""
        if self._compiled:
            return
        model = self._ensure_model()
        import timesfm
        import numpy as np
        model.compile(
            timesfm.ForecastConfig(
                max_context=min(self.context_len, 1024),
                max_horizon=self.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        # 预热推理
        dummy = np.random.randn(self.context_len).astype(np.float32)
        model.forecast(inputs=[dummy])
        self._compiled = True

    def _get_context(self, df: pd.DataFrame) -> np.ndarray:
        """从 DataFrame 提取上下文数据"""
        self._require_cols(df, ["close"])

        close = df["close"].values
        # 取最后 context_len 个点
        ctx = close[-self.context_len :]
        # 归一化：转换为相对变化率（TimesFM 更擅长处理这类输入）
        if len(ctx) > 1:
            # 转为一阶差分序列，保留趋势信息
            ctx = np.diff(ctx, prepend=ctx[0])
        return ctx.astype(np.float32)

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        """从 OHLCV DataFrame 生成 TimesFM 预测信号"""
        self._require_cols(df, ["close"])

        if len(df) < self.min_context_points:
            return []

        signals = []
        close = df["close"].values
        current_price = close[-1]

        try:
            model = self._ensure_model()
            self._compile()
        except Exception as e:
            # 模型加载失败，返回空信号
            return []

        context = self._get_context(df)

        try:
            point_forecast, quantile_forecast = model.forecast(inputs=[context])
        except Exception:
            return []

        # point_forecast: (1, horizon) or similar
        # quantile_forecast: (1, horizon, 10) — P10 to P90
        if hasattr(point_forecast, "numpy"):
            point_forecast = point_forecast.numpy()
        if hasattr(quantile_forecast, "numpy"):
            quantile_forecast = quantile_forecast.numpy()

        # 展平
        point_forecast = np.array(point_forecast).flatten()
        quantile_forecast = np.array(quantile_forecast).flatten()  # (horizon, 10)

        # 计算预测收益
        if self.use_quantile and quantile_forecast.ndim == 2:
            # 加权分位数预测（更稳健）
            weights = np.array(self.quantile_weights).reshape(-1, 1)  # (5, 1)
            # 取中间5个分位数 P10,P30,P50,P70,P90
            q_mid = quantile_forecast[:, [0, 2, 4, 6, 8]]  # (horizon, 5)
            pred_return = np.dot(q_mid, weights.flatten()) / np.sum(weights)
            # pred_return 是差分序列的预测，直接累加到 current_price
            pred_price = current_price + pred_return
        else:
            # 点预测
            # point_forecast 是差分值
            pred_price = current_price + np.sum(point_forecast)

        # horizon 步后的预测价格
        pred_price = float(pred_price)
        pred_return_rate = (pred_price - current_price) / current_price

        timestamp = int(df.iloc[-1]["timestamp"]) if "timestamp" in df.columns else 0

        # 生成信号
        if pred_return_rate > self.buy_threshold:
            strength = min(abs(pred_return_rate) / self.buy_threshold, 1.0)
            signals.append(
                Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=timestamp,
                    price=float(current_price),
                    strength=float(strength),
                    reason=(
                        f"TimesFM forecast: +{pred_return_rate*100:.2f}% "
                        f"over {self.horizon} steps → BUY"
                    ),
                    metadata={
                        "pred_price": pred_price,
                        "pred_return": float(pred_return_rate),
                        "horizon": self.horizon,
                        "model": "timesfm-2.5-200m",
                        "quantile_used": self.use_quantile,
                        "quantile_forecast": (
                            quantile_forecast[:, 4].tolist() if self.use_quantile else None
                        ),
                    },
                )
            )
        elif pred_return_rate < self.sell_threshold:
            strength = min(abs(pred_return_rate) / abs(self.sell_threshold), 1.0)
            signals.append(
                Signal(
                    type=SignalType.SELL,
                    symbol=self.symbol,
                    timestamp=timestamp,
                    price=float(current_price),
                    strength=float(strength),
                    reason=(
                        f"TimesFM forecast: {pred_return_rate*100:.2f}% "
                        f"over {self.horizon} steps → SELL"
                    ),
                    metadata={
                        "pred_price": pred_price,
                        "pred_return": float(pred_return_rate),
                        "horizon": self.horizon,
                        "model": "timesfm-2.5-200m",
                        "quantile_used": self.use_quantile,
                        "quantile_forecast": (
                            quantile_forecast[:, 4].tolist() if self.use_quantile else None
                        ),
                    },
                )
            )

        return signals

    def _require_cols(self, df: pd.DataFrame, required: List[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def get_forecast(
        self, df: pd.DataFrame
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """只做预测不生成信号（用于分析）"""
        self._ensure_model()
        self._compile()

        context = self._get_context(df)
        current_price = df["close"].values[-1]

        point_forecast, quantile_forecast = self.model.forecast(
            horizon=self.horizon,
            inputs=[context],
        )

        if hasattr(point_forecast, "numpy"):
            point_forecast = point_forecast.numpy()
        if hasattr(quantile_forecast, "numpy"):
            quantile_forecast = quantile_forecast.numpy()

        point_forecast = np.array(point_forecast).flatten()
        quantile_forecast = np.array(quantile_forecast)

        if self.use_quantile and quantile_forecast.ndim == 2:
            weights = np.array(self.quantile_weights).reshape(-1, 1)
            q_mid = quantile_forecast[:, [0, 2, 4, 6, 8]]
            pred_return = np.dot(q_mid, weights.flatten()) / np.sum(weights)
            pred_price = current_price + pred_return
        else:
            pred_price = current_price + np.sum(point_forecast)

        pred_return_rate = (pred_price - current_price) / current_price

        return float(pred_return_rate), point_forecast, quantile_forecast
