"""策略加载与热更新"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Optional, Type

from quant_trading.strategy.base import BaseStrategy


BUILTIN_STRATEGIES: Dict[str, str] = {
    "trend_following": "quant_trading.strategy.classic.trend_following.TrendFollowingStrategy",
    "sma_cross": "quant_trading.strategy.classic.sma_cross.SMACrossStrategy",
    "mean_reversion": "quant_trading.strategy.classic.mean_reversion.MeanReversionStrategy",
    "grid_trading": "quant_trading.strategy.classic.grid_trading.GridTradingStrategy",
    "microstructure_rebound": "quant_trading.strategy.advanced.microstructure_rebound.MicrostructureReboundStrategy",
    "hmm_regime": "quant_trading.strategy.advanced.hmm_regime_strategy.HMMRegimeStrategy",
    "arbitrage": "quant_trading.strategy.advanced.arb_strategy.ArbitrageStrategy",
    "copula_pairs": "quant_trading.strategy.advanced.copula_pairs_strategy.CopulaPairsStrategyAdapter",
    "swing": "quant_trading.strategy.swing_strategy.SWINGStrategy",
    "triangular_arb": "quant_trading.strategy.triangular_arb.TriangularArbitrageStrategy",
    "chan_theory": "quant_trading.strategy.advanced.chan_theory_strategy.ChanTheoryStrategy",
    "elliott_wave": "quant_trading.strategy.advanced.elliott_wave_strategy.ElliottWaveStrategy",
    "harmonic": "quant_trading.strategy.advanced.harmonic_strategy.HarmonicStrategy",
    "kalman_pairs": "quant_trading.strategy.advanced.kalman_pairs_strategy.KalmanPairsStrategyAdapter",
    "hft_spread_capture": "quant_trading.strategy.advanced.hft_spread_capture_strategy.HFTSpreadCaptureStrategy",
    "hft_momentum": "quant_trading.strategy.advanced.hft_momentum_strategy.HFTMomentumStrategy",
    "hft_orderbook_imbalance": "quant_trading.strategy.advanced.hft_orderbook_imbalance_strategy.HFTOrderBookImbalanceStrategy",
    "hft_latency_arb": "quant_trading.strategy.advanced.hft_latency_arb_strategy.HFTLatencyArbStrategy",
    # Adaptive Multi-Regime Engine (absorbed from finclaw)
    "adaptive_regime": "quant_trading.strategy.advanced.adaptive_regime_engine.AdaptiveRegimeEngine",
}


@dataclass
class StrategyReloadState:
    module: ModuleType
    module_path: str
    mtime: float
    class_path: str


class StrategyLoader:
    """策略加载器，支持热更新"""

    def __init__(self) -> None:
        self._state: Optional[StrategyReloadState] = None

    def resolve_path(self, name: str) -> str:
        if name in BUILTIN_STRATEGIES:
            return BUILTIN_STRATEGIES[name]
        return name

    def load_class(self, class_path: str) -> Type[BaseStrategy]:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        strategy_cls = getattr(module, class_name)
        if not inspect.isclass(strategy_cls) or not issubclass(strategy_cls, BaseStrategy):
            raise TypeError(f"Invalid strategy class: {class_path}")
        self._state = StrategyReloadState(
            module=module,
            module_path=module.__file__ or "",
            mtime=self._get_mtime(module.__file__),
            class_path=class_path,
        )
        return strategy_cls

    def load_strategy(
        self,
        name_or_path: str,
        symbol: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseStrategy:
        class_path = self.resolve_path(name_or_path)
        strategy_cls = self.load_class(class_path)
        strategy = strategy_cls(symbol=symbol)
        if params:
            for key, value in params.items():
                if hasattr(strategy.params, key):
                    setattr(strategy.params, key, value)
        return strategy


    def reload_if_changed(self) -> Optional[Type[BaseStrategy]]:
        if not self._state or not self._state.module_path:
            return None
        mtime = self._get_mtime(self._state.module_path)
        if mtime <= self._state.mtime:
            return None
        module = importlib.reload(self._state.module)
        self._state = StrategyReloadState(
            module=module,
            module_path=module.__file__ or self._state.module_path,
            mtime=mtime,
            class_path=self._state.class_path,
        )
        module_name, class_name = self._state.class_path.rsplit(".", 1)
        strategy_cls = getattr(module, class_name)
        if not inspect.isclass(strategy_cls) or not issubclass(strategy_cls, BaseStrategy):
            raise TypeError(f"Invalid strategy class after reload: {self._state.class_path}")
        return strategy_cls

    @staticmethod
    def _get_mtime(module_path: Optional[str]) -> float:
        if not module_path or not os.path.exists(module_path):
            return 0.0
        return os.path.getmtime(module_path)
