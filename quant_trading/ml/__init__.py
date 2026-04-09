# -*- coding: utf-8 -*-
"""ML module — machine learning models for quant trading.

Exports:
    DPML: DualProcessVolumePredictor, MetaLearner (ECML PKDD 2022)
    Neural nets: CNN, TDNN, RNN, LSTM models from nn_models.py
    quant_ml: XGBoost, LSTM, RandomForest, FeatureEngineering, WalkForwardValidator, MLPipeline
"""

from __future__ import annotations

from .dpml import (
    DualProcessVolumePredictor,
    MetaLearner,
    System1Linear,
    System1LSTM,
    System2LSTM,
    System2Transformer,
)
from .model_comparison import ModelComparisonSuite
from .nn_models import (
    BaseModel,
    ARMAModel,
    CNN1DModel,
    TDNNModel,
    RNNModel,
    LSTMModel,
)
from .quant_ml import (
    XGBoostPredictor,
    LSTMPredictor,
    RandomForestClassifier,
    FeatureEngineering,
    WalkForwardValidator,
    MLPipeline,
)

__all__ = [
    # DPML (ECML PKDD 2022)
    "DualProcessVolumePredictor",
    "MetaLearner",
    "System1Linear",
    "System1LSTM",
    "System2LSTM",
    "System2Transformer",
    # Neural net models
    "BaseModel",
    "ARMAModel",
    "CNN1DModel",
    "TDNNModel",
    "RNNModel",
    "LSTMModel",
    # Comparison suite
    "ModelComparisonSuite",
    # quant_ml: ML pipeline (XGBoost/LSTM/RF + features + walk-forward)
    "XGBoostPredictor",
    "LSTMPredictor",
    "RandomForestClassifier",
    "FeatureEngineering",
    "WalkForwardValidator",
    "MLPipeline",
]
