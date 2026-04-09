"""
Arbitrage Spread Predictor using Machine Learning
=================================================

期现价差ML预测模块

使用机器学习模型预测期现价差:
- LSTM: 长短期记忆神经网络
- BP: 反向传播神经网络
- SVR: 支持向量回归

价差预测模型基于:
- 历史价差数据
- 现货价格特征
- 期货价格特征
- 时间窗口特征
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal, Dict
import warnings


class ArbitragePredictor:
    """
    期现套利价差预测器

    使用多种机器学习模型预测期货与现货之间的价差,
    支持LSTM、BP神经网络和SVR模型。
    """

    def __init__(
        self,
        spot_price_df: pd.DataFrame,
        futures_price_df: pd.DataFrame,
        price_col: str = 'price'
    ):
        """
        Args:
            spot_price_df: 现货价格数据
            futures_price_df: 期货价格数据
            price_col: 价格列名
        """
        self.spot_price_df = spot_price_df.copy()
        self.futures_price_df = futures_price_df.copy()
        self.price_col = price_col

        self._standardize_columns()
        self._compute_spread()

    def _standardize_columns(self):
        """标准化列名"""
        for df, name in [(self.spot_price_df, 'spot'), (self.futures_price_df, 'futures')]:
            if self.price_col not in df.columns:
                if 'Spot' in df.columns:
                    df.rename(columns={'Spot': 'price'}, inplace=True)
                elif 'Futures' in df.columns:
                    df.rename(columns={'Futures': 'price'}, inplace=True)
                elif 'Close' in df.columns:
                    df.rename(columns={'Close': 'price'}, inplace=True)
                elif len(df.columns) == 1:
                    df.columns = ['price']

    def _compute_spread(self):
        """计算价差序列"""
        min_len = min(len(self.spot_price_df), len(self.futures_price_df))
        self.spread = (
            self.futures_price_df['price'].iloc[:min_len].values -
            self.spot_price_df['price'].iloc[:min_len].values
        )
        self.spot_prices = self.spot_price_df['price'].iloc[:min_len].values
        self.futures_prices = self.futures_price_df['price'].iloc[:min_len].values

    def prepare_features(
        self,
        window_size: int = 30,
        num_features: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备LSTM/ML模型所需的特征数据

        特征包括:
        - 历史价差 (window_size个滞后)
        - 现货价格变动
        - 期货价格变动
        - 价差的移动统计量

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量, 如果为None则使用所有可用特征

        Returns:
            (X, y): 特征矩阵和目标向量
        """
        if num_features is None:
            num_features = 5

        features = []

        # 价差滞后特征
        for i in range(window_size, len(self.spread)):
            row = []

            # 历史价差滞后
            for j in range(window_size):
                row.append(self.spread[i - j - 1])

            # 当前现货价格
            row.append(self.spot_prices[i])

            # 当前期货价格
            row.append(self.futures_prices[i])

            # 价差的移动均值
            row.append(np.mean(self.spread[i - window_size:i]))

            # 价差的移动标准差
            row.append(np.std(self.spread[i - window_size:i]))

            features.append(row)

        X = np.array(features)

        # 目标: 预测下一时刻价差
        y = self.spread[window_size:]

        # 只取前num_features列
        if X.shape[1] > num_features:
            X = X[:, :num_features]

        return X, y

    def prepare_data_for_lstm(
        self,
        window_size: int = 30,
        num_features: Optional[int] = None
    ) -> np.ndarray:
        """
        准备LSTM模型所需的3D输入数据 [samples, timesteps, features]

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量

        Returns:
            X: 3D特征数组 [n_samples, window_size, n_features]
        """
        X, _ = self.prepare_features(window_size, num_features)

        # 重塑为LSTM所需的3D格式 [samples, timesteps, features]
        n_samples = X.shape[0]
        if num_features is None:
            num_features = X.shape[1]

        X_lstm = X.reshape(n_samples, window_size, num_features)

        return X_lstm

    # ============ LSTM Model ============

    def build_lstm_model(
        self,
        window_size: int = 30,
        num_features: int = 5,
        lstm_units: int = 50
    ) -> 'tf.keras.Model':
        """
        构建LSTM模型

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量
            lstm_units: LSTM单元数量

        Returns:
            Keras模型对象
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout

            model = Sequential([
                LSTM(lstm_units, return_sequences=True, input_shape=(window_size, num_features)),
                Dropout(0.2),
                LSTM(lstm_units // 2, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            return model

        except ImportError:
            raise ImportError("TensorFlow is required for LSTM model")

    def train_lstm_model(
        self,
        window_size: int = 30,
        num_features: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """
        训练LSTM模型

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例

        Returns:
            dict: 训练历史和模型
        """
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler

            X, y = self.prepare_features(window_size, num_features)

            # 归一化
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

            # 分割数据
            split = int(len(X_scaled) * (1 - validation_split))
            X_train, X_val = X_scaled[:split], X_scaled[split:]
            y_train, y_val = y_scaled[:split], y_scaled[split:]

            # 重塑为3D [samples, timesteps, features]
            X_train = X_train.reshape(-1, window_size, num_features)
            X_val = X_val.reshape(-1, window_size, num_features)

            # 构建和训练模型
            model = self.build_lstm_model(window_size, num_features)

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0
            )

            self.lstm_model = model
            self.lstm_scaler_X = scaler_X
            self.lstm_scaler_y = scaler_y

            return {
                'model': model,
                'history': history.history,
                'window_size': window_size,
                'num_features': num_features
            }

        except ImportError:
            return {'error': 'TensorFlow is required for LSTM model'}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测

        Args:
            X: 输入特征

        Returns:
            predictions: 预测值
        """
        if hasattr(self, 'lstm_model') and self.lstm_model is not None:
            return self.lstm_model.predict(X, verbose=0)
        else:
            raise RuntimeError("Model not trained. Call train_lstm_model() first.")

    # ============ BP Neural Network ============

    def train_bp_model(
        self,
        window_size: int = 30,
        num_features: int = 5,
        hidden_units: int = 64,
        epochs: int = 100,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        训练BP反向传播神经网络

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量
            hidden_units: 隐藏层单元数
            epochs: 训练轮数
            learning_rate: 学习率

        Returns:
            dict: 训练结果
        """
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.model_selection import train_test_split

            X, y = self.prepare_features(window_size, num_features)

            # 归一化
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

            # 分割数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, test_size=0.2, shuffle=False
            )

            # 构建BP网络
            model = MLPRegressor(
                hidden_layer_sizes=(hidden_units, hidden_units // 2),
                activation='relu',
                solver='adam',
                learning_rate_init=learning_rate,
                max_iter=epochs,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )

            model.fit(X_train, y_train)

            # 评估
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            self.bp_model = model
            self.bp_scaler_X = scaler_X
            self.bp_scaler_y = scaler_y

            return {
                'model': model,
                'train_r2': train_score,
                'val_r2': val_score,
                'window_size': window_size,
                'num_features': num_features
            }

        except ImportError:
            return {'error': 'sklearn is required for BP model'}

    def predict_bp(self, X: np.ndarray) -> np.ndarray:
        """使用BP模型预测"""
        if not hasattr(self, 'bp_model') or self.bp_model is None:
            raise RuntimeError("BP model not trained. Call train_bp_model() first.")

        X_scaled = self.bp_scaler_X.transform(X)
        return self.bp_scaler_y.inverse_transform(
            self.bp_model.predict(X_scaled).reshape(-1, 1)
        )

    # ============ SVR Model ============

    def train_svr_model(
        self,
        window_size: int = 30,
        num_features: int = 5,
        kernel: str = 'rbf',
        C: float = 1.0,
        epsilon: float = 0.1
    ) -> Dict:
        """
        训练SVR支持向量回归模型

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量
            kernel: 核函数 ('rbf', 'linear', 'poly')
            C: 正则化参数
            epsilon: epsilon-SVR的epsilon参数

        Returns:
            dict: 训练结果
        """
        try:
            from sklearn.svm import SVR
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.model_selection import train_test_split

            X, y = self.prepare_features(window_size, num_features)

            # 归一化
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

            # 分割数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, test_size=0.2, shuffle=False
            )

            # 构建SVR模型
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
            model.fit(X_train, y_train)

            # 评估
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            self.svr_model = model
            self.svr_scaler_X = scaler_X
            self.svr_scaler_y = scaler_y

            return {
                'model': model,
                'train_r2': train_score,
                'val_r2': val_score,
                'window_size': window_size,
                'num_features': num_features
            }

        except ImportError:
            return {'error': 'sklearn is required for SVR model'}

    def predict_svr(self, X: np.ndarray) -> np.ndarray:
        """使用SVR模型预测"""
        if not hasattr(self, 'svr_model') or self.svr_model is None:
            raise RuntimeError("SVR model not trained. Call train_svr_model() first.")

        X_scaled = self.svr_scaler_X.transform(X)
        return self.svr_scaler_y.inverse_transform(
            self.svr_model.predict(X_scaled).reshape(-1, 1)
        )

    # ============ Ensemble Prediction ============

    def train_ensemble(
        self,
        window_size: int = 30,
        num_features: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        训练集成模型 (LSTM + BP + SVR)

        Args:
            window_size: 时间窗口大小
            num_features: 特征数量
            weights: 各模型权重, 如 {'lstm': 0.4, 'bp': 0.3, 'svr': 0.3}

        Returns:
            dict: 各模型训练结果
        """
        results = {}

        # 训练各模型
        try:
            results['lstm'] = self.train_lstm_model(window_size, num_features)
        except Exception as e:
            results['lstm'] = {'error': str(e)}

        try:
            results['bp'] = self.train_bp_model(window_size, num_features)
        except Exception as e:
            results['bp'] = {'error': str(e)}

        try:
            results['svr'] = self.train_svr_model(window_size, num_features)
        except Exception as e:
            results['svr'] = {'error': str(e)}

        # 设置默认权重
        if weights is None:
            weights = {'lstm': 0.4, 'bp': 0.3, 'svr': 0.3}

        self.ensemble_weights = weights

        return results

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测

        Args:
            X: 输入特征

        Returns:
            predictions: 加权平均预测值
        """
        predictions = []
        weights = []

        if hasattr(self, 'lstm_model') and self.lstm_model is not None:
            pred = self.lstm_model.predict(X, verbose=0)
            if len(pred.shape) > 1:
                pred = pred.ravel()
            predictions.append(pred)
            weights.append(self.ensemble_weights.get('lstm', 0))

        if hasattr(self, 'bp_model') and self.bp_model is not None:
            pred = self.predict_bp(X)
            if len(pred.shape) > 1:
                pred = pred.ravel()
            predictions.append(pred)
            weights.append(self.ensemble_weights.get('bp', 0))

        if hasattr(self, 'svr_model') and self.svr_model is not None:
            pred = self.predict_svr(X)
            if len(pred.shape) > 1:
                pred = pred.ravel()
            predictions.append(pred)
            weights.append(self.ensemble_weights.get('svr', 0))

        if not predictions:
            raise RuntimeError("No trained models available")

        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]

        # 加权平均
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, w in zip(predictions, weights):
            ensemble_pred += w * pred

        return ensemble_pred

    # ============ Model Evaluation ============

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        评估预测性能

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            dict: 评估指标
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 方向准确率
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(true_direction == pred_direction)
        else:
            direction_accuracy = None

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy) if direction_accuracy else None
        }
