"""
Stock grouping for transfer learning in DQN-based trading agents.

Adapted from deep-q-trading-agent (Jeong et al., 2019)
https://www.sciencedirect.com/science/article/abs/pii/S0957417418306134

Transfer learning addresses the problem of limited financial data for deep learning.
The approach:

1. Train an autoencoder (StonksNet) on component stock prices
2. Measure similarity between stocks using:
   - Pearson correlation with index
   - Autoencoder reconstruction MSE
3. Group stocks by similarity (high correlation, low correlation, mixed)
4. Pretrain DQN on groups before fine-tuning on index

This enables knowledge transfer from similar stocks to the index.

Note: This is a standalone implementation that does NOT require external data.
It provides the grouping logic and autoencoder, but actual stock data must be
provided by the caller.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from typing import List, Tuple, Dict, Optional

from .deep_q_networks import StonksNet


def train_autoencoder(
    prices: Tensor,
    epochs: int = 20,
    lr: float = 0.0001,
) -> StonksNet:
    """
    Train autoencoder on price time series.

    Args:
        prices:   Tensor of shape [num_days, num_components]
        epochs:  Number of training epochs
        lr:      Learning rate

    Returns:
        Trained StonksNet model
    """
    num_days, num_components = tuple(prices.size())
    model = StonksNet(size=num_components)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        losses = []
        for day in prices:
            optimizer.zero_grad()
            output = model(day)
            loss = criterion(output, day)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Autoencoder epoch {epoch+1}/{epochs}, avg loss: {np.mean(losses):.6f}")

    return model


def measure_correlation(
    index_prices: np.ndarray,
    component_prices: np.ndarray,
    component_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Measure Pearson correlation between each component and the index.

    Args:
        index_prices:      1D array of index prices
        component_prices:  2D array [num_components, num_days]
        component_names:   Optional list of component names

    Returns:
        Dict mapping component name/symbol to correlation coefficient
    """
    correlations = {}

    for i, comp_prices in enumerate(component_prices):
        name = component_names[i] if component_names else str(i)
        corr = np.corrcoef(index_prices, comp_prices)[0, 1]
        correlations[name] = corr if not np.isnan(corr) else 0.0

    return correlations


def measure_autoencoder_mse(
    model: StonksNet,
    prices: Tensor,
    component_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Measure autoencoder reconstruction MSE for each component.

    Args:
        model:            Trained StonksNet
        prices:           Tensor of shape [num_days, num_components]
        component_names:  Optional list of component names

    Returns:
        Dict mapping component name/symbol to MSE
    """
    with torch.no_grad():
        predicted_prices = model(prices)

    # Transpose: [num_days, num_components] -> [num_components, num_days]
    prices_t = prices.t()
    predicted_t = predicted_prices.t()

    mse_dict = {}
    for i, name in enumerate(component_names or range(len(prices_t))):
        mse = F.mse_loss(input=predicted_t[i], target=prices_t[i]).item()
        mse_dict[name] = mse

    return mse_dict


def create_groups(
    correlations: Dict[str, float],
    mse_values: Dict[str, float],
    group_size: int,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create stock groups based on correlation and MSE rankings.

    Creates 6 groups:
    - correlation/high:   Top 2n by correlation
    - correlation/low:    Bottom 2n by correlation
    - correlation/highlow: n high + n low by correlation
    - mse/high:           Top 2n by MSE (worst reconstruction)
    - mse/low:            Bottom 2n by MSE (best reconstruction)
    - mse/highlow:        n high + n low by MSE

    Args:
        correlations:  Dict[symbol -> correlation coefficient]
        mse_values:     Dict[symbol -> reconstruction MSE]
        group_size:     Size parameter n (actual group size = 2n)

    Returns:
        Nested dict: {method: {group_name: [symbols]}}
    """
    half = group_size // 2

    # Sort by correlation
    corr_sorted = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    corr_symbols = [s for s, _ in corr_sorted]

    # Sort by MSE
    mse_sorted = sorted(mse_values.items(), key=lambda x: x[1], reverse=True)
    mse_symbols = [s for s, _ in mse_sorted]

    groups = {
        "correlation": {
            "high": corr_symbols[:group_size],
            "low": corr_symbols[-group_size:],
            "highlow": corr_symbols[-half:] + corr_symbols[:half],
        },
        "mse": {
            "high": mse_symbols[:group_size],
            "low": mse_symbols[-group_size:],
            "highlow": mse_sorted[-half:] + mse_sorted[:half],
        },
    }

    # Fix highlow to return symbols not tuples
    groups["mse"]["highlow"] = [s for s, _ in groups["mse"]["highlow"]]

    return groups


class ConfusedMarketDetector:
    """
    Detects "confused market" conditions where the agent cannot make
    a robust decision.

    A confused market occurs when:
        |Q(s, a_BUY) - Q(s, a_SELL)| / sum|Q(s, a)| < threshold

    In this state, the agent falls back to a predetermined strategy
    (e.g., HOLD) to minimize potential losses from uncertain information.

    Reference: Jeong et al., 2019
    """

    def __init__(self, threshold: float = 0.0002):
        """
        Args:
            threshold: Confusion threshold (lower = more tolerant)
        """
        self.threshold = threshold

    def is_confused(self, q_values: np.ndarray) -> bool:
        """
        Determine if market is in a confused state.

        Args:
            q_values: Array of Q-values [Q_BUY, Q_HOLD, Q_SELL]

        Returns:
            True if market is confused (should use fallback strategy)
        """
        q_abs = np.abs(q_values)
        numerator = np.abs(q_values[0] - q_values[2])  # |Q_BUY - Q_SELL|
        denominator = np.sum(q_abs) + 1e-8  # sum|Q(s,a)|
        confidence = numerator / denominator

        return confidence < self.threshold

    def recommend_action(
        self,
        q_values: np.ndarray,
        fallback_action: int = 1,  # HOLD
        fallback_ratio: float = 0.5,
    ) -> Tuple[int, float, bool]:
        """
        Recommend action given Q-values, with confused market fallback.

        Args:
            q_values:         Array of Q-values [Q_BUY, Q_HOLD, Q_SELL]
            fallback_action:  Action to use in confused market (default HOLD)
            fallback_ratio:   Share ratio to use in confused market

        Returns:
            (action, share_ratio, is_confused)
        """
        is_confused = self.is_confused(q_values)

        if is_confused:
            return fallback_action, fallback_ratio, True

        # Normal case: pick action with highest Q-value
        action = int(np.argmax(q_values))

        # Share ratio is argmax-based for NumDReg variants
        return action, 0.5, False


def detect_market_regime(
    q_values_history: List[np.ndarray],
    confusion_threshold: float = 0.0002,
) -> str:
    """
    Detect overall market regime based on Q-value history.

    Args:
        q_values_history: List of Q-value arrays over time
        confusion_threshold: Threshold for confused market detection

    Returns:
        "CALM", "CHAOS", or "CRISIS"
    """
    if not q_values_history:
        return "CALM"

    confused_count = 0
    volatility = 0.0

    for q in q_values_history:
        numerator = np.abs(q[0] - q[2])
        denominator = np.sum(np.abs(q)) + 1e-8
        confidence = numerator / denominator

        if confidence < confusion_threshold:
            confused_count += 1

        volatility += np.std(q)

    confused_ratio = confused_count / len(q_values_history)
    avg_volatility = volatility / len(q_values_history)

    if confused_ratio > 0.5 or avg_volatility > 5.0:
        return "CRISIS"
    elif confused_ratio > 0.2 or avg_volatility > 2.0:
        return "CHAOS"
    else:
        return "CALM"
