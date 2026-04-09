"""
CrossAttentionActorCriticPolicy: Cross-attention actor-critic policy for multi-asset trading.

Adapted from MultiStockRLTrading (D:/Hive/Data/trading_repos/MultiStockRLTrading/).

Key features:
- Temporal attention: encodes each asset's price history into a fixed-length embedding
- Cross-asset attention: models inter-asset relationships via multi-head self-attention
- Shared encoder with separate policy and value heads

Compatible with stable-baselines3 PPO / A2C.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn
from torch.nn import functional as F

try:
    from gymnasium import spaces
except ImportError:
    import gym as spaces

from stable_baselines3.common.policies import ActorCriticPolicy


class AttentionPooling(nn.Module):
    """Single-head attention pooling over a sequence of tokens."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: (batch, seq_len, embed_dim)
        weights = th.softmax(self.score(x), dim=1)  # (batch, seq_len, 1)
        return th.sum(weights * x, dim=1)  # (batch, embed_dim)


class MultiAssetCrossAttentionNetwork(nn.Module):
    """
    Cross-attention encoder for multi-asset observations.

    Architecture
    ------------
    1. Feature projection: Linear(feature_dim -> embed_dim) + LayerNorm + GELU
    2. Temporal attention: MultiheadAttention over the window timesteps for each asset
       (assets are processed independently here, then pooled)
    3. Attention pooling: compresses each asset's window into a single embedding
    4. Cross-asset attention: MultiheadAttention between asset embeddings
       — this is the core inter-asset relationship modeling
    5. MLP: shared hidden layer, then separate policy and value heads

    Input shape : (batch, num_assets, window_size, feature_dim)
    Output      : (policy_latent, value_latent)
                  each of shape (batch, last_layer_dim_pi/vf)
    """

    def __init__(
        self,
        num_assets: int,
        window_size: int,
        feature_dim: int,
        last_layer_dim_pi: int,
        last_layer_dim_vf: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.num_assets = num_assets
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # Feature projection per asset-timestep token
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal attention: attend over the window for each (batch, asset) token stream
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.temporal_pool = AttentionPooling(embed_dim)

        # Cross-asset attention: self-attention over the asset embeddings
        self.asset_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.asset_norm = nn.LayerNorm(embed_dim)
        self.asset_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        flattened_dim = num_assets * embed_dim

        self.policy_net = nn.Sequential(
            nn.Linear(flattened_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, last_layer_dim_pi),
            nn.Tanh(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(flattened_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, last_layer_dim_vf),
            nn.Tanh(),
        )

    def _encode_assets(self, observations: th.Tensor) -> th.Tensor:
        batch_size, num_assets, window_size, feature_dim = observations.shape
        if num_assets != self.num_assets or window_size != self.window_size or feature_dim != self.feature_dim:
            raise ValueError(
                "Observation shape mismatch. "
                f"Expected (_, {self.num_assets}, {self.window_size}, {self.feature_dim}), "
                f"got {tuple(observations.shape)}"
            )

        # Project each (batch*asset, window, feature) token sequence
        temporal_tokens = observations.reshape(batch_size * num_assets, window_size, feature_dim)
        temporal_tokens = self.feature_projection(temporal_tokens)

        # Temporal self-attention
        temporal_context, _ = self.temporal_attention(temporal_tokens, temporal_tokens, temporal_tokens)
        temporal_tokens = self.temporal_norm(temporal_tokens + temporal_context)

        # Pool over time → one embedding per asset
        asset_tokens = self.temporal_pool(temporal_tokens).reshape(batch_size, num_assets, self.embed_dim)

        # Cross-asset attention
        cross_asset_context, _ = self.asset_attention(asset_tokens, asset_tokens, asset_tokens)
        asset_tokens = self.asset_norm(asset_tokens + cross_asset_context)
        asset_tokens = self.final_norm(asset_tokens + self.asset_mlp(asset_tokens))

        return asset_tokens.reshape(batch_size, -1)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        shared_latent = self._encode_assets(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(self._encode_assets(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(self._encode_assets(features))


class CrossAttentionActorCriticPolicy(ActorCriticPolicy):
    """
    Stable-baselines3 actor-critic policy using the MultiAssetCrossAttentionNetwork.

    Usage
    -----
    model = PPO(
        CrossAttentionActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs={
            "cross_attention_kwargs": {
                "embed_dim": 64,
                "num_heads": 4,
                "dropout": 0.1,
            }
        },
    )

    Note
    ----
    - ``observation_space.shape`` must be (num_assets, window_size, feature_dim).
    - The policy extracts ``num_assets`` from ``observation_space.shape[0]``,
      ``window_size`` from ``shape[1]``, and ``feature_dim`` from ``shape[2]``.
    - ``action_space.shape[0]`` is used for both policy and value last-layer dimensions.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ) -> None:
        self.cross_attention_kwargs = kwargs.pop(
            "cross_attention_kwargs",
            {"embed_dim": 64, "num_heads": 4, "dropout": 0.1},
        )
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization for the custom cross-attention weights
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MultiAssetCrossAttentionNetwork(
            num_assets=self.observation_space.shape[0],
            window_size=self.observation_space.shape[1],
            feature_dim=self.observation_space.shape[2],
            last_layer_dim_pi=self.action_space.shape[0],
            last_layer_dim_vf=self.action_space.shape[0],
            **self.cross_attention_kwargs,
        )
