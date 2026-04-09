"""
Policy classes for market-making agents.
Adapted from Market-Making-RL/MarketMaker/policy.py

Provides:
- BasePolicy: abstract policy interface
- GaussianPolicy: continuous action distribution (MultivariateNormal)
- CategoricalPolicy: discrete action distribution (Categorical)
- build_mlp: helper to construct MLP networks
- MaskedSequential: Sequential that propagates torch.masked.MaskedTensor

Compatible with stable-baselines3 PPO via the act() interface.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.masked as masked
import torch.distributions as ptd

# -------------------------------------------------------------------------- #
# Device
# -------------------------------------------------------------------------- #

_device = "cpu"
if torch.cuda.is_available():
    _device = "cuda"

def device():
    return _device


# -------------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------------- #

def np2torch(x, requires_grad: bool = False, cast_double_to_float: bool = True):
    """Convert numpy array (or MaskedArray) to torch tensor on the correct device."""
    if isinstance(x, np.ma.MaskedArray):
        mask = torch.from_numpy(~x.mask)
        x = torch.from_numpy(x.data)
        x = torch.masked.as_masked_tensor(x, mask)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.Tensor(x)

    if requires_grad:
        x = x.float()
        x.requires_grad = True
    elif cast_double_to_float and x.dtype == torch.float64:
        x = x.float()

    return x.to(_device)


def torch2np(x: torch.Tensor) -> np.ndarray:
    """Convert torch (Masked)Tensor back to numpy MaskedArray."""
    if isinstance(x, torch.masked.MaskedTensor):
        mask = ~x._masked_mask.detach().cpu().numpy()
        data = x._masked_data.detach().cpu().numpy()
        return np.ma.MaskedArray(data, mask=mask)
    return x.detach().cpu().numpy()


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to zero mean, unit variance."""
    return (x - x.mean()) / (x.std() + 1e-10)


# -------------------------------------------------------------------------- #
# Network builder
# -------------------------------------------------------------------------- #

class MaskedSequential(nn.Sequential):
    """nn.Sequential that propagates torch.masked.MaskedTensor through layers."""

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        is_masked = isinstance(x, masked.MaskedTensor)
        if is_masked:
            mask = x._masked_mask[..., :1]
            x = x._masked_data
            req_grad = x.requires_grad

        for module in self:
            x = module(x)

        if is_masked:
            newshape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
            mask = mask.repeat(*newshape)
            return masked.masked_tensor(x, mask, requires_grad=req_grad).to(_device)

        return x


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    hidden_size: int,
    activation: nn.Module = nn.ReLU(),
) -> MaskedSequential:
    """
    Build a multi-layer perceptron.

    Returns a MaskedSequential that handles both regular and masked tensors.
    """
    layers = [nn.Linear(input_size, hidden_size), activation]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)
    layers.append(nn.Linear(hidden_size, output_size))
    return MaskedSequential(*layers).to(_device)


# -------------------------------------------------------------------------- #
# Base Policy
# -------------------------------------------------------------------------- #

class BasePolicy:
    """
    Abstract policy interface.

    Subclasses must implement:
    - action_distribution(observations) -> Distribution
    """

    def action_distribution(self, observations: torch.Tensor):
        raise NotImplementedError

    def log_probs(
        self,
        distribution: ptd.Distribution,
        actions: torch.Tensor | masked.MaskedTensor,
    ) -> torch.Tensor | masked.MaskedTensor:
        """Return log probabilities of actions under the distribution."""
        if isinstance(actions, masked.MaskedTensor):
            data = torch.nan_to_num(actions._masked_data, 1).to(_device)
            mask = actions._masked_mask[..., 0].detach()
            req_grad = actions.requires_grad
            probs = distribution.log_prob(data).detach()
            return masked.masked_tensor(probs, mask, requires_grad=req_grad).to(_device)
        return distribution.log_prob(actions).to(_device)

    def entropy(
        self,
        distribution: ptd.Distribution,
        observations: torch.Tensor | masked.MaskedTensor,
    ) -> torch.Tensor | masked.MaskedTensor:
        """Return entropy of the action distribution."""
        entropy = distribution.entropy()
        if not isinstance(observations, masked.MaskedTensor):
            return entropy
        mask = observations._masked_mask[..., 0].detach()
        req_grad = observations.requires_grad
        return masked.masked_tensor(entropy.detach(), mask, requires_grad=req_grad).to(_device)

    def act(
        self,
        observations: np.ndarray,
        return_log_prob: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action from the policy given observations.

        Args:
            observations: np.ndarray of shape [batch, obs_dim]
            return_log_prob: if True, also return log probability of the action

        Returns:
            actions: np.ndarray of shape [batch, act_dim]
            log_probs (optional): np.ndarray of shape [batch]
        """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        actions = distribution.sample()
        sampled_actions = torch2np(actions)

        if return_log_prob:
            log_probs = torch2np(self.log_probs(distribution, actions))
            return sampled_actions, log_probs

        return sampled_actions


# -------------------------------------------------------------------------- #
# Gaussian Policy (continuous)
# -------------------------------------------------------------------------- #

class GaussianPolicy(BasePolicy, nn.Module):
    """
    Continuous Gaussian policy with state-dependent mean and diagonal covariance.

    Mean: network(observations)
    Std:  learnable log_std parameter (clamped positive via exp)

    Action distribution: MultivariateNormal(mean, scale_tril=diag(exp(log_std)))
    """

    def __init__(self, network: nn.Module, action_dim: int):
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(np2torch(np.zeros(action_dim)), requires_grad=False)

    def std(self) -> torch.Tensor:
        """Return std vector = exp(log_std)."""
        return torch.exp(self.log_std)

    def action_distribution(self, observations: torch.Tensor):
        """Build a MultivariateNormal with diagonal covariance."""
        means = self.network(observations)
        if isinstance(means, masked.MaskedTensor):
            means = means._masked_data.nan_to_num(0)
        scale = torch.diag(torch.exp(self.log_std))
        return ptd.MultivariateNormal(means, scale_tril=scale)


# -------------------------------------------------------------------------- #
# Categorical Policy (discrete)
# -------------------------------------------------------------------------- #

class CategoricalPolicy(BasePolicy, nn.Module):
    """
    Discrete categorical policy for discrete action spaces.

    Action distribution: Categorical(logits=network(observations))
    """

    def __init__(self, network: nn.Module):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations: torch.Tensor):
        """Build a Categorical distribution from logits."""
        vals = self.network(observations)
        if isinstance(vals, masked.MaskedTensor):
            vals = vals._masked_data.nan_to_num(0)
        return ptd.Categorical(logits=vals)


# -------------------------------------------------------------------------- #
# Baseline (Value) Network
# -------------------------------------------------------------------------- #

class BaselineNetwork(nn.Module):
    """
    Value-function baseline network for advantage estimation.

    Input:  observations of shape [batch, val_dim]
    Output: scalar value [batch]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_layers: int = 2,
        layer_size: int = 64,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.network = build_mlp(input_dim, output_dim, n_layers, layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        values = self.network(observations)
        if isinstance(values, masked.MaskedTensor):
            mask = values._masked_mask.squeeze()
            data = values._masked_data.squeeze()
            req_grad = values.requires_grad
            return masked.masked_tensor(data, mask, requires_grad=req_grad).to(_device)
        return values.squeeze()

    def calculate_advantage(self, returns: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Advantage = returns - baseline(observations)."""
        return returns - torch2np(self.forward(np2torch(observations)))

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray) -> float:
        """Fit baseline to returns via MSE loss."""
        returns_t = np2torch(returns, True)
        obs_t = np2torch(observations, True)
        baseline = self.forward(obs_t)
        self.optimizer.zero_grad()
        loss = torch.mean((baseline - returns_t) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# -------------------------------------------------------------------------- #
# PPO Policy wrapper (for use with stable-baselines3 or custom training)
# -------------------------------------------------------------------------- #

class PPOPolicy:
    """
    PPO policy wrapper that exposes act() for use with stable-baselines3.

    This is a lightweight wrapper — for SB3, use sb3.common.policies.ActorCriticPolicy
    directly. This class is provided for standalone training outside of SB3.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_layers: int = 2,
        layer_size: int = 64,
        lr: float = 1e-3,
        log_std_init: float = 0.0,
        device: str = "auto",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        actual_device = _device if device == "auto" else device
        self.network = build_mlp(obs_dim, act_dim, n_layers, layer_size).to(actual_device)
        self.actor = GaussianPolicy(self.network, act_dim).to(actual_device)
        self.actor.log_std.data = np2torch(np.full(act_dim, log_std_init))

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def act(self, observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Sample an action from the policy.

        Args:
            observations: np.ndarray of shape [obs_dim] or [batch, obs_dim]
            deterministic: if True, return mean action instead of sampling

        Returns:
            action: np.ndarray
        """
        obs = np.atleast_2d(observations)
        with torch.no_grad():
            obs_t = np2torch(obs)
            dist = self.actor.action_distribution(obs_t)
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.sample()
        return torch2np(actions).squeeze()

    def evaluate_actions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> tuple:
        """
        Evaluate log_probs and entropy for given (observations, actions).
        Used during PPO training.

        Returns:
            log_probs: np.ndarray
            entropy: float
        """
        obs_t = np2torch(observations, requires_grad=True)
        act_t = np2torch(actions, requires_grad=True)
        dist = self.actor.action_distribution(obs_t)
        log_probs = self.actor.log_probs(dist, act_t)
        entropy = torch.mean(self.actor.entropy(dist, obs_t))
        return torch2np(log_probs), float(torch2np(entropy).mean())
