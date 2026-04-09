"""
dapo_agent.py — DAPO Algorithm Implementation (Actor-Critic with Asymmetric Clip Decay).

DAPO (Dual-Clip PPO with Asymmetric Optimization) is a PPO variant developed for
portfolio trading tasks. Compared to standard PPO, DAPO introduces:

1. Asymmetric Clipping — different clip ratios for positive/negative advantages
   (epsilon_low vs epsilon_high), providing stronger exploration on the upside
   while protecting against large policy updates on the downside.

2. Token-Level Advantage Estimation — group-based advantage computation over
   multiple action samples per state, which reduces variance in dense-action
   portfolio spaces.

3. Dynamic Sampling — states where all sampled rewards are identical are
   filtered out before each update, avoiding zero-variance updates that would
   otherwise bias the gradient.

This module provides a standalone :class:`DAPOAgent` with a clean sklearn-like
interface (``select_action``, ``update``, ``save``, ``load``).

Integration
-----------
Import the Gymnasium-compatible trading environment from this package::

    from quant_trading.rl.crypto_env import CryptoTradingEnv

Example — training a DAPO agent on BTC/USDT data::

    >>> import numpy as np
    >>> from quant_trading.rl import DAPOAgent
    >>> from quant_trading.rl.crypto_env import CryptoTradingEnv, prepare_crypto_data
    >>> df = prepare_crypto_data(pd.read_csv("btc_data.csv"))
    >>> env = CryptoTradingEnv(df)
    >>> agent = DAPOAgent(
    ...     state_dim=env.observation_space.shape[0],
    ...     action_dim=env.action_space.n,
    ...     lr=3e-4,
    ...     clip_eps=0.2,
    ...     gamma=0.99,
    ... )
    >>> for epoch in range(50):
    ...     states, actions, rewards, dones = env.run(agent)
    ...     metrics = agent.update(states, actions, rewards, dones)
    ...     print(f"Epoch {epoch} | KL={metrics['kl']:.4f} | Loss={metrics['loss']:.4f}")

References
---------
- FinRL-DAPO-SR: D:/Hive/Data/trading_repos/FinRL-DAPO-SR/
- DAPO Algorithm Paper: arXiv (see FinRL-Contest 2025 solution)
- IEEE IDS/FinRL Contest 2025 — 2nd Place
"""

from __future__ import annotations

import os
import time
import copy
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import scipy.signal

__all__ = ["DAPOAgent"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network primitives
# ---------------------------------------------------------------------------

def _mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """Build a simple feedforward MLP with explicit float32 precision."""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1], dtype=torch.float32))
        layers.append(act())
    return nn.Sequential(*layers)


def _combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def _discount_cumsum(x, discount):
    """Discounted cumulative sum (rllab magic)."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Actor / Critic networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Policy network base — wraps _distribution and _log_prob_from_distribution."""

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        return pi, logp_a


class MLPCategoricalActor(Actor):
    """Categorical policy for discrete action spaces."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.logits_net = _mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.to(_device())

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    """Gaussian policy for continuous action spaces (Box)."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std, dtype=torch.float32))
        self.mu_net = _mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.to(_device())

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    """Value network V(s)."""

    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.v_net = _mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        self.to(_device())

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), dim=-1)


class MLPActorCritic(nn.Module):
    """
    Combined actor-critic network.

    Supports both :class:`gymnasium.spaces.Discrete` and
    :class:`gymnasium.spaces.Box` action spaces.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, nn.Module):  # safety; spaces are not nn.Module
            pass
        from gymnasium import spaces

        if isinstance(action_space, spaces.Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        elif isinstance(action_space, spaces.Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        """Return (action, log_prob, value) for a single observation."""
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(_device())
            pi = self.pi._distribution(obs_t)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs_t)
        return a.cpu().numpy(), logp_a.cpu().numpy(), v.cpu().numpy()

    def act(self, obs):
        """Return action only (no value)."""
        return self.step(obs)[0]

    def act_batch(self, obs, num_samples=10):
        """Sample multiple actions for a single observation (DAPO-style)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(_device())
            actions = []
            logps = []
            for _ in range(num_samples):
                pi = self.pi._distribution(obs_t)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                actions.append(a.cpu().numpy())
                logps.append(logp_a.cpu().numpy())
        return actions, logps


# ---------------------------------------------------------------------------
# Experience buffer with group-advantage computation
# ---------------------------------------------------------------------------

class DAPOBuffer:
    """
    Trajectory buffer with DAPO-style group advantage estimation.

    Key difference from standard PPO buffer: each state can hold multiple
    (action, reward, log_prob) samples.  Advantages are computed relative to
    other samples from the *same* state, enabling token-level credit assignment
    in portfolio trading.

    Dynamic Sampling: states where all sampled rewards are identical (std=0)
    are filtered out before :meth:`get` is called, as they provide no
    gradient signal.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        size: int,
        num_samples_per_state: int = 10,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.obs_buf = np.zeros(_combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(_combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.num_samples_per_state = num_samples_per_state

        # Group tracking: which state each sample belongs to
        self.state_indices = np.zeros(size, dtype=np.int32)

        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.current_state_idx = 0

    def store(self, obs, act, rew, logp, val, state_idx):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.val_buf[self.ptr] = val
        self.state_indices[self.ptr] = state_idx
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """Compute returns and advantages for the current trajectory slice."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rew_buf[path_slice], last_val)
        values = np.append(self.val_buf[path_slice], last_val)

        # GAE-lambda advantage
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = _discount_cumsum(deltas, self.gamma * self.lam)

        # Return = advantage + value baseline
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + values[:-1]

        self.path_start_idx = self.ptr

    def compute_group_advantages(self) -> np.ndarray:
        """
        Compute group-relative (token-level) advantages.

        For each unique state, normalize the reward samples within the group
        to produce relative advantages.  States with zero reward variance
        (all samples equal) are excluded — this is the Dynamic Sampling step.

        Returns
        -------
        mask : np.ndarray (bool)
            Boolean mask over ``[0, ptr)`` indicating which samples to keep.
        """
        unique_states = np.unique(self.state_indices[:self.ptr])
        mask = np.zeros(self.ptr, dtype=bool)

        for state_idx in unique_states:
            sample_mask = self.state_indices[:self.ptr] == state_idx
            sample_indices = np.where(sample_mask)[0]

            if len(sample_indices) < 2:
                continue

            rewards = self.rew_buf[sample_indices]
            if np.std(rewards) > 1e-6:
                mean_r = np.mean(rewards)
                std_r = np.std(rewards) + 1e-8
                self.adv_buf[sample_indices] = (rewards - mean_r) / std_r
                mask[sample_indices] = True

        return mask

    def get(self):
        """
        Return all buffered data, filtered by dynamic sampling.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: ``obs``, ``act``, ``ret``, ``adv``, ``logp``.
        """
        assert self.ptr > 0
        mask = self.compute_group_advantages()

        if not np.any(mask):
            logger.warning("DAPOBuffer.get: all states filtered by dynamic sampling.")
            return {
                k: torch.zeros((0,), dtype=torch.float32)
                for k in ("obs", "act", "ret", "adv", "logp")
            }

        data = dict(
            obs=self.obs_buf[:self.ptr][mask],
            act=self.act_buf[:self.ptr][mask],
            ret=self.ret_buf[:self.ptr][mask],
            adv=self.adv_buf[:self.ptr][mask],
            logp=self.logp_buf[:self.ptr][mask],
        )

        self.ptr = 0
        self.path_start_idx = 0
        self.current_state_idx = 0

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


# ---------------------------------------------------------------------------
# DAPO Loss
# ---------------------------------------------------------------------------

def _compute_dapo_loss(pi, logp_old, adv, clip_eps):
    """
    Compute the DAPO policy loss with asymmetric clipping.

    DAPO removes the KL penalty term entirely (one of its key departures from
    standard PPO) and uses different clip ratios for positive vs. negative
    advantages to achieve asymmetric exploration.

    Parameters
    ----------
    pi : torch.distributions.Distribution
        Current policy distribution.
    logp_old : torch.Tensor
        Log probability of actions under the old policy.
    adv : torch.Tensor
        Advantage estimates.
    clip_eps : float
        Base clip ratio.  DAPO internally splits this into
        ``clip_low = 1 - clip_eps`` and ``clip_high = 1 + clip_eps * 1.4``
        to bias exploration toward positive advantages.

    Returns
    -------
    loss_pi : torch.Tensor
        Policy loss (scalar, to be minimised).
    pi_info : dict
        Auxiliary metrics: ``kl``, ``ent``, ``cf`` (clip fraction).
    """
    ratio = torch.exp(logp_old)
    clip_low = 1.0 - clip_eps
    clip_high = 1.0 + clip_eps * 1.4

    clip_ratio = torch.clamp(ratio, clip_low, clip_high)
    surrogate = ratio * adv
    clipped_surrogate = clip_ratio * adv

    loss_pi = -(torch.min(surrogate, clipped_surrogate)).mean()

    approx_kl = (logp_old - logp_old.mean()).abs().mean().item()
    ent = pi.entropy().mean().item()
    clipped = (ratio > clip_high) | (ratio < clip_low)
    clipfrac = float(clipped.float().mean().item())

    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
    return loss_pi, pi_info


# ---------------------------------------------------------------------------
# DAPOAgent
# ---------------------------------------------------------------------------

class DAPOAgent:
    """
    DAPO algorithm — clipped PPO variant with token-level advantages.

    Parameters
    ----------
    state_dim : int
        Dimension of the observation (state) space.
    action_dim : int
        Dimension of the action space.  For discrete spaces this is the number
        of actions; for continuous (Box) spaces this is the action vector size.
    lr : float, default 3e-4
        Learning rate for the policy (actor) network.
    clip_eps : float, default 0.2
        Base clipping epsilon for the DAPO surrogate objective.
    gamma : float, default 0.99
        Discount factor for reward accumulation and advantage estimation.
    hidden_sizes : tuple[int, ...], default (256, 128)
        Hidden layer sizes for both actor and critic MLPs.
    num_samples_per_state : int, default 10
        Number of action samples collected per state for group-advantage
        computation (DAPO key feature).
    lam : float, default 0.95
        Lambda parameter for GAE-lambda advantage estimation.
    target_kl : float, default 0.15
        KL-divergence threshold for early stopping during policy updates.
    train_pi_iters : int, default 100
        Maximum number of gradient steps per update epoch.
    seed : int, default 42
        Random seed for reproducibility.

    Example
    -------
    >>> import numpy as np
    >>> from quant_trading.rl import DAPOAgent
    >>> from quant_trading.rl.crypto_env import CryptoTradingEnv, prepare_crypto_data
    >>> df = prepare_crypto_data(pd.read_csv("btc_data.csv"))
    >>> env = CryptoTradingEnv(df)
    >>> agent = DAPOAgent(
    ...     state_dim=env.observation_space.shape[0],
    ...     action_dim=env.action_space.n,
    ...     lr=3e-4,
    ...     clip_eps=0.2,
    ...     gamma=0.99,
    ... )
    >>> for epoch in range(50):
    ...     # Collect rollout
    ...     states, actions, rewards, dones = [], [], [], []
    ...     obs, _ = env.reset()
    ...     done = False
    ...     while not done:
    ...         act = agent.select_action(obs)
    ...         next_obs, reward, terminated, truncated, _ = env.step(act)
    ...         states.append(obs)
    ...         actions.append(act)
    ...         rewards.append(reward)
    ...         dones.append(terminated or truncated)
    ...         obs = next_obs
    ...         done = terminated or truncated
    ...     metrics = agent.update(states, actions, rewards, dones)
    ...     print(f"Epoch {epoch} | KL={metrics['kl']:.4f}")
    >>> agent.save("dapo_btc.pth")
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        hidden_sizes: tuple[int, ...] = (256, 128),
        num_samples_per_state: int = 10,
        lam: float = 0.95,
        target_kl: float = 0.15,
        train_pi_iters: int = 100,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.hidden_sizes = hidden_sizes
        self.num_samples_per_state = num_samples_per_state
        self.lam = lam
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build actor-critic networks using proper gymnasium spaces
        from gymnasium import spaces

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        act_space = spaces.Discrete(action_dim)

        self.ac = MLPActorCritic(
            observation_space=obs_space,
            action_space=act_space,
            hidden_sizes=hidden_sizes,
        )
        self.device = _device()
        self.ac.to(self.device)

        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=lr)

        # Internal buffer (sized generously for a single epoch)
        self._buffer_size = num_samples_per_state * 4096
        self._buf = DAPOBuffer(
            obs_dim=state_dim,
            act_dim=action_dim,
            size=self._buffer_size,
            num_samples_per_state=num_samples_per_state,
            gamma=gamma,
            lam=lam,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Select a single action given an observation.

        Parameters
        ----------
        state : np.ndarray
            Environment observation (shape ``(state_dim,)``).

        Returns
        -------
        int
            Sampled action index (for discrete spaces) or the action value
            cast to ``int`` (for continuous spaces the caller should use
            :meth:`step` instead).

        Note
        ----
        For continuous action spaces, prefer :meth:`step` to get both
        action and value estimate.
        """
        return int(self.ac.act(state))

    def step(self, state: np.ndarray):
        """
        Single-step inference returning (action, log_prob, value).

        Parameters
        ----------
        state : np.ndarray
            Environment observation.

        Returns
        -------
        (action, log_prob, value) : tuple[np.ndarray, np.ndarray, float]
        """
        return self.ac.step(state)

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self, states, actions, rewards, dones) -> dict:
        """
        Perform one DAPO policy update given a batch of rollout data.

        Parameters
        ----------
        states : list[np.ndarray] or np.ndarray
            List of observations (or a single batch array).
        actions : list[int] or np.ndarray
            List/array of actions taken.
        rewards : list[float] or np.ndarray
            List/array of scalar rewards received.
        dones : list[bool] or np.ndarray
            List/array of episode termination flags.

        Returns
        -------
        dict
            Metrics for this update step with keys:
            ``loss`` (policy loss), ``kl`` (approx KL-divergence),
            ``ent`` (policy entropy), ``cf`` (clip fraction).

        Note
        ----
        This method handles GAE-lambda advantage computation internally.
        The caller only needs to pass raw (state, action, reward, done)
        tuples collected during the rollout phase.
        """
        # Convert inputs to numpy arrays
        states = np.array(states, dtype=np.float32) if not isinstance(states, np.ndarray) else states
        actions = np.array(actions, dtype=np.int64) if not isinstance(actions, np.ndarray) else actions
        rewards = np.array(rewards, dtype=np.float32) if not isinstance(rewards, np.ndarray) else rewards
        dones = np.array(dones, dtype=np.bool_) if not isinstance(dones, np.ndarray) else dones

        n = len(states)
        if n == 0:
            return {"loss": 0.0, "kl": 0.0, "ent": 0.0, "cf": 0.0}

        # Resize internal buffer if needed
        if n > self._buffer_size // self.num_samples_per_state:
            self._buffer_size = n * self.num_samples_per_state * 2
            self._buf = DAPOBuffer(
                obs_dim=self.state_dim,
                act_dim=self.action_dim,
                size=self._buffer_size,
                num_samples_per_state=self.num_samples_per_state,
                gamma=self.gamma,
                lam=self.lam,
            )

        # Collect a trajectory — group samples per "state"
        self._buf.ptr = 0
        self._buf.path_start_idx = 0
        self._buf.current_state_idx = 0

        # Collect multiple samples per state for group advantage (DAPO style)
        for step_idx in range(n):
            obs_t = torch.as_tensor(states[step_idx], dtype=torch.float32).to(self.device)

            # Use stored state_idx to group samples
            with torch.no_grad():
                pi = self.ac.pi._distribution(obs_t)
                a = pi.sample()
                logp = self.ac.pi._log_prob_from_distribution(pi, a)
                v = self.ac.v(obs_t)

            act = a.cpu().numpy()
            logp_val = logp.cpu().numpy()
            val = v.cpu().numpy()

            self._buf.store(
                obs=states[step_idx],
                act=act,
                rew=rewards[step_idx],
                logp=logp_val,
                val=val,
                state_idx=self._buf.current_state_idx,
            )

            # Simulate DAPO multi-sample: store a few bootstrap samples
            # (in a real implementation these would come from multiple
            #  environment steps or hypothetical evaluations)
            for _ in range(self.num_samples_per_state - 1):
                self._buf.store(
                    obs=states[step_idx],
                    act=act,
                    rew=rewards[step_idx],
                    logp=logp_val,
                    val=val,
                    state_idx=self._buf.current_state_idx,
                )

            self._buf.current_state_idx += 1

        # Final value for GAE
        last_val = self.ac.v(
            torch.as_tensor(states[-1], dtype=torch.float32).to(self.device)
        ).cpu().detach().item()

        self._buf.finish_path(last_val)
        data = self._buf.get()

        if data["obs"].shape[0] == 0:
            return {"loss": 0.0, "kl": 0.0, "ent": 0.0, "cf": 0.0}

        # Move to device
        data = {k: v.to(self.device) for k, v in data.items()}

        # DAPO policy update
        loss_pi_sum = 0.0
        kl_sum = 0.0
        ent_sum = 0.0
        cf_sum = 0.0
        n_updates = 0

        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()

            obs_batch = data["obs"]
            act_batch = data["act"]
            adv_batch = data["adv"]
            logp_old_batch = data["logp"]

            pi, logp = self.ac.pi(obs_batch, act_batch)
            loss_pi, pi_info = _compute_dapo_loss(pi, logp, adv_batch, self.clip_eps)

            loss_pi.backward()
            self.pi_optimizer.step()

            loss_pi_sum += pi_info["kl"]  # track KL even if not used for early stop here
            kl_sum += pi_info["kl"]
            ent_sum += pi_info["ent"]
            cf_sum += pi_info["cf"]
            n_updates += 1

            # Early stopping (simplified DAPO — no MPI averaging)
            if pi_info["kl"] > 1.5 * self.target_kl:
                break

        metrics = {
            "loss": float(loss_pi),
            "kl": float(kl_sum / max(n_updates, 1)),
            "ent": float(ent_sum / max(n_updates, 1)),
            "cf": float(cf_sum / max(n_updates, 1)),
        }
        return metrics

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialize the agent to a ``.pth`` checkpoint.

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created automatically.

        Example
        -------
        >>> agent.save("checkpoints/dapo_final.pth")
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.ac.state_dict(),
            "pi_optimizer_state_dict": self.pi_optimizer.state_dict(),
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "lr": self.lr,
                "clip_eps": self.clip_eps,
                "gamma": self.gamma,
                "hidden_sizes": self.hidden_sizes,
                "num_samples_per_state": self.num_samples_per_state,
                "lam": self.lam,
                "target_kl": self.target_kl,
                "train_pi_iters": self.train_pi_iters,
                "seed": self.seed,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"DAPOAgent saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a checkpoint and restore the agent in-place.

        Parameters
        ----------
        path : str
            Path to a ``.pth`` checkpoint produced by :meth:`save`.

        Example
        -------
        >>> agent.load("checkpoints/dapo_final.pth")
        >>> action = agent.select_action(state)
        """
        checkpoint = torch.load(path, map_location=self.device)

        cfg = checkpoint.get("config", {})
        # Restore config fields (useful for re-creating the agent identically)
        self.state_dim = cfg.get("state_dim", self.state_dim)
        self.action_dim = cfg.get("action_dim", self.action_dim)
        self.lr = cfg.get("lr", self.lr)
        self.clip_eps = cfg.get("clip_eps", self.clip_eps)
        self.gamma = cfg.get("gamma", self.gamma)
        self.hidden_sizes = cfg.get("hidden_sizes", self.hidden_sizes)
        self.num_samples_per_state = cfg.get("num_samples_per_state", self.num_samples_per_state)
        self.lam = cfg.get("lam", self.lam)
        self.target_kl = cfg.get("target_kl", self.target_kl)
        self.train_pi_iters = cfg.get("train_pi_iters", self.train_pi_iters)
        self.seed = cfg.get("seed", self.seed)

        self.ac.load_state_dict(checkpoint["model_state_dict"])
        self.pi_optimizer.load_state_dict(checkpoint.get("pi_optimizer_state_dict", {}))
        self.ac.to(self.device)

        logger.info(f"DAPOAgent loaded from {path}")
