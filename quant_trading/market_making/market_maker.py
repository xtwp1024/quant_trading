"""
Market-maker training classes.
Adapted from Market-Making-RL/MarketMaker/marketmaker.py

Provides three market-maker variants:

1. UniformMarketMaker:  fixed-length trajectories (all same nt)
2. MaskedMarketMaker:    variable-length trajectories with np.ma.MaskedArray
3. MarketMaker:         sparse (list-of-dicts) variable-length trajectories

All three support:
- Custom PPO update via the PolicyGradient interface
- stable-baselines3 PPO-compatible act() method
- Avellaneda-Stoikov reward framework

Usage with stable-baselines3:
------------------------------
    from stable_baselines3 import PPO
    from quant_trading.market_making import MarketEnv

    env = MarketEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

Usage with custom training:
---------------------------
    from quant_trading.market_making import MarketMaker, MarketEnv

    env = MarketEnv()
    mm = MarketMaker(env)
    mm.train(epochs=100, batch_size=64)
"""

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from quant_trading.market_making.order_book import OrderBook
from quant_trading.market_making.market_env import BaseMarket, MarketEnv
from quant_trading.market_making.rewards import AvellanedaStoikovReward, make_reward_function
from quant_trading.market_making.policies import (
    BasePolicy,
    GaussianPolicy,
    CategoricalPolicy,
    BaselineNetwork,
    build_mlp,
    np2torch,
    torch2np,
    device,
    normalize,
)


# -------------------------------------------------------------------------- #
# Trajectory collectors
# -------------------------------------------------------------------------- #

class UniformCollector:
    """
    Collects fixed-length (nt) trajectories from BaseMarket.

    All trajectories have the same length; unused slots are zeros.
    """

    def __init__(
        self,
        market: BaseMarket,
        policy: BasePolicy,
        reward_fn: AvellanedaStoikovReward,
        dt: float,
        max_t: float,
        nt: int,
        gamma: float = 0.9999,
        do_ppo: bool = True,
        track_all: bool = False,
    ):
        self.market = market
        self.policy = policy
        self.reward_fn = reward_fn
        self.dt = dt
        self.max_t = max_t
        self.nt = nt
        self.gamma = gamma
        self.do_ppo = do_ppo
        self.track_all = track_all

    def collect(self, nbatch: int) -> Dict[str, np.ndarray]:
        """
        Collect `nbatch` trajectories.

        Returns:
            dict with keys: tra (trajectories), obs (observations),
                            act (actions), rew (rewards), val (final values),
                            and optionally wea (wealth), inv (inventory)
        """
        dt = self.dt
        nt = self.nt
        T = self.max_t

        val_dim = 6   # n_bid, delta_b, n_ask, delta_a, dW, dI
        act_dim = 4   # n_bid, delta_b, n_ask, delta_a
        obs_dim = 4   # n_bid, delta_b, n_ask, delta_a

        trajectories = np.zeros((nbatch, nt, val_dim))
        observations = np.zeros((nbatch, nt, obs_dim))
        actions = np.zeros((nbatch, nt, act_dim))
        rewards = np.zeros((nbatch, nt))
        values = np.zeros((nbatch,))

        wealth_all = np.zeros((nbatch, nt)) if self.track_all else None
        inv_all = np.zeros((nbatch, nt), dtype=int) if self.track_all else None

        for b in range(nbatch):
            self.market.reset()
            W = self.market.W
            I = self.market.I
            midprice = self.market.book.midprice

            for t in range(nt):
                time_left = T - t * dt
                state = self.market.state()
                obs = np.array(state, dtype=np.float32)
                observations[b, t] = obs

                # Get action from policy
                action, logprob = self.policy.act(obs, return_log_prob=True)
                actions[b, t] = action
                self.market.submit(*action)

                # Market step
                dW, dI, midprice = self.market.step()

                W += dW
                I += dI

                # Reward
                rew = self.reward_fn.reward(dW, dI, time_left)
                rewards[b, t] = rew

                trajectories[b, t] = np.array(
                    [obs[0], obs[1], obs[2], obs[3], dW, dI], dtype=np.float32
                )

                if self.track_all:
                    wealth_all[b, t] = W
                    inv_all[b, t] = I

            # Final reward
            final_val = self.reward_fn.final_reward(W, I, midprice)
            rewards[b, -1] += final_val
            values[b] = W + I * midprice

        out = {
            "tra": trajectories,
            "obs": observations,
            "act": actions,
            "rew": rewards,
            "val": values,
        }
        if self.track_all:
            out["wea"] = wealth_all
            out["inv"] = inv_all

        return out


class MaskedCollector:
    """
    Collects variable-length trajectories using np.ma.MaskedArray.
    Trajectories that terminate early are masked out.
    """

    def __init__(
        self,
        market: BaseMarket,
        policy: BasePolicy,
        reward_fn: AvellanedaStoikovReward,
        dt: float,
        max_t: float,
        nt: int,
        gamma: float = 0.9999,
        do_ppo: bool = True,
        track_all: bool = False,
    ):
        self.market = market
        self.policy = policy
        self.reward_fn = reward_fn
        self.dt = dt
        self.max_t = max_t
        self.nt = nt
        self.gamma = gamma
        self.do_ppo = do_ppo
        self.track_all = track_all

    def collect(self, nbatch: int) -> Dict[str, np.ndarray]:
        nt = self.nt
        dt = self.dt
        T = self.max_t

        val_dim = 6
        act_dim = 4
        obs_dim = 4

        trajectories = np.ma.empty((nbatch, nt, val_dim))
        trajectories.mask = True
        actions = np.ma.empty((nbatch, nt, act_dim))
        actions.mask = True
        rewards = np.ma.empty((nbatch, nt))
        rewards.mask = True
        logprobs = np.ma.empty((nbatch, nt))
        logprobs.mask = True

        wealth_all = np.ma.empty((nbatch, nt)) if self.track_all else None
        inv_all = np.ma.empty((nbatch, nt)) if self.track_all else None
        if self.track_all:
            wealth_all.mask = True
            inv_all.mask = True

        avg_len = 0.0

        for b in range(nbatch):
            self.market.reset()
            W = self.market.W
            I = self.market.I
            midprice = self.market.book.midprice
            terminated = False

            for t in range(nt):
                time_left = T - t * dt
                state = self.market.state()
                obs = np.array(state, dtype=np.float32)

                action, logprob = self.policy.act(obs, return_log_prob=True)
                self.market.submit(*action)

                dW, dI, midprice = self.market.step()
                W += dW
                I += dI

                rew = self.reward_fn.reward(dW, dI, time_left)

                logprobs[b, t] = logprob
                actions[b, t] = action
                rewards[b, t] = rew
                trajectories[b, t] = np.array([obs[0], obs[1], obs[2], obs[3], dW, dI], dtype=np.float32)

                if self.track_all:
                    wealth_all[b, t] = W
                    inv_all[b, t] = I

                if self.market.is_empty():
                    terminated = True
                    avg_len += t + 1
                    break

            if not terminated:
                # Terminal reward
                final_val = self.reward_fn.final_reward(W, I, midprice)
                rewards[b, t] += final_val
                avg_len += t + 1

        avg_len /= nbatch

        out = {
            "tra": trajectories,
            "obs": trajectories[..., :obs_dim],
            "act": actions,
            "rew": rewards,
            "old": logprobs,
            "val": W + I * midprice,
        }
        if self.track_all:
            out["wea"] = wealth_all
            out["inv"] = inv_all

        return out


# -------------------------------------------------------------------------- #
# Return calculators
# -------------------------------------------------------------------------- #

def compute_mc_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute Monte-Carlo (cumulative) returns from rewards.

    G_t = sum_{k=t}^{T} gamma^{k-t} * r_k
    """
    returns = np.empty_like(rewards)
    returns[:, -1] = rewards[:, -1]
    for t in reversed(range(rewards.shape[1] - 1)):
        returns[:, t] = rewards[:, t] + discount * returns[:, t + 1]
    return returns


def compute_td_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    discount: float,
    lambd: float = 0.9,
) -> np.ndarray:
    """
    Compute TD(lambda) returns.

    G_t^lambda = r_t + gamma * ((1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}^lambda)
    """
    returns = np.empty_like(rewards)
    returns[:, -1] = rewards[:, -1] + discount * values[:, -1]
    for t in reversed(range(rewards.shape[1] - 1)):
        returns[:, t] = (
            rewards[:, t]
            + discount * ((1 - lambd) * values[:, t + 1] + lambd * returns[:, t + 1])
        )
    return returns


# -------------------------------------------------------------------------- #
# Market Maker base
# -------------------------------------------------------------------------- #

class MarketMakerBase:
    """
    Base class for market-making agents.

    Provides:
    - collect()        : sample trajectories
    - compute_returns(): compute returns from rewards
    - update_policy()  : PPO-style policy gradient update

    Subclasses should override collect() if they need masked/sparse trajectories.
    """

    def __init__(
        self,
        env: MarketEnv,
        policy: Optional[BasePolicy] = None,
        reward_fn: Optional[AvellanedaStoikovReward] = None,
        gamma: float = 0.9999,
        lr: float = 1e-3,
        eps_clip: float = 0.2,
        entropy_coef: float = 0.0,
        do_clip: bool = True,
        use_baseline: bool = True,
        normalize_advantages: bool = True,
        trajectory: str = "MC",   # "MC" or "TD"
        lambd: float = 0.9,
        n_obs: int = 1,
        update_freq: int = 5,
    ):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.do_clip = do_clip
        self.use_baseline = use_baseline
        self.normalize_advantages = normalize_advantages
        self.trajectory = trajectory
        self.lambd = lambd
        self.n_obs = n_obs
        self.update_freq = update_freq

        # Reward function
        self.reward_fn = reward_fn or make_reward_function(
            gamma=env.gamma_as,
            sigma=env.sigma_as,
            max_t=env.max_t,
            immediate_reward=env.immediate_reward,
            add_inventory=True,
            add_time=env.time_penalty > 0,
            always_final=env.always_final,
            inventory_penalty=env.inventory_penalty,
            time_penalty=env.time_penalty,
            midprice=env.midprice_cfg,
        )

        # Policy
        obs_dim = 4 * n_obs + 1
        act_dim = 4
        self.policy = policy
        if self.policy is None:
            network = build_mlp(obs_dim, act_dim, n_layers=2, hidden_size=64)
            self.policy = GaussianPolicy(network, act_dim)
            self.policy.to(device())
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        else:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Baseline
        if use_baseline:
            self.baseline = BaselineNetwork(
                input_dim=obs_dim,
                output_dim=1,
                n_layers=2,
                layer_size=64,
                lr=lr,
            )
        else:
            self.baseline = None

        # Tracking
        self.final_returns: List[float] = []
        self.final_values: List[float] = []

    def collect(self, nbatch: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def get_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Compute returns based on trajectory type."""
        if self.trajectory == "MC":
            return compute_mc_returns(rewards, self.gamma)
        elif self.trajectory == "TD":
            obs = np.zeros((rewards.shape[0], rewards.shape[1], 5))
            values = self.baseline.forward(np2torch(obs)).detach().cpu().numpy()
            return compute_td_returns(rewards, values, self.gamma, self.lambd)
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory}")

    def get_advantages(self, returns: np.ndarray, observations: np.ndarray) -> np.ndarray:
        """Compute advantages (possibly normalized)."""
        if self.use_baseline:
            advantages = self.baseline.calculate_advantage(returns, observations)
        else:
            advantages = returns
        if self.normalize_advantages:
            advantages = normalize(advantages)
        return advantages

    def update_policy_ppo(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logprobs: np.ndarray,
    ):
        """Perform one PPO update."""
        obs_t = np2torch(observations, True)
        act_t = np2torch(actions, True)
        adv_t = np2torch(advantages, True)
        old_lp_t = np2torch(old_logprobs, True)

        dist = self.policy.action_distribution(obs_t)
        log_probs = self.policy.log_probs(dist, act_t)
        entropy_loss = torch.mean(self.policy.entropy(dist, obs_t))

        z_ratio = torch.exp(log_probs - old_lp_t)

        if self.do_clip:
            clip_z = torch.clip(z_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            minimum = torch.min(z_ratio * adv_t, clip_z * adv_t)
        else:
            minimum = z_ratio * adv_t

        self.policy_optimizer.zero_grad()
        loss = -torch.mean(minimum) - self.entropy_coef * entropy_loss
        loss.backward()
        self.policy_optimizer.step()

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray) -> float:
        """Update the baseline value network."""
        if self.use_baseline:
            return self.baseline.update_baseline(returns, observations)
        return 0.0

    def train_step(self, paths: Dict[str, np.ndarray]):
        """Perform one training epoch: collect, compute returns, update."""
        returns = self.get_returns(paths["rew"])
        advantages = self.get_advantages(returns, paths["obs"])

        # Update baseline
        self.update_baseline(returns, paths["obs"])

        # Policy updates
        for _ in range(self.update_freq):
            old_logprobs = paths.get("old")
            if old_logprobs is None:
                old_logprobs = np.zeros_like(paths["rew"])
            self.update_policy_ppo(
                paths["obs"],
                paths["act"],
                advantages,
                old_logprobs,
            )

        # Track metrics
        if isinstance(paths["rew"], np.ma.MaskedArray):
            final_rews = np.array([r[~r.mask][-1] if not r.mask.all() else 0.0 for r in paths["rew"]])
        else:
            final_rews = paths["rew"][:, -1]
        self.final_returns.append(final_rews.mean())
        self.final_values.append(np.mean(paths.get("val", [0.0])))


# -------------------------------------------------------------------------- #
# Uniform Market Maker (fixed-length trajectories)
# -------------------------------------------------------------------------- #

class UniformMarketMaker(MarketMakerBase):
    """
    Market maker with uniform (fixed-length) trajectories.

    All trajectories have the same length `nt`; no early termination handling.
    """

    def __init__(self, env: MarketEnv, **kwargs):
        super().__init__(env, **kwargs)
        self.collector = UniformCollector(
            market=self._make_market(),
            policy=self.policy,
            reward_fn=self.reward_fn,
            dt=env.dt,
            max_t=env.max_t,
            nt=env.nt,
            gamma=self.gamma,
            do_ppo=True,
        )

    def _make_market(self) -> BaseMarket:
        """Re-create a fresh BaseMarket matching env config."""
        return BaseMarket(
            inventory=0,
            wealth=0.0,
            midprice=self.env.midprice_cfg,
            spread=self.env.spread_cfg,
            nstocks=self.env.nstocks_cfg,
            make_bell=self.env.make_bell_cfg,
            nsteps=self.env.nsteps_cfg,
            substeps=self.env.substeps_cfg,
            max_t=self.env.max_t,
            dt=self.env.dt,
            discount=self.gamma,
            gamma_as=self.env.gamma_as,
            sigma_as=self.env.sigma_as,
        )

    def collect(self, nbatch: int) -> Dict[str, np.ndarray]:
        # Re-create market each collection (reset)
        self.collector.market = self._make_market()
        return self.collector.collect(nbatch)

    def train(self, epochs: int, batch_size: int, log_every: int = 10):
        """
        Train for `epochs` epochs of `batch_size` trajectories each.

        Args:
            epochs:     number of training epochs
            batch_size: trajectories per epoch
            log_every:  print progress every N epochs
        """
        for epoch in range(epochs):
            paths = self.collect(batch_size)
            self.train_step(paths)

            if (epoch + 1) % log_every == 0:
                msg = (
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Return: {self.final_returns[-1]:.4f} | "
                    f"Value:  {self.final_values[-1]:.4f}"
                )
                print(msg)


# -------------------------------------------------------------------------- #
# Masked Market Maker (variable-length, MaskedArray)
# -------------------------------------------------------------------------- #

class MaskedMarketMaker(MarketMakerBase):
    """
    Market maker that handles variable-length trajectories via np.ma.MaskedArray.

    Trajectories that terminate early (e.g., book empty) are masked out
    so that only valid timesteps contribute to the loss.
    """

    def __init__(self, env: MarketEnv, **kwargs):
        super().__init__(env, **kwargs)
        self.collector = MaskedCollector(
            market=self._make_market(),
            policy=self.policy,
            reward_fn=self.reward_fn,
            dt=env.dt,
            max_t=env.max_t,
            nt=env.nt,
            gamma=self.gamma,
            do_ppo=True,
        )

    def _make_market(self) -> BaseMarket:
        return BaseMarket(
            inventory=0,
            wealth=0.0,
            midprice=self.env.midprice_cfg,
            spread=self.env.spread_cfg,
            nstocks=self.env.nstocks_cfg,
            make_bell=self.env.make_bell_cfg,
            nsteps=self.env.nsteps_cfg,
            substeps=self.env.substeps_cfg,
            max_t=self.env.max_t,
            dt=self.env.dt,
            discount=self.gamma,
            gamma_as=self.env.gamma_as,
            sigma_as=self.env.sigma_as,
        )

    def collect(self, nbatch: int) -> Dict[str, np.ndarray]:
        self.collector.market = self._make_market()
        return self.collector.collect(nbatch)

    def train(self, epochs: int, batch_size: int, log_every: int = 10):
        for epoch in range(epochs):
            paths = self.collect(batch_size)
            self.train_step(paths)

            if (epoch + 1) % log_every == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Return: {self.final_returns[-1]:.4f} | "
                    f"Value:  {self.final_values[-1]:.4f}"
                )


# -------------------------------------------------------------------------- #
# Market Maker (alias — pick MaskedMarketMaker for variable-length)
# -------------------------------------------------------------------------- #

class MarketMaker(MaskedMarketMaker):
    """
    Default market-making agent.

    Alias for MaskedMarketMaker — handles variable-length trajectories
    with masked tensor support.
    """
