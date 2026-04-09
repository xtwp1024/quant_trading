"""
DAPO (Dual-clipping Asymmetric Policy Optimization) Algorithm.

Adapted from FinRL-DAPO-SR for quant_trading package.

DAPO is a policy gradient reinforcement learning algorithm that extends PPO
with asymmetric clipping and dynamic sampling:
- Decoupled clipping ranges (epsilon_low, epsilon_high) for positive/negative advantages
- Group-relative advantages computed from multiple action samples per state
- Dynamic sampling filters out states where all samples receive identical rewards
- Optional LLM risk/sentiment reward adjustment

Key differences from PPO:
- Uses separate clip ratios for positive (1+epsilon_high) and negative (1-epsilon_low) advantages
- No KL penalty term in the loss (one of DAPO's key features)
- Group advantage calculation with per-state action sampling

Key reference:
  "DAPO: Dual-Clipping Policy Optimization with Asymmetric Clip Range" (2024)
"""

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import scipy.signal
from gymnasium.spaces import Box, Discrete
import os

# Try to import MPI, but make it optional
try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    # Create a mock MPI module for single-process operation
    class MockMPI:
        COMM_WORLD = None

    MPI = MockMPI()


# Default device
device = torch.device("cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    Magic from rllab for computing discounted cumulative sums of vectors.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def mpi_avg(val, comm=None):
    """Average a value across MPI processes. Falls back to returning the value if MPI unavailable."""
    if not MPI_AVAILABLE or comm is None:
        return val
    return comm.allreduce(val, op=MPI.SUM) / max(comm.Get_size(), 1)


def num_procs():
    """Return number of MPI processes. Returns 1 if MPI is not available."""
    if not MPI_AVAILABLE:
        return 1
    return MPI.COMM_WORLD.Get_size()


def proc_id():
    """Return MPI process ID. Returns 0 if MPI is not available."""
    if not MPI_AVAILABLE:
        return 0
    return MPI.COMM_WORLD.Get_rank()


def sync_params(module, device):
    """
    Synchronize parameters across MPI processes.
    Falls back to no-op if MPI is unavailable.
    """
    if not MPI_AVAILABLE or num_procs() == 1:
        return

    params = [p.detach().cpu() for p in module.parameters()]
    flat_params = torch.cat([p.reshape(-1) for p in params])
    flat_params_np = flat_params.numpy()

    local_sum = flat_params_np.copy()
    global_sum = np.zeros_like(local_sum)

    MPI.COMM_WORLD.Allreduce(local_sum, global_sum, op=MPI.SUM)
    avg_params_np = global_sum / num_procs()
    avg_params = torch.from_numpy(avg_params_np).float()

    idx = 0
    for p in module.parameters():
        shape = p.shape
        numel = p.numel()
        new_p = avg_params[idx : idx + numel].reshape(shape).to(device)
        p.data.copy_(new_p)
        idx += numel


def setup_pytorch_for_mpi():
    """Setup PyTorch for MPI. No-op on systems without MPI."""
    pass


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        self.to(device)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), logp_a.cpu().numpy()

    def act_batch(self, obs, num_samples=10):
        """Sample multiple actions for a single observation (DAPO key feature)."""
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            actions = []
            logps = []
            for _ in range(num_samples):
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                actions.append(a.cpu().numpy())
                logps.append(logp_a.cpu().numpy())
            return actions, logps

    def act(self, obs):
        return self.step(obs)[0]


class DAPOBuffer:
    """
    Experience buffer for DAPO with group-relative advantage computation.

    Stores trajectories and computes advantages relative to other action samples
    taken from the same state. This is the key mechanism that enables DAPO's
    dynamic sampling feature.
    """

    def __init__(
        self, obs_dim, act_dim, size, num_samples_per_state=10, gamma=0.99
    ):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.state_indices = np.zeros(size, dtype=np.int32)
        self.num_samples_per_state = num_samples_per_state
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.current_state_idx = 0

    def store(self, obs, act, rew, logp, state_idx):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.state_indices[self.ptr] = state_idx
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def compute_group_advantages(self):
        """
        Compute group-relative advantages with dynamic sampling.

        For each state group, normalizes rewards relative to other samples from
        the same state. Filters out states where all rewards are identical
        (no gradient signal for the policy).
        """
        unique_states = np.unique(self.state_indices[: self.ptr])
        states_to_keep = []

        for state_idx in unique_states:
            sample_indices = np.where(self.state_indices[: self.ptr] == state_idx)[
                0
            ]

            if len(sample_indices) > 1:
                rewards = self.rew_buf[sample_indices]

                # Dynamic sampling: only keep states with reward variance
                if np.std(rewards) > 1e-6:
                    states_to_keep.append(state_idx)
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)
                    self.adv_buf[sample_indices] = normalized_rewards

        mask = np.zeros(self.ptr, dtype=bool)
        for state_idx in states_to_keep:
            mask = mask | (self.state_indices[: self.ptr] == state_idx)

        return mask

    def get(self):
        assert self.ptr > 0

        mask = self.compute_group_advantages()

        if np.any(mask):
            data = dict(
                obs=self.obs_buf[: self.ptr][mask],
                act=self.act_buf[: self.ptr][mask],
                ret=self.ret_buf[: self.ptr][mask],
                adv=self.adv_buf[: self.ptr][mask],
                logp=self.logp_buf[: self.ptr][mask],
            )
        else:
            data = dict(
                obs=np.zeros((0,) + self.obs_buf.shape[1:], dtype=np.float32),
                act=np.zeros((0,) + self.act_buf.shape[1:], dtype=np.float32),
                ret=np.zeros(0, dtype=np.float32),
                adv=np.zeros(0, dtype=np.float32),
                logp=np.zeros(0, dtype=np.float32),
            )

        self.ptr, self.path_start_idx = 0, 0
        self.current_state_idx = 0

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def dapo(
    env_fn,
    actor_critic=MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[256, 128], activation=torch.nn.ReLU),
    seed=42,
    steps_per_epoch=20000,
    epochs=100,
    gamma=0.995,
    epsilon_low=0.2,
    epsilon_high=0.28,
    pi_lr=3e-5,
    train_pi_iters=100,
    lam=0.95,
    max_ep_len=3000,
    target_kl=0.15,
    logger_kwargs=dict(),
    save_freq=10,
    num_samples_per_state=10,
    env_kwargs=None,
    adjustment_type="both",
    alpha=1.0,
    beta=1.0,
    force_cpu=False,
):
    """
    DAPO: Dual-Clipping Asymmetric Policy Optimization.

    Args:
        env_fn: Environment factory function
        actor_critic: Actor-critic network class
        ac_kwargs: Keyword args for actor-critic network
        seed: Random seed
        steps_per_epoch: Steps per epoch per process
        epochs: Number of training epochs
        gamma: Discount factor
        epsilon_low: Lower clip ratio (for negative advantages)
        epsilon_high: Upper clip ratio (for positive advantages)
        pi_lr: Policy learning rate
        train_pi_iters: Max policy update iterations per epoch
        lam: GAE lambda
        max_ep_len: Maximum episode length
        target_kl: Early stopping KL threshold
        logger_kwargs: Logger configuration
        save_freq: Model save frequency (epochs)
        num_samples_per_state: Action samples per state for group advantage
        env_kwargs: Environment configuration dict
        adjustment_type: LLM adjustment type ('both', 'sentiment', 'risk', 'none')
        alpha: Exponent for sentiment adjustment
        beta: Exponent for risk adjustment
        force_cpu: Force CPU usage

    Returns:
        Trained actor-critic module
    """
    global device
    if force_cpu:
        device = torch.device("cpu")

    setup_pytorch_for_mpi()

    # Set random seeds
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    sync_params(ac, device)

    var_counts = tuple(count_vars(module) for module in [ac.pi])

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / max(num_procs(), 1))
    buf = DAPOBuffer(
        obs_dim,
        act_dim,
        local_steps_per_epoch * num_samples_per_state,
        num_samples_per_state,
        gamma,
    )

    # Helper functions for hypothetical returns
    def calculate_portfolio_return(action, current_prices, next_prices):
        """Calculate portfolio return for a given action."""
        price_changes = next_prices / current_prices - 1
        return np.sum(action * price_changes)

    def extract_prices(state):
        """Extract stock prices from state."""
        stock_dim = (
            env_kwargs["stock_dim"] if env_kwargs and "stock_dim" in env_kwargs else 84
        )
        return state[0, 1 : stock_dim + 1]

    def extract_llm_features(state):
        """Extract LLM sentiment and risk scores from state."""
        stock_dim = (
            env_kwargs["stock_dim"] if env_kwargs and "stock_dim" in env_kwargs else 84
        )
        sentiment_start = -(2 * stock_dim)
        risk_start = -stock_dim
        llm_sentiments = state[0, sentiment_start:risk_start]
        llm_risks = state[0, risk_start:]
        return llm_sentiments, llm_risks

    # DAPO policy loss with decoupled asymmetric clipping
    def compute_loss_pi(data):
        obs = data["obs"].to(device)
        act = data["act"].to(device)
        adv = data["adv"].to(device)
        logp_old = data["logp"].to(device)

        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        # Asymmetric clipping (DAPO key feature)
        clip_low = 1.0 - epsilon_low
        clip_high = 1.0 + epsilon_high

        clip_ratio = torch.clamp(ratio, clip_low, clip_high)
        surrogate = ratio * adv
        clipped_surrogate = clip_ratio * adv

        # DAPO loss: minimize negative of minimum
        loss_pi = -(torch.min(surrogate, clipped_surrogate)).mean()

        # No KL penalty (DAPO removes this)
        total_loss = loss_pi

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(clip_high) | ratio.lt(clip_low)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return total_loss, pi_info

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)

    def update():
        data = buf.get()

        if data["obs"].shape[0] == 0:
            print(
                "No valid data for update (all states filtered by dynamic sampling)"
            )
            return

        data = {k: v.to(device) for k, v in data.items()}

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)

            kl = pi_info["kl"]
            kl_avg = mpi_avg(kl)

            if kl_avg > 1.5 * target_kl:
                print("Early stopping at step %d due to reaching max kl." % i)
                break

            loss_pi.backward()

            # Gradient averaging across MPI processes
            if MPI_AVAILABLE and num_procs() > 1:
                for p in ac.pi.parameters():
                    if p.grad is not None:
                        p_grad_cpu = p.grad.detach().cpu().numpy()
                        p_grad_avg = np.zeros_like(p_grad_cpu)
                        MPI.COMM_WORLD.Allreduce(
                            p_grad_cpu, p_grad_avg, op=MPI.SUM
                        )
                        p.grad.copy_(torch.from_numpy(p_grad_avg).to(device))

            pi_optimizer.step()

    # Training loop
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    state_idx = 0

    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        actual_env_steps = int(local_steps_per_epoch / num_samples_per_state)

        for t in range(actual_env_steps):
            current_state = o
            current_prices = extract_prices(current_state)

            # Sample multiple actions per state (DAPO key feature)
            actions, logps = ac.act_batch(o, num_samples=num_samples_per_state)

            # Use first action to step environment
            next_o, r, d, _ = env.step(actions[0])
            ep_ret += r
            ep_len += 1

            next_state = next_o
            next_prices = extract_prices(next_state)
            llm_sentiments, llm_risks = extract_llm_features(next_state)

            # Process all sampled actions with reward adjustment
            for i in range(num_samples_per_state):
                action = actions[i]
                logp = logps[i]

                if i == 0:
                    base_reward = r
                else:
                    base_reward = calculate_portfolio_return(
                        action, current_prices, next_prices
                    )

                position_values = action * next_prices
                total_value = np.sum(position_values)

                # LLM risk/sentiment reward adjustment
                if total_value == 0:
                    adjusted_reward = base_reward
                else:
                    risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
                    sentiment_to_weight = {
                        1: 0.99,
                        2: 0.995,
                        3: 1.0,
                        4: 1.005,
                        5: 1.01,
                    }

                    llm_risks_weights = np.vectorize(risk_to_weight.get)(llm_risks)
                    llm_sentiment_weights = np.vectorize(sentiment_to_weight.get)(
                        llm_sentiments
                    )

                    stock_weights = position_values / total_value
                    aggregated_sentiment = np.dot(stock_weights, llm_sentiment_weights)
                    aggregated_risk = np.dot(stock_weights, llm_risks_weights)

                    if adjustment_type == "both":
                        adjustment_factor = (aggregated_sentiment ** alpha) / (
                            (aggregated_risk ** beta) + 1e-8
                        )
                        adjusted_reward = base_reward * adjustment_factor
                    elif adjustment_type == "sentiment":
                        adjusted_reward = base_reward * (aggregated_sentiment ** alpha)
                    elif adjustment_type == "risk":
                        adjusted_reward = base_reward / (
                            (aggregated_risk ** beta) + 1e-8
                        )
                    else:
                        adjusted_reward = base_reward

                buf.store(current_state, action, adjusted_reward, logp, state_idx)

            state_idx += 1
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == actual_env_steps - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(
                        "Warning: trajectory cut off by epoch at %d steps."
                        % ep_len,
                        flush=True,
                    )

                if timeout or epoch_ended:
                    last_val = 0
                else:
                    last_val = 0

                buf.finish_path(last_val)

                if terminal:
                    pass  # Could log EpRet/EpLen here

                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save checkpoint
        if epoch % save_freq == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"agent_dapo_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ac.state_dict(),
                    "pi_optimizer_state_dict": pi_optimizer.state_dict(),
                },
                checkpoint_path,
            )

        update()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{epochs} completed in {elapsed:.1f}s"
        )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "agent_dapo_final.pth")
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": ac.state_dict(),
        },
        final_model_path,
    )
    print(f"\nTraining finished. Final model saved in {final_model_path}")

    return ac
