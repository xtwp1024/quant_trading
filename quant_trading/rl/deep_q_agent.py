"""
Deep Q-Trading Agent with share sizing and soft target updates.

Adapted from deep-q-trading-agent (Jeong et al., 2019)
https://www.sciencedirect.com/science/article/abs/pii/S0957417418306134

Key features:
- Three DQN architectures: NumQ (joint), NumDReg-AD (action-dependent), NumDReg-ID (action-independent)
- Share sizing as part of the DQN output (outputs both action AND share quantity)
- Soft target network updates (tau interpolation)
- Online learning with sliding-window memory (not experience replay buffer)
- Confused market detection with threshold-based strategy switching

The "NumQ" approach: network outputs both action (buy/sell/hold) AND share size,
making it a more realistic trading agent that decides how many shares to trade.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, Tensor
from collections import deque
from typing import Tuple, Optional, List, Dict, Any

from .deep_q_networks import (
    NUMQ, NUMDREG_AD, NUMDREG_ID,
    ACT_MODE, NUM_MODE, FULL_MODE,
    BUY, HOLD, SELL,
    NumQModel, NumDRegModel,
)


# Default hyperparameters
DEFAULT_CONFIG = {
    "LOOKBACK": 200,
    "REWARD_WINDOW": 100,
    "GAMMA": 0.99,
    "TAU": 0.0003,           # Soft update interpolation factor
    "STEPS_PER_SOFT_UPDATE": 1,
    "LR": 0.001,
    "LR_NUMDREGAD": 0.001,
    "LR_NUMDREGID": 0.001,
    "BATCH_SIZE": 64,
    "MEMORY_CAPACITY": 64,    # Sliding window size for online learning
    "SHARE_TRADE_LIMIT": 100, # Max shares to trade per action
    "THRESHOLD": 0.0002,      # Confused market threshold
    "STRATEGY": HOLD,         # Default strategy in confused market (0=BUY, 1=HOLD, 2=SELL)
    "STRATEGY_NUM": 0.5,     # Default share ratio in confused market
    "USE_STRATEGY_TRAIN": False,
    "LOSS": "SMOOTH_L1_LOSS",
}


class DQN:
    """
    Deep Q-Network agent with share sizing capability.

    Supports three architectures:
    - NUMQ:        Joint Q-network (single branch)
    - NUMDREG_AD:  Action-Dependent dual branch
    - NUMDREG_ID:  Action-Independent dual branch
    """

    BUY = BUY
    HOLD = HOLD
    SELL = SELL

    def __init__(self, method: int, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.method = method
        self.mode = FULL_MODE

        self.policy_net: Optional[torch.nn.Module] = None
        self.target_net: Optional[torch.nn.Module] = None

        # Build networks based on method
        if self.method == NUMQ:
            self.policy_net = NumQModel()
            self.target_net = NumQModel()
        elif self.method == NUMDREG_AD:
            self.policy_net = NumDRegModel(NUMDREG_AD, self.mode)
            self.target_net = NumDRegModel(NUMDREG_AD, self.mode)
        elif self.method == NUMDREG_ID:
            self.policy_net = NumDRegModel(NUMDREG_ID, self.mode)
            self.target_net = NumDRegModel(NUMDREG_ID, self.mode)

        # Initialize target with same weights as policy
        self.hard_update()

    def hard_update(self):
        """Hard update: copy all weights from policy to target."""
        if self.policy_net is not None and self.target_net is not None:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self, tau: Optional[float] = None):
        """
        Soft update target network parameters.
        θ_target = τ*θ_policy + (1-τ)*θ_target

        Args:
            tau: Interpolation parameter (default from config)
        """
        if tau is None:
            tau = self.config["TAU"]

        if self.policy_net is None or self.target_net is None:
            return

        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def set_mode(self, mode: int):
        """Set training mode for NumDReg models."""
        self.mode = mode
        if self.policy_net is not None:
            self.policy_net.mode = mode
        if self.target_net is not None:
            self.target_net.mode = mode

    def get_optimizer(self):
        """Get optimizer for policy network."""
        lr = self.config["LR"]
        if self.method == NUMDREG_AD:
            lr = self.config["LR_NUMDREGAD"]
        elif self.method == NUMDREG_ID:
            lr = self.config["LR_NUMDREGID"]

        return optim.Adam(self.policy_net.parameters(), lr=lr)


class ReplayMemory:
    """
    Sliding-window replay memory for online learning.

    Unlike traditional experience replay that samples randomly from all past,
    this uses a fixed-size window of the most recent transitions.
    """

    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def update(self, transition: Tuple[Tensor, int, Tensor, Tensor]):
        """Append transition (state, action, rewards_all_actions, next_state)."""
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List:
        """Sample most recent batch_size transitions."""
        return list(self.memory)[-batch_size:]

    def __len__(self) -> int:
        return len(self.memory)


class FinanceEnvironment:
    """
    Finance trading environment for DQN training.

    State: 200-day price differences (lookback window)
    Actions: BUY=0, HOLD=1, SELL=2
    Reward: Based on share count, action, and price change

    The environment supports:
    - Computing profit and reward for a given action + share count
    - Computing rewards for ALL actions (for Q-learning update)
    - Replay memory management
    """

    def __init__(
        self,
        price_history: np.ndarray,
        lookback: int = 200,
        reward_window: int = 100,
    ):
        """
        Args:
            price_history: Array of price values (1D or column vector)
            lookback: Number of past days for state representation
            reward_window: Window for reward normalization
        """
        self.lookback = lookback
        self.reward_window = reward_window

        #BUY, HOLD, SELL = (1, 0, -1)
        self.action_space = (1, 0, -1)

        self.price_history = price_history
        self._init_prices()

        self.profits: List[float] = []
        self.replay_memory = ReplayMemory(capacity=DEFAULT_CONFIG["MEMORY_CAPACITY"])

    def _init_prices(self):
        """Initialize price deltas (state representation)."""
        # Pad with first price to handle lookback at start
        pad = np.tile(self.price_history[0], (self.lookback,))
        padded = np.concatenate([pad, self.price_history])

        # Price differences: p_t - p_{t-1}
        self.price_deltas = np.diff(padded, axis=0)
        self.current_idx = self.lookback

    def start_episode(self):
        """Reset episode-level tracking."""
        self.episode_losses: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_profit = 0.0
        self.current_idx = self.lookback

    def step(self) -> Tuple[Tensor, bool]:
        """
        Advance one timestep.

        Returns:
            state: Tensor of lookback price deltas
            done:  Boolean indicating end of episode
        """
        self.price = self.price_history[self.current_idx]
        self.prev_price = self.price_history[self.current_idx - 1]
        self.init_price = self.price_history[
            max(0, self.current_idx - self.reward_window)
        ]

        # State: last lookback price deltas
        self.state = torch.tensor(
            self.price_deltas[self.current_idx - self.lookback : self.current_idx],
            dtype=torch.float64,
        )

        self.current_idx += 1

        # End if we've exhausted data
        done = self.current_idx >= len(self.price_history)

        if done:
            self.next_state = torch.tensor([], dtype=torch.float64)
        else:
            self.next_state = torch.tensor(
                self.price_deltas[
                    self.current_idx - self.lookback + 1 : self.current_idx + 1
                ],
                dtype=torch.float64,
            )

        return self.state, done

    def compute_profit_and_reward(
        self, action_index: int, num: float
    ) -> Tuple[float, float]:
        """
        Compute profit and reward for taking action with num shares.

        Args:
            action_index: 0=BUY, 1=HOLD, 2=SELL
            num: Number of shares

        Returns:
            profit, reward
        """
        self.action = action_index
        self.num = num
        action_value = self.action_space[action_index]

        price_rel = (self.price - self.prev_price) / self.prev_price
        profit = num * action_value * price_rel
        reward = num * (1 + action_value * price_rel) * self.prev_price / self.init_price

        self.profit = profit
        self.reward = reward
        self.episode_rewards.append(reward)
        self.episode_profit += profit

        return profit, reward

    def compute_reward_all_actions(self, action_index: int, num: float):
        """
        Compute rewards for ALL actions (buy, hold, sell) at current step.

        This is key to the Q-learning update: we can evaluate what the
        reward WOULD have been for any action, not just the one taken.
        """
        profit, reward = self.compute_profit_and_reward(action_index, num)

        rewards_all = []
        for action in self.action_space:
            r = _reward(
                num=num,
                action_value=action,
                price=self.price,
                prev_price=self.prev_price,
                init_price=self.init_price,
            )
            rewards_all.append(r)

        self.rewards_all_actions = torch.tensor(rewards_all)
        return profit, reward, rewards_all

    def add_loss(self, loss: Optional[float]):
        if loss is None:
            return
        self.episode_losses.append(loss)

    def update_replay_memory(self):
        """Add current transition to replay memory."""
        if self.next_state.numel():
            self.replay_memory.update(
                (self.state, self.action, self.rewards_all_actions, self.next_state)
            )

    def on_episode_end() -> Tuple[float, float, float]:
        """Return avg_loss, avg_reward, total_profit."""
        pass  # Implemented in subclass below


def _reward(
    num: float, action_value: int, price: float, prev_price: float, init_price: float
) -> float:
    """Compute reward given action parameters."""
    r = 1 + action_value * (price - prev_price) / prev_price
    r = num * r * prev_price / init_price
    return r


def _profit(
    num: float, action_value: int, price: float, prev_price: float
) -> float:
    """Compute profit given action parameters."""
    return num * action_value * (price - prev_price) / prev_price


def select_action(
    model: DQN,
    state: Tensor,
    strategy: int = HOLD,
    strategy_num: float = 0.5,
    use_strategy: bool = False,
    only_use_strategy: bool = False,
) -> Tuple[List[float], int, np.ndarray, float]:
    """
    Select action and share count given current state.

    Args:
        model:           DQN agent
        state:           Current state tensor
        strategy:        Default action in confused market (BUY=0, HOLD=1, SELL=2)
        strategy_num:    Default share ratio in confused market
        use_strategy:    Whether to use strategy in confused market
        only_use_strategy: If True, always use strategy (ignores model)

    Returns:
        q_values:      Q-values for all 3 actions
        action_index:  Selected action (0, 1, or 2)
        num_values:    Share ratios for all actions
        selected_num:   Final share count to trade
    """
    with torch.no_grad():
        q, num = model.policy_net(state)

    q = q.squeeze().detach().numpy()
    num = num.squeeze().detach().numpy()

    # Select best action based on Q-values
    selected_action_index = int(np.argmax(q))

    # Determine share count
    share_limit = model.config["SHARE_TRADE_LIMIT"]

    if model.method == NUMDREG_ID:
        if model.mode == ACT_MODE:
            selected_num = share_limit * float(num[0])
        else:
            selected_num = share_limit * float(num)
    else:
        num = list(num)
        selected_num = share_limit * num[selected_action_index]

    # Confused market detection
    # |Q(s,a_BUY) - Q(s,a_SELL)| / sum|Q(s,a)| < threshold
    confidence = np.abs(q[BUY] - q[SELL]) / (np.sum(np.abs(q)) + 1e-8)

    if only_use_strategy:
        selected_action_index = strategy
        selected_num = share_limit * strategy_num
    elif use_strategy and confidence < model.config["THRESHOLD"]:
        selected_action_index = strategy

    return list(q), selected_action_index, num, selected_num


def get_batches(memory: ReplayMemory, batch_size: int):
    """Get batches from replay memory."""
    batch = list(zip(*memory.sample(batch_size=batch_size)))

    state_batch = torch.stack(batch[0])
    action_batch = torch.unsqueeze(torch.tensor(batch[1]), dim=1)
    reward_batch = torch.stack(batch[2])
    next_state_batch = torch.stack(batch[3])

    return state_batch, action_batch, reward_batch, next_state_batch


def optimize_model(
    model: DQN,
    optimizer: optim.Adam,
    memory: ReplayMemory,
    optim_actions: bool = True,
) -> float:
    """
    Optimize policy network on a batch from memory.

    Args:
        model:         DQN agent
        optimizer:     PyTorch optimizer
        memory:        Replay memory
        optim_actions: If True, optimize Q-values; else optimize share ratios

    Returns:
        loss value
    """
    state_batch, action_batch, reward_batch, next_state_batch = get_batches(
        memory, model.config["BATCH_SIZE"]
    )

    # Get Q values from policy net
    q_batch, num_batch = model.policy_net(state_batch)

    # Get next state Q values from target net
    next_q_batch, next_num_batch = model.target_net(next_state_batch)

    # Select which to optimize
    if optim_actions:
        pred_batch = q_batch
        next_pred_batch = next_q_batch
    else:
        if model.method == NUMDREG_ID:
            pred_batch = num_batch.repeat(1, 3)
            next_pred_batch = next_num_batch.repeat(1, 3)
        else:
            pred_batch = num_batch
            next_pred_batch = next_num_batch

    # Max Q for next state
    next_max_pred, _ = next_pred_batch.detach().max(dim=1)

    # Expected Q values
    gamma = model.config["GAMMA"]
    expected_pred = reward_batch + (gamma * torch.unsqueeze(next_max_pred, dim=1))

    # Compute loss
    if model.config["LOSS"] == "MSE_LOSS":
        loss = F.mse_loss(expected_pred, pred_batch)
    else:
        loss = F.smooth_l1_loss(expected_pred, pred_batch)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    model: DQN,
    price_history: np.ndarray,
    episodes: int = 100,
    strategy: int = HOLD,
    use_strategy: bool = False,
) -> Tuple[DQN, List[float], List[float], List[float]]:
    """
    Train DQN agent on price history.

    Args:
        model:          DQN agent (will be modified in-place)
        price_history:  1D array of prices
        episodes:       Number of training episodes
        strategy:       Default action in confused market
        use_strategy:   Use confused market strategy during training

    Returns:
        (model, losses, rewards, profits)
    """
    env = FinanceEnvironment(price_history)
    optimizer = model.get_optimizer()

    losses, rewards, profits = [], [], []

    for e in range(episodes):
        env.start_episode()
        optim_steps = 0

        while True:
            state, done = env.step()

            # Select action
            _, selected_action, _, selected_num = select_action(
                model=model,
                state=state,
                strategy=strategy,
                use_strategy=use_strategy,
            )

            # Compute reward for all actions
            env.compute_reward_all_actions(selected_action, selected_num)
            env.update_replay_memory()

            # Wait until memory is large enough
            if len(env.replay_memory) < model.config["BATCH_SIZE"]:
                if done:
                    break
                continue

            # Optimize
            if model.method == NUMDREG_AD or model.method == NUMDREG_ID:
                if model.mode == ACT_MODE:
                    loss = optimize_model(model, optimizer, env.replay_memory, True)
                elif model.mode == NUM_MODE:
                    loss = optimize_model(model, optimizer, env.replay_memory, False)
                else:  # FULL_MODE
                    act_loss = optimize_model(model, optimizer, env.replay_memory, True)
                    num_loss = optimize_model(model, optimizer, env.replay_memory, False)
                    loss = act_loss + num_loss
            else:
                loss = optimize_model(model, optimizer, env.replay_memory)

            env.add_loss(loss)
            optim_steps += 1

            # Soft update target network
            if optim_steps % model.config["STEPS_PER_SOFT_UPDATE"] == 0:
                model.soft_update()

            if done:
                break

        # Episode summary
        avg_loss = np.mean(env.episode_losses) if env.episode_losses else 0.0
        avg_reward = np.mean(env.episode_rewards) if env.episode_rewards else 0.0

        losses.append(avg_loss)
        rewards.append(avg_reward)
        profits.append(env.episode_profit)

        if (e + 1) % 10 == 0:
            print(
                f"Episode {e+1}/{episodes} | "
                f"Profit: {env.episode_profit:.4f} | "
                f"Avg Reward: {avg_reward:.4f} | "
                f"Avg Loss: {avg_loss:.4f}"
            )

    return model, losses, rewards, profits


def evaluate(
    model: DQN,
    price_history: np.ndarray,
    strategy: int = HOLD,
    strategy_num: float = 0.5,
    use_strategy: bool = False,
    only_use_strategy: bool = False,
) -> Tuple[List[float], List[float], float]:
    """
    Evaluate DQN agent on price history.

    Returns:
        (rewards, profits, total_profit)
    """
    env = FinanceEnvironment(price_history)
    env.start_episode()

    rewards, profits = [], []

    actions_taken = [0, 0, 0]

    while True:
        state, done = env.step()

        _, selected_action, _, selected_num = select_action(
            model=model,
            state=state,
            strategy=strategy,
            strategy_num=strategy_num,
            use_strategy=use_strategy,
            only_use_strategy=only_use_strategy,
        )

        actions_taken[selected_action] += 1

        profit, reward = env.compute_profit_and_reward(selected_action, selected_num)
        rewards.append(reward)
        profits.append(profit)

        if done:
            break

    print(f"Actions taken: {actions_taken}")
    return rewards, profits, env.episode_profit


class NumQAgent:
    """
    Convenience wrapper for NumQ (joint) DQN agent.

    Usage:
        agent = NumQAgent()
        agent.train(prices, episodes=100)
        rewards, profits, total = agent.evaluate(prices)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.model = DQN(NUMQ, config)
        self.config = self.model.config

    def train(
        self,
        price_history: np.ndarray,
        episodes: int = 100,
        strategy: int = HOLD,
        use_strategy: bool = False,
    ):
        self.model, self.losses, self.rewards, self.profits = train(
            self.model, price_history, episodes, strategy, use_strategy
        )
        return self

    def evaluate(self, price_history: np.ndarray):
        return evaluate(self.model, price_history)

    def predict(self, state: np.ndarray) -> Tuple[int, float]:
        """Predict action and share count for a state."""
        state_t = torch.tensor(state, dtype=torch.float64)
        _, action, _, num = select_action(self.model, state_t)
        return action, num


class NumDRegAgent:
    """
    Convenience wrapper for NumDReg (dual-branch) DQN agent.

    Supports both NUMDREG_AD (action-dependent) and NUMDREG_ID (action-independent).

    Usage:
        # Action-dependent
        agent = NumDRegAgent(method=NUMDREG_AD)
        agent.train(prices, episodes=100)

        # Action-independent
        agent = NumDRegAgent(method=NUMDREG_ID)
        agent.train(prices, episodes=100)
    """

    def __init__(
        self,
        method: int = NUMDREG_AD,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = DQN(method, config)
        self.config = self.model.config

    def train(
        self,
        price_history: np.ndarray,
        episodes: int = 100,
        strategy: int = HOLD,
        use_strategy: bool = False,
    ):
        self.model, self.losses, self.rewards, self.profits = train(
            self.model, price_history, episodes, strategy, use_strategy
        )
        return self

    def evaluate(self, price_history: np.ndarray):
        return evaluate(self.model, price_history)

    def predict(self, state: np.ndarray) -> Tuple[int, float]:
        """Predict action and share count for a state."""
        state_t = torch.tensor(state, dtype=torch.float64)
        _, action, _, num = select_action(self.model, state_t)
        return action, num
