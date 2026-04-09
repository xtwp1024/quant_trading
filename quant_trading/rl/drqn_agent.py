"""
DRQNAgent - Deep Recurrent Q-Network agent for stock trading.

Key features:
- LSTM layer for temporal credit assignment over sequential states.
- DQN with experience replay and soft target network updates.
- Epsilon-greedy exploration with decay.
- Action augmentation support: stores all 3 action transitions per step.
- Soft target network update (tau parameter).

Based on the original DRQN_Stock_Trading Agent and DQN model from:
- code_server/agent.py  (Agent class)
- code_server/model.py (DQN class)

Usage:
    from quant_trading.rl import DRQNAgent

    agent = DRQNAgent(state_dim=state_dim, action_dim=3)
    agent.store(state, all_actions, all_next_states, all_rewards, selected_action)
    agent.optimize(step)

    # In training loop:
    action = agent.act(state)          # epsilon-greedy
    agent.store(state, actions, next_states, rewards, action)
    agent.optimize(step)
"""

import copy
import math
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import normal_ as normal_init

from .replay_memory import ReplayMemory, Transition


# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRQNNetwork(nn.Module):
    """
    Deep Recurrent Q-Network with LSTM for temporal credit assignment.

    Architecture:
        Input (state_dim) -> FC(256, ELU) -> FC(256, ELU) -> LSTM(256, 256)
         -> FC(256, 3)

    Data flow:
        Input shape:  (batch_size, seq_length, state_dim)
        LSTM output:  (batch_size, seq_length, 256)
        Flattened:    (batch_size * seq_length, 256)
        Q-values:     (batch_size * seq_length, 3)

    Args:
        state_dim:   Dimension of the state vector.
        action_dim:  Number of actions (3: short, hold, long).
        hidden_size: LSTM hidden dimension (default 256).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Feature extraction: two FC layers with ELU activation
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()

        # LSTM layer for sequential/temporal modeling
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Output Q-values for each action
        self.fc_out = nn.Linear(hidden_size, action_dim)

        # Weight initialization (normal distribution)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply normal weight initialization to all parameters."""
        for param in self.parameters():
            if param.dim() >= 2:
                normal_init(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_length, state_dim).

        Returns:
            Q-values tensor of shape (batch_size * seq_length, action_dim).
        """
        # Feature extraction
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))

        # LSTM forward
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_size)

        # Flatten for linear output: (batch_size * seq_length, hidden_size)
        batch_size, seq_len, hid_dim = lstm_out.shape
        linear_in = lstm_out.contiguous().view(seq_len * batch_size, hid_dim)

        return self.fc_out(linear_in)


class DRQNAgent:
    """
    Deep Recurrent Q-Network agent with:
    - LSTM for temporal dependencies
    - Experience replay (ReplayMemory)
    - Soft target network updates
    - Epsilon-greedy exploration
    - Action augmentation (stores all action transitions per step)

    The key training pattern from the original DRQN_Stock_Trading:
    - During soft-update phase (step < T): ALL 3 action transitions stored
    - After: only the selected action's transition is stored

    Args:
        state_dim:      State dimension.
        T:              Sequence length / soft-update period (default 96).
        memory_cap:     Replay memory capacity (default 10000).
        gamma:          Discount factor (default 0.99).
        lr:             Learning rate (default 0.00025).
        batch_size:     Training batch size (default 16).
        epsilon:        Initial exploration rate (default 1.0).
        epsilon_min:    Minimum exploration rate (default 0.01).
        epsilon_decay:  Exploration decay rate (default 0.995).
        tau:            Soft target update coefficient (default 0.001).
        target_update_period: Steps between target net hard updates (default T).
        hidden_size:    LSTM hidden dimension (default 256).
        is_eval:        Evaluation mode (disables exploration).
    """

    def __init__(
        self,
        state_dim: int,
        T: int = 96,
        memory_cap: int = 10000,
        gamma: float = 0.99,
        lr: float = 0.00025,
        batch_size: int = 16,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        tau: float = 0.001,
        target_update_period: Optional[int] = None,
        hidden_size: int = 256,
        is_eval: bool = False,
    ):
        self.state_dim = state_dim
        self.action_dim = 3
        self.T = T
        self.memory_cap = memory_cap
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.target_update_period = target_update_period or T
        self.hidden_size = hidden_size
        self.is_eval = is_eval

        # Replay memory
        self.memory = ReplayMemory(capacity=memory_cap)

        # Policy and target networks
        self.policy_net = DRQNNetwork(state_dim, self.action_dim, hidden_size).to(
            DEVICE
        )
        self.target_net = DRQNNetwork(state_dim, self.action_dim, hidden_size).to(
            DEVICE
        )
        # Initialize target net with same weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def act(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state (can be a sequence for LSTM).

        Returns:
            Action index (0, 1, or 2) corresponding to (-1, 0, 1).
        """
        if not self.is_eval and np.random.rand() <= self.epsilon:
            # Random action
            return int(np.random.randint(0, self.action_dim))

        # Greedy: use target net for evaluation stability
        with torch.no_grad():
            tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            # Handle both single step and sequence inputs
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)  # (1, 1, state_dim)
            q_values = self.target_net(tensor)
            # Use Q-values at the last timestep
            return int(np.argmax(q_values[-1].cpu().numpy()))  # index 0,1,2

    def store(
        self,
        state: np.ndarray,
        actions: List[int],
        next_states: List[np.ndarray],
        rewards: np.ndarray,
        action: int,
        step: int,
    ) -> None:
        """
        Store transitions in replay memory (action augmentation).

        This is the NOVEL action augmentation technique: for each environment step,
        all 3 action transitions are stored (not just the selected one). This gives
        3x more training signal per step.

        During soft-update phase (step < T): all 3 transitions stored.
        After: only the selected action's transition is stored.

        Args:
            state:        Current state.
            actions:      List of all 3 action values [-1, 0, 1].
            next_states:  List of 3 next states (one per action).
            rewards:      Array of 3 rewards (one per action).
            action:       Selected action index (0, 1, or 2).
            step:         Current step number.
        """
        if step < self.T:
            # Soft update phase: store ALL 3 action transitions
            for n in range(len(actions)):
                self.memory.push(
                    state,
                    actions[n],
                    next_states[n] if next_states[n] is not None else None,
                    float(rewards[n]),
                )
        else:
            # After soft phase: only store the selected action's transition
            for n in range(len(actions)):
                if actions[n] == action:
                    self.memory.push(
                        state,
                        actions[n],
                        next_states[n] if next_states[n] is not None else None,
                        float(rewards[n]),
                    )
                    break

    def store_augmented(
        self,
        state: np.ndarray,
        all_rewards: np.ndarray,
        all_next_states: List[Dict[str, np.ndarray]],
        selected_action_idx: int,
        selected_reward: float,
        step: int,
    ) -> None:
        """
        Store transitions using augmented info from DRQNTradingEnv.step().

        This is a convenience method that works directly with the info dict
        returned by DRQNTradingEnv.step().

        Args:
            state:                 Current state.
            all_rewards:           Rewards for all 3 actions.
            all_next_states:       List of 3 dicts with normalized/action_enc.
            selected_action_idx:   Index of selected action (0, 1, or 2).
            selected_reward:       Reward for selected action.
            step:                  Current step number.
        """
        actions = [-1, 0, 1]
        next_states = []

        for i in range(3):
            ns_data = all_next_states[i]
            # Rebuild full next state from components
            time_enc = self._compute_time_encoding(step + 1)
            next_state_vec = np.concatenate([
                ns_data["normalized"],
                ns_data["action_enc"],
                time_enc,
            ])
            next_states.append(next_state_vec.astype(np.float32))

        self.store(state, actions, next_states, all_rewards, selected_action_idx, step)

    def _compute_time_encoding(self, step: int) -> np.ndarray:
        """Compute sinusoidal time encoding (matches DRQNTradingEnv)."""
        minute_f = np.sin(2 * np.pi * (step % 60) / 60.0)
        hour_f = np.sin(2 * np.pi * (step % (60 * 24)) / (60.0 * 24.0))
        day_f = np.sin(2 * np.pi * ((step // (60 * 24)) % 7) / 7.0)
        return np.array([minute_f, hour_f, day_f], dtype=np.float32)

    def optimize(self, step: int) -> Optional[float]:
        """
        Perform one optimization step on a sampled batch.

        Uses the DRQN-specific loss computation from the original:
        - Samples from replay memory
        - Computes Q(s_t, a) using policy net (only at final timestep T-1)
        - Computes V(s_{t+1}) using target net (only at final timestep T-1)
        - MSELoss between expected and actual Q values

        Args:
            step: Current step number (used for periodic target update).

        Returns:
            Loss value, or None if not enough samples.
        """
        if len(self.memory) < self.batch_size * 10:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Build tensors
        next_state_batch = torch.FloatTensor(
            np.array(batch.next_state)
        ).to(DEVICE)
        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            dtype=torch.bool,
            device=DEVICE,
        )
        non_final_next_states = torch.cat(
            [s for s in next_state_batch if s is not None]
        )

        state_batch = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        action_batch = torch.LongTensor(
            [a + 1 for a in batch.action]  # shift -1,0,1 -> 0,1,2
        ).to(DEVICE)
        reward_batch = torch.FloatTensor(list(batch.reward)).to(DEVICE)

        # Handle state sequence dimensions
        # state_batch shape: (batch_size, seq_len, state_dim)
        seq_len = state_batch.size(1)

        # Compute Q(s_t, a) - policy net
        # We only look at the Q-values at the LAST timestep (T-1 = 95)
        policy_out = self.policy_net(state_batch)  # (batch*seq, action_dim)
        l_total = policy_out.size(0)
        # Index 95, 191, 287, ... (every seq_len steps for each sample in batch)
        # For batch_size=16, seq_len=96: indices 95, 191, 287, ...
        last_idx = seq_len - 1
        indices = torch.arange(0, l_total, seq_len).to(DEVICE) + last_idx
        state_action_values = policy_out[indices].gather(
            1, action_batch.unsqueeze(1)
        ).squeeze(-1)

        # Compute V(s_{t+1}) - target net
        next_state_values = torch.zeros(
            self.batch_size, device=DEVICE
        )
        if non_final_mask.any():
            target_out = self.target_net(next_state_batch)
            t_l_total = target_out.size(0)
            t_indices = torch.arange(0, t_l_total, seq_len).to(DEVICE) + last_idx
            next_state_values[non_final_mask] = (
                target_out[t_indices].max(1)[0].detach()
            )

        # Expected Q values
        expected = (next_state_values * self.gamma) + reward_batch

        # Compute loss and optimize
        loss = F.mse_loss(state_action_values, expected)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (from original)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # Soft target network update (every T steps)
        if step % self.T == 0:
            self._soft_update()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def _soft_update(self) -> None:
        """
        Soft update target network parameters:
        theta_target = tau * theta_policy + (1 - tau) * theta_target
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data
                + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self) -> None:
        """Hard update: copy policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save policy and target network weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        torch.save(self.target_net.state_dict(), path.replace("policy", "target"))

    def load(self, path: str) -> None:
        """Load policy and target network weights."""
        self.policy_net.load_state_dict(
            torch.load(path, map_location=DEVICE)
        )
        self.target_net.load_state_dict(
            torch.load(path.replace("policy", "target"), map_location=DEVICE)
        )

    def set_eval(self) -> None:
        """Set agent to evaluation mode (no exploration)."""
        self.is_eval = True

    def set_train(self) -> None:
        """Set agent to training mode (exploration enabled)."""
        self.is_eval = False
