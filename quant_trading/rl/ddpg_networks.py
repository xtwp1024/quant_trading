"""
DDPG Actor-Critic Networks.

Adapted from PyTorch-DDPG-Stock-Trading (JohsuaWu1997).
Key features:
- 3-timestep state window for temporal awareness
- Actor outputs continuous portfolio weights
- Critic outputs Q-value for state-action pair
- Separate target networks with soft updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_copy(target, source):
    """Hard copy: copy weights directly from source to target."""
    for target_weight, source_weight in zip(target, source):
        target_weight.data = source_weight.data.clone()


def soft_copy(target, source, tau=0.01):
    """Soft copy: softly update target weights toward source weights."""
    for target_weight, source_weight in zip(target, source):
        target_weight.data.copy_(target_weight.data * (1.0 - tau) + source_weight.data * tau)


class ActorNet(nn.Module):
    """Actor network that maps state to continuous action (portfolio weights)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorNet, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.nn(x)


class CriticNet(nn.Module):
    """Critic network that outputs Q-value for state-action pair."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CriticNet, self).__init__()
        # Feature extraction from state
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=1),
        )
        # Combine action and state features
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, a, x):
        x_out = self.nn1(x)
        ax = torch.cat((a, x_out), dim=1)
        out = self.nn2(ax)
        return out


class Actor:
    """Actor for DDPG with separate target network."""

    def __init__(self, time_dim, state_dim, action_dim, hidden_dim, device=None):
        self.device = device or torch.device("cpu")
        input_dim = state_dim * (time_dim + 1)  # time_dim timesteps + current amount

        self.actor = ActorNet(input_dim, hidden_dim, action_dim).to(self.device)
        self.target = ActorNet(input_dim, hidden_dim, action_dim).to(self.device)
        self.actor_weights = list(self.actor.parameters())
        self.target_weights = list(self.target.parameters())
        self.optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize target with same weights
        hard_copy(self.target_weights, self.actor_weights)

    def train(self, loss_grad):
        """Train actor using gradient from critic."""
        self.optimizer.zero_grad()
        # Use negative gradient for policy gradient
        self.actor_weights[-1].backward(-loss_grad)
        self.optimizer.step()

    def actor_action(self, state):
        """Get action from online network (requires gradient)."""
        self.actor.zero_grad()
        return self.actor(state)

    def target_action(self, state):
        """Get action from target network (no gradient)."""
        with torch.no_grad():
            return self.target(state)

    def update_target(self):
        """Soft update target network toward online network."""
        soft_copy(self.target_weights, self.actor_weights)


class Critic:
    """Critic for DDPG with separate target network."""

    def __init__(self, time_dim, state_dim, action_dim, hidden_dim, device=None):
        self.device = device or torch.device("cpu")
        self.action_dim = action_dim
        input_dim = state_dim * (time_dim + 1)

        self.critic = CriticNet(input_dim, hidden_dim, action_dim).to(self.device)
        self.target = CriticNet(input_dim, hidden_dim, action_dim).to(self.device)
        self.critic_weights = list(self.critic.parameters())
        self.target_weights = list(self.target.parameters())
        self.optimizer = torch.optim.Adam(self.critic.parameters())
        self.loss = torch.tensor(0.0, device=self.device)

        # Initialize target with same weights
        hard_copy(self.target_weights, self.critic_weights)

    def train(self, y_batch, action_batch, state_batch):
        """Train critic using TD target and return gradient for actor."""
        criterion = nn.MSELoss()
        y_pred = self.critic(action_batch, state_batch)
        self.loss = criterion(y_pred, y_batch)
        self.optimizer.zero_grad()
        self.loss.backward()

        # Compute gradient for actor update (policy gradient)
        grad = torch.mean(self.critic_weights[0].grad[:, :self.action_dim], dim=0)

        self.optimizer.step()
        return grad

    def target_q(self, next_action_batch, next_state_batch):
        """Get Q-value from target network."""
        with torch.no_grad():
            return self.target(next_action_batch, next_state_batch).view(-1)

    def update_target(self):
        """Soft update target network toward online network."""
        soft_copy(self.target_weights, self.critic_weights)
