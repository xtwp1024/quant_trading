"""
DDPG Agent for Continuous Portfolio Control.

Adapted from PyTorch-DDPG-Stock-Trading (JohsuaWu1997).
Key features:
- DDPG with 3-timestep state window
- Actor-Critic networks with separate target networks
- Ornstein-Uhlenbeck exploration noise for continuous action spaces
- Min-max scaling normalization on replay buffer
- Batch training with GPU support

Reference:
    - Continuous Control with Deep Reinforcement Learning (Lillicrap et al., 2015)
    - Deep Deterministic Policy Gradient (DDPG)
"""

import torch
import torch.nn.functional as F

from .ddpg_networks import Actor, Critic
from .ou_noise import OUNoise

# Default gamma (discount factor) - high value for long-term optimization
GAMMA = 0.9999999993340943687843739933894


def min_max_scale(data):
    """Min-max scaling with handling for zero-range data.

    Args:
        data: 1D or 2D tensor to scale

    Returns:
        Scaled data with same shape
    """
    data_min = torch.min(data, 0).values.view(1, -1)
    data_max = torch.max(data, 0).values.view(1, -1)
    data_max[data_max - data_min == 0] = 0  # Handle constant features
    return (data - data_min) / (data_max - data_min)


class DDPGAgent:
    """DDPG Agent for continuous portfolio control.

    Implements the DDPG (Deep Deterministic Policy Gradient) algorithm
    with 3-timestep state window and Ornstein-Uhlenbeck exploration noise.
    """

    def __init__(self, env, time_steps=12, hidden_dim=1024, device=None):
        """Initialize DDPG agent.

        Args:
            env: Market environment with observation_space and action_space
            time_steps: Number of timesteps in state window (default: 12)
            hidden_dim: Hidden layer dimension for networks (default: 1024)
            device: torch device (default: cpu)
        """
        self.device = device or torch.device("cpu")
        self.name = 'DDPG'

        # Environment properties
        self.scale = env.asset
        self.unit = env.unit
        self.seed = env.rd_seed

        # DDPG hyperparameters
        self.time_dim = time_steps
        self.state_dim = env.observation_space.shape[1]  # Number of stocks
        self.action_dim = env.action_space.shape[0]  # Action dimension
        self.batch_size = 64
        self.memory_size = self.time_dim + self.batch_size * 10
        self.start_size = self.time_dim + self.batch_size * 2

        # Initialize actor & critic networks
        self.actor_network = Actor(
            self.time_dim, self.state_dim, self.action_dim, hidden_dim, self.device
        )
        self.critic_network = Critic(
            self.time_dim, self.state_dim, self.action_dim, hidden_dim, self.device
        )

        # Initialize replay buffer
        self.replay_state = torch.zeros(
            (self.start_size - 1, 3, self.state_dim), device=self.device
        )
        self.replay_next_state = torch.zeros(
            (self.start_size - 1, 3, self.state_dim), device=self.device
        )
        self.replay_action = torch.zeros(
            (self.start_size - 1, 1, self.state_dim), device=self.device
        )
        self.replay_reward = torch.zeros(
            (self.start_size - 1,), device=self.device
        )

        # Initialize Ornstein-Uhlenbeck noise for exploration
        self.exploration_noise = OUNoise(self.action_dim, sigma=0.01 / self.action_dim)

        self.initial()

    def initial(self):
        """Reset agent state for new episode."""
        self.steps = 0
        self.action = torch.zeros(self.action_dim, device=self.device)
        self.replay_state = torch.zeros(
            (self.start_size - 1, 3, self.state_dim), device=self.device
        )
        self.replay_next_state = torch.zeros(
            (self.start_size - 1, 3, self.state_dim), device=self.device
        )
        self.replay_action = torch.zeros(
            (self.start_size - 1, self.state_dim), device=self.device
        )
        self.replay_reward = torch.zeros(
            (self.start_size - 1,), device=self.device
        )

    def train_on_batch(self):
        """Train on a batch of samples from replay buffer.

        Performs:
        1. Sample random minibatch of transitions
        2. Compute TD target: y = r + gamma * Q_target(s', a')
        3. Update critic to minimize MSE(y, Q(s, a))
        4. Update actor using policy gradient from critic
        5. Soft update target networks
        """
        # Sample a random minibatch of N transitions from replay buffer
        sample = torch.randint(
            self.time_dim,
            self.replay_reward.shape[0],
            [self.batch_size],
            device=self.device
        )
        # Get indices for time window (time_dim timesteps before each sample)
        index = torch.stack([sample - i for i in range(self.time_dim, 0, -1)]).t().reshape(-1)

        # Min-max scale state data
        state_data = min_max_scale(self.replay_state[:, 0, :])
        amount_data = min_max_scale(self.replay_state[:, 2, :])
        next_state_data = min_max_scale(self.replay_next_state[:, 0, :])
        next_amount_data = min_max_scale(self.replay_next_state[:, 2, :])

        # Build state batches
        state_batch = torch.index_select(state_data, 0, index).view(self.batch_size, -1)
        amount_data = torch.index_select(amount_data, 0, sample).view(self.batch_size, -1)
        state_batch = torch.cat([state_batch, amount_data], dim=1)

        next_state_batch = torch.index_select(next_state_data, 0, index).view(self.batch_size, -1)
        next_amount_data = torch.index_select(next_amount_data, 0, sample).view(self.batch_size, -1)
        next_state_batch = torch.cat([next_state_batch, next_amount_data], dim=1)

        action_batch = torch.index_select(self.replay_action / self.unit, 0, sample)
        reward_batch = torch.index_select(self.replay_reward, 0, sample)

        # Calculate y_batch (TD target)
        next_action_batch = self.actor_network.target_action(next_state_batch)
        q_batch = self.critic_network.target_q(next_action_batch, next_state_batch)
        y_batch = torch.add(reward_batch, q_batch, alpha=GAMMA).view(-1, 1)

        # Train actor-critic by target loss
        self.actor_network.train(
            self.critic_network.train(y_batch, action_batch, state_batch)
        )

        # Update target networks by soft update
        self.actor_network.update_target()
        self.critic_network.update_target()

    def perceive(self, state, action, reward, next_state, done):
        """Store transition in replay buffer.

        Args:
            state: Current state (3, state_dim) tensor
            action: Action taken (action_dim) tensor
            reward: Reward received
            next_state: Next state (3, state_dim) tensor
            done: Whether episode is done
        """
        if self.steps < self.start_size - 1:
            # Still filling initial buffer
            self.replay_state[self.steps] = state
            self.replay_next_state[self.steps] = next_state
            self.replay_action[self.steps] = action
            self.replay_reward[self.steps] = reward
        else:
            # Buffer is full, use FIFO replacement
            if self.steps >= self.memory_size:
                self.replay_state = self.replay_state[1:]
                self.replay_next_state = self.replay_next_state[1:]
                self.replay_action = self.replay_action[1:]
                self.replay_reward = self.replay_reward[1:]

            self.replay_state = torch.cat((self.replay_state, state.unsqueeze(0)), dim=0)
            self.replay_next_state = torch.cat((self.replay_next_state, next_state.unsqueeze(0)), dim=0)
            self.replay_action = torch.cat((self.replay_action, action.unsqueeze(0)), dim=0)
            self.replay_reward = torch.cat((self.replay_reward, reward.unsqueeze(0)), dim=0)

        self.steps += 1

    def act(self, next_state, portfolio):
        """Choose action given state using actor network with exploration.

        Args:
            next_state: Current state (3, state_dim) tensor
            portfolio: Current portfolio value

        Returns:
            Action tensor (portfolio allocation in units)
        """
        if self.steps > self.start_size:
            # Build input for actor network
            next_state_data = min_max_scale(
                self.replay_next_state[:, 0, :]
            )[-self.time_dim:].view(1, -1)
            next_amount_data = min_max_scale(
                self.replay_next_state[:, 2, :]
            )[-1].view(1, -1)
            next_state_data = torch.cat([next_state_data, next_amount_data], dim=1)

            # Train on batch
            self.train_on_batch()

            # Get action from target actor with OU noise
            allocation = self.actor_network.target_action(next_state_data).data.view(-1)
            allocation += torch.tensor(
                self.exploration_noise.noise().tolist(), device=self.device
            )

            # Post-process action: ensure valid portfolio weights
            allocation[allocation < 0] = 0  # No negative positions
            allocation /= sum(allocation)  # Normalize to sum to 1

            # Convert to units
            allocation = torch.floor(
                portfolio * allocation / next_state[1, :] / self.unit
            ) * self.unit
            self.action = allocation

        return self.action.clone()


def train_ddpg(env, agent, num_epochs, print_every=100):
    """Train DDPG agent on environment.

    Args:
        env: Market environment
        agent: DDPGAgent instance
        num_epochs: Number of training epochs
        print_every: Print frequency

    Returns:
        Trained agent
    """
    for t in range(num_epochs):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state, env.portfolio)
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

        if t % print_every == 0:
            print(f'Epoch {t}: Portfolio = {int(env.portfolio)}, Total Reward = {episode_reward:.2f}')

    return agent


if __name__ == '__main__':
    # Test DDPG agent
    from .ddpg_market import create_sample_market_data, DDPGMarketEnv

    # Create sample market
    data = create_sample_market_data(timesteps=100, stocks=3)
    env = DDPGMarketEnv(data, seed=0, asset=1000000.0)

    # Create agent
    agent = DDPGAgent(env, time_steps=12, hidden_dim=256)

    # Quick training test
    print("Testing DDPG agent...")
    state = env.reset()
    for _ in range(200):
        action = agent.act(state, env.portfolio)
        next_state, reward, done, _ = env.step(action)
        agent.perceive(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    print(f"Final portfolio: {env.portfolio}")
    print("DDPG agent test completed!")
