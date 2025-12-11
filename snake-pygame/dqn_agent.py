
"""
Improved Deep Q-Network (DQN) Agent for Snake Game
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Experience replay buffer - stores transitions for training"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Neural network for Q-function approximation

    Architecture explanation:
    - Input layer: Takes state vector (16 features)
    - Hidden layers: Process and find patterns in the state
    - Output layer: Produces Q-value for each action (4 actions)

    The Q-value represents "expected future reward if I take this action"
    """

    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()

        # Deeper network with more capacity
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

        # Initialize weights using Xavier initialization
        # This helps with gradient flow at the start of training
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    Improved DQN Agent with Double DQN and better training

    What is DQN?
    -----------
    Q-Learning learns Q(s,a) = expected reward for taking action 'a' in state 's'.
    Traditional Q-Learning uses a table, but that doesn't scale to large state spaces.

    DQN uses a neural network to approximate Q(s,a). Given a state, the network
    outputs Q-values for all actions, and we pick the action with highest Q-value.

    What is Double DQN?
    ------------------
    Standard DQN has a problem: it uses the same network to:
    1. SELECT the best action (argmax)
    2. EVALUATE that action's value

    This leads to overestimation because any noise in Q-values will bias
    the max operation upward.

    Double DQN fixes this by using:
    - Policy network to SELECT actions
    - Target network to EVALUATE those actions

    This decorrelates selection and evaluation, reducing overestimation.
    """

    def __init__(self, state_size=16, action_size=4,
                 learning_rate=0.0005,  # Lower LR for stability
                 gamma=0.99,  # Higher gamma - care more about future
                 epsilon=1.0,
                 epsilon_decay=0.997,  # Slower decay
                 epsilon_min=0.01,
                 batch_size=64,
                 buffer_size=50000,  # Larger buffer
                 target_update=500,  # More frequent updates
                 use_double_dqn=True):  # Enable Double DQN

        self.state_size = state_size
        self.action_size = action_size
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double_dqn = use_double_dqn

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # Small regularization
        )

        # Huber loss (SmoothL1) - more stable than MSE
        # MSE squares errors, so a -100 death penalty creates huge gradients
        # Huber loss is linear for large errors, preventing gradient explosions
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Tracking
        self.steps = 0
        self.last_state = None
        self.last_action = None
        self.total_rewards = 0
        self.losses = []

        # Game constants (needed for state calculation)
        self.window_x = 720
        self.window_y = 480
        self.block_size = 10

    def get_state_vector(self, game_state):
        """
        Convert game state to feature vector for neural network

        State representation is CRUCIAL for learning. We need features that:
        1. Capture all relevant information
        2. Are normalized (similar scales)
        3. Are meaningful (help distinguish good vs bad situations)

        Our 16 features:
        - Danger indicators (3): Is there danger straight/left/right?
        - Food direction (4): Is food up/down/left/right?
        - Current direction (4): One-hot encoding of current direction
        - Distance to food (2): Normalized x and y distance
        - Distance to walls (2): How close are we to walls?
        - Snake length (1): Normalized length
        """
        head_x, head_y = game_state['snake_pos']
        food_x, food_y = game_state['food_pos']
        snake_body = game_state['snake_body']
        direction = game_state['direction']

        # === DANGER DETECTION (relative to current direction) ===
        danger_straight = float(self._is_danger_straight(head_x, head_y, direction, snake_body))
        danger_left = float(self._is_danger_left(head_x, head_y, direction, snake_body))
        danger_right = float(self._is_danger_right(head_x, head_y, direction, snake_body))

        # === FOOD DIRECTION (absolute, not relative) ===
        # This gives more information than relative direction
        food_up = float(food_y < head_y)
        food_down = float(food_y > head_y)
        food_left = float(food_x < head_x)
        food_right = float(food_x > head_x)

        # === CURRENT DIRECTION (one-hot) ===
        dir_up = float(direction == 'UP')
        dir_down = float(direction == 'DOWN')
        dir_left = float(direction == 'LEFT')
        dir_right = float(direction == 'RIGHT')

        # === NORMALIZED DISTANCES ===
        # These give the agent a sense of "how far" not just "which direction"
        # Normalized to [-1, 1] range
        food_dist_x = (food_x - head_x) / self.window_x
        food_dist_y = (food_y - head_y) / self.window_y

        # Distance to nearest wall (normalized)
        wall_dist_x = min(head_x, self.window_x - head_x) / (self.window_x / 2)
        wall_dist_y = min(head_y, self.window_y - head_y) / (self.window_y / 2)

        # === SNAKE LENGTH ===
        # Normalized - longer snake means more obstacles
        snake_length = len(snake_body) / 100.0

        state = np.array([
            # Danger (3)
            danger_straight, danger_left, danger_right,
            # Food direction (4)
            food_up, food_down, food_left, food_right,
            # Current direction (4)
            dir_up, dir_down, dir_left, dir_right,
            # Distances (4)
            food_dist_x, food_dist_y,
            wall_dist_x, wall_dist_y,
            # Length (1)
            snake_length
        ], dtype=np.float32)

        return state

    def _is_collision(self, x, y, snake_body):
        """Check if position causes collision with wall or self"""
        # Wall collision
        if x < 0 or x >= self.window_x or y < 0 or y >= self.window_y:
            return True
        # Self collision (skip head)
        for block in snake_body[1:]:
            if x == block[0] and y == block[1]:
                return True
        return False

    def _is_danger_straight(self, head_x, head_y, direction, snake_body):
        """Check danger in current direction"""
        moves = {'UP': (0, -10), 'DOWN': (0, 10), 'LEFT': (-10, 0), 'RIGHT': (10, 0)}
        dx, dy = moves[direction]
        return self._is_collision(head_x + dx, head_y + dy, snake_body)

    def _is_danger_left(self, head_x, head_y, direction, snake_body):
        """Check danger to the left of current direction"""
        left_of = {'UP': 'LEFT', 'DOWN': 'RIGHT', 'LEFT': 'DOWN', 'RIGHT': 'UP'}
        moves = {'UP': (0, -10), 'DOWN': (0, 10), 'LEFT': (-10, 0), 'RIGHT': (10, 0)}
        dx, dy = moves[left_of[direction]]
        return self._is_collision(head_x + dx, head_y + dy, snake_body)

    def _is_danger_right(self, head_x, head_y, direction, snake_body):
        """Check danger to the right of current direction"""
        right_of = {'UP': 'RIGHT', 'DOWN': 'LEFT', 'LEFT': 'UP', 'RIGHT': 'DOWN'}
        moves = {'UP': (0, -10), 'DOWN': (0, 10), 'LEFT': (-10, 0), 'RIGHT': (10, 0)}
        dx, dy = moves[right_of[direction]]
        return self._is_collision(head_x + dx, head_y + dy, snake_body)

    def get_action(self, game_state):
        """
        Choose action using epsilon-greedy policy

        Epsilon-greedy balances exploration vs exploitation:
        - With probability epsilon: take random action (explore)
        - With probability 1-epsilon: take best known action (exploit)

        Early training: high epsilon -> lots of exploration
        Late training: low epsilon -> mostly exploitation
        """
        state = self.get_state_vector(game_state)
        current_direction = game_state['direction']
        valid_actions = self._get_valid_actions(current_direction)

        if random.random() < self.epsilon:
            # EXPLORE: Random valid action
            action = random.choice(valid_actions)
        else:
            # EXPLOIT: Best action according to network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)

                # Only consider valid actions (no 180° turns)
                valid_indices = [self.actions.index(a) for a in valid_actions]
                valid_q = q_values[0][valid_indices]
                best_valid_idx = valid_indices[valid_q.argmax().item()]
                action = self.actions[best_valid_idx]

        # Store for learning
        self.last_state = state
        self.last_action = self.actions.index(action)

        return action

    def _get_valid_actions(self, current_direction):
        """Get valid actions (can't do 180° turn)"""
        opposite = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        return [a for a in self.actions if a != opposite[current_direction]]

    def learn(self, reward, new_game_state, done):
        """
        Store experience and train

        The learning process:
        1. Store (s, a, r, s', done) in replay buffer
        2. Sample random batch from buffer
        3. Compute target Q-values using Bellman equation
        4. Update network to minimize prediction error
        """
        if self.last_state is None:
            return

        self.total_rewards += reward

        # Store experience
        next_state = self.get_state_vector(new_game_state)
        self.memory.push(self.last_state, self.last_action, reward, next_state, done)

        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size:
            loss = self._train_step()
            if loss is not None:
                self.losses.append(loss)

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _train_step(self):
        """
        One training step using Double DQN

        Standard DQN target:
            Q_target = r + γ * max_a'[Q_target(s', a')]

        Double DQN target:
            a* = argmax_a'[Q_policy(s', a')]  # Policy net SELECTS action
            Q_target = r + γ * Q_target(s', a*)  # Target net EVALUATES it

        This reduces overestimation bias significantly!
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the actions we actually took
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: policy net selects, target net evaluates
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(1)[0]

            # Bellman equation: Q(s,a) = r + γ * Q(s', a')
            # If done, there's no next state, so just use reward
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        """Reset for new episode"""
        self.last_state = None
        self.last_action = None
        self.total_rewards = 0

    def save_model(self, filename='dqn_model_improved.pth'):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='dqn_model_improved.pth'):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"No model found at {filename}")
            return False

    def get_stats(self):
        """Get current training statistics"""
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
        return {
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards,
            'buffer_size': len(self.memory),
            'steps': self.steps,
            'avg_loss': avg_loss
        }