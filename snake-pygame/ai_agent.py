# ai_agent.py
import random
import pickle
import os
import numpy as np


class BaseAgent:
    def __init__(self):
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    def get_action(self, game_state):
        raise NotImplementedError("Subclasses should implement this!")


class RandomAgent(BaseAgent):
    def get_action(self, game_state):
        return random.choice(self.actions)


class QLearningAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        """
        Q-Learning Agent for Snake Game

        Args:
            alpha: Learning rate (how much we update Q-values)
            gamma: Discount factor (how much we value future rewards)
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum epsilon value
        """
        super().__init__()
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # For tracking state transitions
        self.last_state = None
        self.last_action = None

        # Training statistics
        self.total_rewards = 0

    def get_state(self, game_state):
        """
        Convert game state to a simplified tuple for Q-learning.
        We use relative positions and danger indicators instead of absolute coordinates.
        """
        head_x, head_y = game_state['snake_pos']
        food_x, food_y = game_state['food_pos']
        snake_body = game_state['snake_body']
        direction = game_state['direction']

        # Check for danger in each direction (collision with wall or body)
        danger_straight = self._is_danger_straight(head_x, head_y, direction, snake_body)
        danger_left = self._is_danger_left(head_x, head_y, direction, snake_body)
        danger_right = self._is_danger_right(head_x, head_y, direction, snake_body)

        # Food direction relative to current direction
        food_straight = self._is_food_straight(head_x, head_y, food_x, food_y, direction)
        food_left = self._is_food_left(head_x, head_y, food_x, food_y, direction)
        food_right = self._is_food_right(head_x, head_y, food_x, food_y, direction)

        # Create state tuple (must be hashable for dictionary)
        state = (
            danger_straight,
            danger_left,
            danger_right,
            food_straight,
            food_left,
            food_right,
            direction
        )

        return state

    def _is_collision(self, x, y, snake_body):
        """Check if position (x,y) would cause a collision"""
        # Wall collision
        if x < 0 or x >= 720 or y < 0 or y >= 480:
            return True
        # Body collision
        for block in snake_body[1:]:
            if x == block[0] and y == block[1]:
                return True
        return False

    def _is_danger_straight(self, head_x, head_y, direction, snake_body):
        if direction == 'UP':
            return self._is_collision(head_x, head_y - 10, snake_body)
        elif direction == 'DOWN':
            return self._is_collision(head_x, head_y + 10, snake_body)
        elif direction == 'LEFT':
            return self._is_collision(head_x - 10, head_y, snake_body)
        else:  # RIGHT
            return self._is_collision(head_x + 10, head_y, snake_body)

    def _is_danger_left(self, head_x, head_y, direction, snake_body):
        if direction == 'UP':
            return self._is_collision(head_x - 10, head_y, snake_body)
        elif direction == 'DOWN':
            return self._is_collision(head_x + 10, head_y, snake_body)
        elif direction == 'LEFT':
            return self._is_collision(head_x, head_y + 10, snake_body)
        else:  # RIGHT
            return self._is_collision(head_x, head_y - 10, snake_body)

    def _is_danger_right(self, head_x, head_y, direction, snake_body):
        if direction == 'UP':
            return self._is_collision(head_x + 10, head_y, snake_body)
        elif direction == 'DOWN':
            return self._is_collision(head_x - 10, head_y, snake_body)
        elif direction == 'LEFT':
            return self._is_collision(head_x, head_y - 10, snake_body)
        else:  # RIGHT
            return self._is_collision(head_x, head_y + 10, snake_body)

    def _is_food_straight(self, head_x, head_y, food_x, food_y, direction):
        if direction == 'UP':
            return food_y < head_y
        elif direction == 'DOWN':
            return food_y > head_y
        elif direction == 'LEFT':
            return food_x < head_x
        else:  # RIGHT
            return food_x > head_x

    def _is_food_left(self, head_x, head_y, food_x, food_y, direction):
        if direction == 'UP':
            return food_x < head_x
        elif direction == 'DOWN':
            return food_x > head_x
        elif direction == 'LEFT':
            return food_y > head_y
        else:  # RIGHT
            return food_y < head_y

    def _is_food_right(self, head_x, head_y, food_x, food_y, direction):
        if direction == 'UP':
            return food_x > head_x
        elif direction == 'DOWN':
            return food_x < head_x
        elif direction == 'LEFT':
            return food_y < head_y
        else:  # RIGHT
            return food_y > head_y

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair"""
        return self.q_table.get((state, action), 0.0)

    def get_action(self, game_state):
        """
        Choose action using epsilon-greedy strategy
        """
        state = self.get_state(game_state)
        current_direction = game_state['direction']

        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon:
            # Explore: random action (but valid - no 180 degree turns)
            valid_actions = self._get_valid_actions(current_direction)
            action = random.choice(valid_actions)
        else:
            # Exploit: choose best action based on Q-values
            valid_actions = self._get_valid_actions(current_direction)
            q_values = [(action, self.get_q_value(state, action)) for action in valid_actions]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [action for action, q in q_values if q == max_q]
            action = random.choice(best_actions)

        # Store for learning
        self.last_state = state
        self.last_action = action

        return action

    def _get_valid_actions(self, current_direction):
        """Get valid actions (no 180-degree turns)"""
        opposite = {
            'UP': 'DOWN',
            'DOWN': 'UP',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT'
        }
        return [a for a in self.actions if a != opposite[current_direction]]

    def learn(self, reward, new_game_state, done):
        """
        Update Q-values using the Q-learning update rule:
        Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',a')) - Q(s,a))
        """
        if self.last_state is None:
            return

        self.total_rewards += reward

        # Get current Q-value
        old_q = self.get_q_value(self.last_state, self.last_action)

        if done:
            # Terminal state: no future rewards
            new_q = old_q + self.alpha * (reward - old_q)
        else:
            # Non-terminal: consider future rewards
            new_state = self.get_state(new_game_state)
            future_q_values = [self.get_q_value(new_state, a) for a in self.actions]
            max_future_q = max(future_q_values)
            new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)

        # Update Q-table
        self.q_table[(self.last_state, self.last_action)] = new_q

    def decay_epsilon(self):
        """Decay exploration rate over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        """Reset episode-specific variables"""
        self.last_state = None
        self.last_action = None
        self.total_rewards = 0

    def save_model(self, filename='q_learning_model.pkl'):
        """Save Q-table to file"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename='q_learning_model.pkl'):
        """Load Q-table from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.q_table = model_data['q_table']
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.alpha = model_data.get('alpha', self.alpha)
            self.gamma = model_data.get('gamma', self.gamma)
            print(f"Model loaded from {filename}")
            print(f"Q-table size: {len(self.q_table)} state-action pairs")
            return True
        else:
            print(f"No saved model found at {filename}")
            return False

    def get_stats(self):
        """Get training statistics"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'total_rewards': self.total_rewards
        }