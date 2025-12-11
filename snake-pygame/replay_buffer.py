# replay_buffer.py
"""
Experience Replay Buffer for DQN
Stores experiences and samples random batches for training
"""
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Experience replay buffer

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of experiences

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions = np.array([exp[1] for exp in batch], dtype=np.int64)
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()