
"""
Improved DQN Training for Snake

Key improvements:
1. Better reward shaping (not just binary closer/farther)
2. Adaptive timeout based on snake length
3. Better logging and visualization
4. Learning rate scheduling
"""
import pygame
import random
import numpy as np
from collections import deque
import sys
import matplotlib.pyplot as plt

# Initialize pygame in headless mode
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()

from dqn_agent import DQNAgent

# Game constants
WINDOW_X = 720
WINDOW_Y = 480
BLOCK_SIZE = 10


class SnakeGameTrainer:
    """
    Snake game environment optimized for training

    Key differences from play version:
    - Headless (no rendering) for speed
    - Better reward function
    - Adaptive timeout
    - Detailed statistics tracking
    """

    def __init__(self, agent):
        self.agent = agent
        self.reset_game()

    def reset_game(self):
        """Initialize/reset game state"""
        # Start in center-ish area
        start_x = WINDOW_X // 2
        start_y = WINDOW_Y // 2

        self.snake_pos = [start_x, start_y]
        self.snake_body = [
            [start_x, start_y],
            [start_x - BLOCK_SIZE, start_y],
            [start_x - 2 * BLOCK_SIZE, start_y]
        ]
        self.food_pos = self._spawn_food()
        self.direction = 'RIGHT'
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0  # Track steps without eating
        self.prev_distance = self._manhattan_distance()

        self.agent.reset_episode()

    def _spawn_food(self):
        """Spawn food not on snake"""
        while True:
            pos = [
                random.randrange(1, WINDOW_X // BLOCK_SIZE) * BLOCK_SIZE,
                random.randrange(1, WINDOW_Y // BLOCK_SIZE) * BLOCK_SIZE
            ]
            if pos not in self.snake_body:
                return pos

    def _manhattan_distance(self):
        """Manhattan distance from head to food"""
        return abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])

    def _get_game_state(self):
        """Package current state for agent"""
        return {
            'snake_pos': self.snake_pos.copy(),
            'snake_body': [s.copy() for s in self.snake_body],
            'food_pos': self.food_pos.copy(),
            'direction': self.direction,
            'score': self.score
        }

    def _calculate_reward(self, food_eaten, game_over):
        """
        Improved reward function

        The reward function is CRITICAL for learning. Bad rewards = bad learning.

        Design principles:
        1. Sparse rewards (food, death) should dominate
        2. Shaping rewards should be small and balanced
        3. Don't punish necessary exploration too much

        Rewards:
        - Eating food: +10 (main goal)
        - Death: -10 (main penalty)
        - Moving closer to food: +0.1 * (distance_reduced / max_distance)
        - Moving away: -0.1 * (distance_increased / max_distance)
        - Survival bonus: +0.01 per step (small reward for staying alive)

        Note: Rewards are normalized relative to game size to be consistent.
        """
        if game_over:
            return -10.0

        if food_eaten:
            self.steps_since_food = 0
            return 10.0

        # Distance-based shaping (normalized and balanced)
        current_dist = self._manhattan_distance()
        max_dist = WINDOW_X + WINDOW_Y  # Maximum possible distance

        # Change in distance (positive = got closer)
        dist_change = self.prev_distance - current_dist

        # Normalize the reward
        # If we moved one step closer, dist_change = 10 (one block)
        # Normalize by max possible single-step change
        normalized_change = dist_change / BLOCK_SIZE

        # Small shaping reward (symmetric: +0.1 for closer, -0.1 for farther)
        shaping_reward = 0.1 * normalized_change

        # Tiny survival bonus (encourages not dying)
        survival_bonus = 0.01

        self.prev_distance = current_dist

        return shaping_reward + survival_bonus

    def step(self):
        """Execute one game step"""
        self.steps += 1
        self.steps_since_food += 1

        # Get action from agent
        game_state = self._get_game_state()
        action = self.agent.get_action(game_state)

        # Update direction (prevent 180° turn)
        opposite = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if action != opposite.get(self.direction):
            self.direction = action

        # Move snake
        moves = {'UP': (0, -BLOCK_SIZE), 'DOWN': (0, BLOCK_SIZE),
                 'LEFT': (-BLOCK_SIZE, 0), 'RIGHT': (BLOCK_SIZE, 0)}
        dx, dy = moves[self.direction]
        self.snake_pos[0] += dx
        self.snake_pos[1] += dy

        # Check food
        food_eaten = (self.snake_pos[0] == self.food_pos[0] and
                      self.snake_pos[1] == self.food_pos[1])

        self.snake_body.insert(0, list(self.snake_pos))
        if food_eaten:
            self.score += 10
            self.food_pos = self._spawn_food()
            self.prev_distance = self._manhattan_distance()
        else:
            self.snake_body.pop()

        # Check game over
        game_over = False

        # Wall collision
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= WINDOW_X or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= WINDOW_Y):
            game_over = True

        # Self collision
        if not game_over:
            for block in self.snake_body[1:]:
                if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                    game_over = True
                    break

        # Adaptive timeout: longer snake gets more time
        # Base: 100 steps per length, minimum 200 steps
        timeout = max(200, len(self.snake_body) * 100)
        if self.steps_since_food > timeout:
            game_over = True

        # Calculate reward
        reward = self._calculate_reward(food_eaten, game_over)

        # Train agent
        new_state = self._get_game_state()
        self.agent.learn(reward, new_state, game_over)

        return reward, game_over, self.score

    def play_episode(self):
        """Play one complete episode"""
        self.reset_game()
        total_reward = 0

        while True:
            reward, done, score = self.step()
            total_reward += reward
            if done:
                break

        return total_reward, self.score, self.steps


def plot_training_results(history, save_path='training_results.png'):
    """
    Plot training metrics over time

    Creates a 2x2 grid of plots:
    1. Score per episode (with moving average)
    2. Total reward per episode
    3. Epsilon decay
    4. Loss over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Snake Training Results', fontsize=16, fontweight='bold')

    episodes = range(1, len(history['scores']) + 1)

    # --- Plot 1: Scores ---
    ax1 = axes[0, 0]
    ax1.plot(episodes, history['scores'], alpha=0.3, color='blue', label='Score')
    # Moving average (window of 100)
    if len(history['scores']) >= 100:
        moving_avg = np.convolve(history['scores'], np.ones(100) / 100, mode='valid')
        ax1.plot(range(100, len(history['scores']) + 1), moving_avg,
                 color='red', linewidth=2, label='Moving Avg (100)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Score per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Rewards ---
    ax2 = axes[0, 1]
    ax2.plot(episodes, history['rewards'], alpha=0.3, color='green', label='Reward')
    if len(history['rewards']) >= 100:
        reward_avg = np.convolve(history['rewards'], np.ones(100) / 100, mode='valid')
        ax2.plot(range(100, len(history['rewards']) + 1), reward_avg,
                 color='darkgreen', linewidth=2, label='Moving Avg (100)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Total Reward per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Epsilon ---
    ax3 = axes[1, 0]
    ax3.plot(episodes, history['epsilons'], color='orange', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon) Decay')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # --- Plot 4: Loss ---
    ax4 = axes[1, 1]
    if history['losses'] and any(l > 0 for l in history['losses']):
        ax4.plot(episodes, history['losses'], alpha=0.5, color='purple', label='Avg Loss')
        if len(history['losses']) >= 100:
            loss_avg = np.convolve(history['losses'], np.ones(100) / 100, mode='valid')
            ax4.plot(range(100, len(history['losses']) + 1), loss_avg,
                     color='darkviolet', linewidth=2, label='Moving Avg (100)')
        ax4.legend()
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {save_path}")


def plot_live_update(history, save_path='training_progress.png'):
    """
    Quick plot for periodic updates during training
    Shows just score with moving average
    """
    plt.figure(figsize=(10, 6))

    episodes = range(1, len(history['scores']) + 1)
    plt.plot(episodes, history['scores'], alpha=0.3, color='blue', label='Score')

    if len(history['scores']) >= 100:
        moving_avg = np.convolve(history['scores'], np.ones(100) / 100, mode='valid')
        plt.plot(range(100, len(history['scores']) + 1), moving_avg,
                 color='red', linewidth=2, label='Moving Avg (100)')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'Training Progress - Episode {len(history["scores"])}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def train(num_episodes=1000, save_interval=500, print_interval=50, plot_interval=10):
    """
    Main training loop

    Training DQN requires patience. Here's what to expect:

    Episodes 1-500: Mostly random, learning basics
    - Scores: 0-20 (0-2 food)
    - Agent is exploring, filling replay buffer

    Episodes 500-1500: Starting to learn
    - Scores: 20-50 (2-5 food)
    - Agent learns to avoid walls, finds food sometimes

    Episodes 1500-3000: Refinement
    - Scores: 50-100+ (5-10+ food)
    - Agent develops consistent strategies

    Beyond 3000: Fine-tuning
    - Scores can reach 150+ with good hyperparameters
    """
    print("=" * 60)
    print("DQN Snake Training - Improved Version")
    print("=" * 60)

    # Create agent with 16-dimensional state
    agent = DQNAgent(
        state_size=16,  # Our improved state representation
        action_size=4,
        learning_rate=0.0005,  # Lower = more stable
        gamma=0.99,  # High = care about future
        epsilon=1.0,  # Start fully random
        epsilon_decay=0.997,  # Decay per episode
        epsilon_min=0.01,  # Never go below 1% random
        batch_size=64,
        buffer_size=50000,
        target_update=500,
        use_double_dqn=True  # Use Double DQN
    )

    # Try to load existing model
    agent.load_model('dqn_model_improved.pth')

    game = SnakeGameTrainer(agent)

    # Statistics tracking (for plotting)
    history = {
        'scores': [],
        'rewards': [],
        'epsilons': [],
        'losses': [],
        'steps': []
    }

    # Rolling window for console output
    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    best_score = 0

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Save interval: {save_interval}, Print interval: {print_interval}, Plot interval: {plot_interval}")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        total_reward, score, steps = game.play_episode()

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Get current stats
        stats = agent.get_stats()

        # Track full history (for plotting)
        history['scores'].append(score)
        history['rewards'].append(total_reward)
        history['epsilons'].append(stats['epsilon'])
        history['losses'].append(stats['avg_loss'])
        history['steps'].append(steps)

        # Track rolling stats (for console)
        recent_scores.append(score)
        recent_rewards.append(total_reward)

        # Update best score
        if score > best_score:
            best_score = score
            agent.save_model('dqn_model_best.pth')

        # Print progress
        if episode % print_interval == 0:
            avg_score = np.mean(recent_scores)
            avg_reward = np.mean(recent_rewards)

            print(f"Episode {episode:5d} | "
                  f"Score: {score:3d} | "
                  f"Avg(100): {avg_score:6.1f} | "
                  f"Best: {best_score:3d} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Buffer: {stats['buffer_size']:5d} | "
                  f"Loss: {stats['avg_loss']:.4f}")

        # Save model checkpoint
        if episode % save_interval == 0:
            agent.save_model('dqn_model_improved.pth')
            print(f">>> Checkpoint saved at episode {episode}")

        # Update plots
        if episode % plot_interval == 0:
            plot_live_update(history, 'training_progress.png')
            print(f">>> Progress plot updated")

    # Final save
    agent.save_model('dqn_model_improved.pth')

    # Generate final comprehensive plot
    plot_training_results(history, 'training_results.png')

    # Save training history to file
    np.savez('training_history.npz',
             scores=history['scores'],
             rewards=history['rewards'],
             epsilons=history['epsilons'],
             losses=history['losses'],
             steps=history['steps'])
    print("Training history saved to training_history.npz")

    print("\n" + "=" * 60)
    print(f"Training complete! Best score: {best_score}")
    print(f"Final avg score (last 100): {np.mean(recent_scores):.1f}")
    print(f"Models saved: dqn_model_improved.pth, dqn_model_best.pth")
    print(f"Plots saved: training_results.png, training_progress.png")
    print("=" * 60)

    return agent, history


if __name__ == '__main__':
    # You can adjust these parameters
    agent, history = train(
        num_episodes=3000,  # Total episodes to train
        save_interval=500,  # Save model every N episodes
        print_interval=100,  # Print stats every N episodes
        plot_interval=10  # Update progress plot every N episodes
    )