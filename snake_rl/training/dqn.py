import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pygame

from snake_rl.agents.dqn_agent import DQNAgent
from snake_rl.paths import (
    DQN_BEST_MODEL_PATH,
    DQN_HISTORY_PATH,
    DQN_MODEL_PATH,
    DQN_PROGRESS_PLOT_PATH,
    DQN_RESULTS_PLOT_PATH,
)


os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()

WINDOW_X = 720
WINDOW_Y = 480
BLOCK_SIZE = 10


class SnakeGameTrainer:
    """
    Snake game environment optimized for training
    """

    def __init__(self, agent):
        self.agent = agent
        self.reset_game()

    def reset_game(self):
        """Initialize or reset game state"""
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
        self.steps_since_food = 0
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
            'snake_body': [segment.copy() for segment in self.snake_body],
            'food_pos': self.food_pos.copy(),
            'direction': self.direction,
            'score': self.score
        }

    def _calculate_reward(self, food_eaten, game_over):
        """
        Improved reward function
        """
        if game_over:
            return -10.0

        if food_eaten:
            self.steps_since_food = 0
            return 10.0

        current_dist = self._manhattan_distance()
        dist_change = self.prev_distance - current_dist
        normalized_change = dist_change / BLOCK_SIZE
        shaping_reward = 0.1 * normalized_change
        survival_bonus = 0.01

        self.prev_distance = current_dist

        return shaping_reward + survival_bonus

    def step(self):
        """Execute one game step"""
        self.steps += 1
        self.steps_since_food += 1

        game_state = self._get_game_state()
        action = self.agent.get_action(game_state)

        opposite = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if action != opposite.get(self.direction):
            self.direction = action

        moves = {'UP': (0, -BLOCK_SIZE), 'DOWN': (0, BLOCK_SIZE),
                 'LEFT': (-BLOCK_SIZE, 0), 'RIGHT': (BLOCK_SIZE, 0)}
        dx, dy = moves[self.direction]
        self.snake_pos[0] += dx
        self.snake_pos[1] += dy

        food_eaten = (self.snake_pos[0] == self.food_pos[0] and
                      self.snake_pos[1] == self.food_pos[1])

        self.snake_body.insert(0, list(self.snake_pos))
        if food_eaten:
            self.score += 10
            self.food_pos = self._spawn_food()
            self.prev_distance = self._manhattan_distance()
        else:
            self.snake_body.pop()

        game_over = False

        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= WINDOW_X or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= WINDOW_Y):
            game_over = True

        if not game_over:
            for block in self.snake_body[1:]:
                if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                    game_over = True
                    break

        timeout = max(200, len(self.snake_body) * 100)
        if self.steps_since_food > timeout:
            game_over = True

        reward = self._calculate_reward(food_eaten, game_over)

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


def plot_training_results(history, save_path=DQN_RESULTS_PLOT_PATH):
    """
    Plot training metrics over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Snake Training Results', fontsize=16, fontweight='bold')

    episodes = range(1, len(history['scores']) + 1)

    ax1 = axes[0, 0]
    ax1.plot(episodes, history['scores'], alpha=0.3, color='blue', label='Score')
    if len(history['scores']) >= 100:
        moving_avg = np.convolve(history['scores'], np.ones(100) / 100, mode='valid')
        ax1.plot(range(100, len(history['scores']) + 1), moving_avg,
                 color='red', linewidth=2, label='Moving Avg (100)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Score per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    ax3 = axes[1, 0]
    ax3.plot(episodes, history['epsilons'], color='orange', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon) Decay')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    ax4 = axes[1, 1]
    if history['losses'] and any(loss > 0 for loss in history['losses']):
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


def plot_live_update(history, save_path=DQN_PROGRESS_PLOT_PATH):
    """
    Quick plot for periodic updates during training
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
    """
    print("=" * 60)
    print("DQN Snake Training - Improved Version")
    print("=" * 60)

    agent = DQNAgent(
        state_size=16,
        action_size=4,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.01,
        batch_size=64,
        buffer_size=50000,
        target_update=500,
        use_double_dqn=True
    )

    agent.load_model(DQN_MODEL_PATH)

    game = SnakeGameTrainer(agent)

    history = {
        'scores': [],
        'rewards': [],
        'epsilons': [],
        'losses': [],
        'steps': []
    }

    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    best_score = 0

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Save interval: {save_interval}, Print interval: {print_interval}, Plot interval: {plot_interval}")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        total_reward, score, steps = game.play_episode()

        agent.decay_epsilon()
        stats = agent.get_stats()

        history['scores'].append(score)
        history['rewards'].append(total_reward)
        history['epsilons'].append(stats['epsilon'])
        history['losses'].append(stats['avg_loss'])
        history['steps'].append(steps)

        recent_scores.append(score)
        recent_rewards.append(total_reward)

        if score > best_score:
            best_score = score
            agent.save_model(DQN_BEST_MODEL_PATH)

        if episode % print_interval == 0:
            avg_score = np.mean(recent_scores)
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode:5d} | "
                  f"Score: {score:3d} | "
                  f"Avg(100): {avg_score:6.1f} | "
                  f"Best: {best_score:3d} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Buffer: {stats['buffer_size']:5d} | "
                  f"Loss: {stats['avg_loss']:.4f} | "
                  f"Reward Avg: {avg_reward:7.2f}")

        if episode % save_interval == 0:
            agent.save_model(DQN_MODEL_PATH)
            print(f">>> Checkpoint saved at episode {episode}")

        if episode % plot_interval == 0:
            plot_live_update(history, DQN_PROGRESS_PLOT_PATH)
            print(">>> Progress plot updated")

    agent.save_model(DQN_MODEL_PATH)
    plot_training_results(history, DQN_RESULTS_PLOT_PATH)

    np.savez(DQN_HISTORY_PATH,
             scores=history['scores'],
             rewards=history['rewards'],
             epsilons=history['epsilons'],
             losses=history['losses'],
             steps=history['steps'])
    print(f"Training history saved to {DQN_HISTORY_PATH}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best score: {best_score}")
    print(f"Final avg score (last 100): {np.mean(recent_scores):.1f}")
    print(f"Models saved: {DQN_MODEL_PATH.name}, {DQN_BEST_MODEL_PATH.name}")
    print(f"Plots saved: {DQN_RESULTS_PLOT_PATH.name}, {DQN_PROGRESS_PLOT_PATH.name}")
    print("=" * 60)

    return agent, history


def main():
    agent, history = train(
        num_episodes=3000,
        save_interval=500,
        print_interval=100,
        plot_interval=10
    )
    return agent, history
