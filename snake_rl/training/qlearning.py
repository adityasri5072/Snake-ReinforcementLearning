from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.game.snake_game import SnakeGame
from snake_rl.paths import (
    Q_LEARNING_BEST_MODEL_PATH,
    Q_LEARNING_HISTORY_PATH,
    Q_LEARNING_MODEL_PATH,
    Q_LEARNING_PROGRESS_PLOT_PATH,
    Q_LEARNING_RESULTS_PLOT_PATH,
)


def plot_training_results(history, save_path=Q_LEARNING_RESULTS_PLOT_PATH):
    """
    Plot Q-Learning training metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning Snake Training Results', fontsize=16, fontweight='bold')

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
    ax4.plot(episodes, history['qtable_sizes'], color='purple', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Q-Table Size (states)')
    ax4.set_title('Q-Table Growth (Unique States Learned)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {save_path}")


def plot_live_update(history, save_path=Q_LEARNING_PROGRESS_PLOT_PATH):
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
    plt.title(f'Q-Learning Training Progress - Episode {len(history["scores"])}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def train(num_episodes=2000, save_interval=500, print_interval=50, plot_interval=100):
    """
    Train Q-Learning agent
    """
    print("=" * 60)
    print("Q-LEARNING SNAKE TRAINING")
    print("=" * 60)

    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    if agent.load_model(Q_LEARNING_MODEL_PATH):
        print("Loaded existing model - continuing training")
    else:
        print("Starting fresh training")

    game = SnakeGame(agent, training_mode=True, fps=0)

    history = {
        'scores': [],
        'rewards': [],
        'epsilons': [],
        'qtable_sizes': [],
        'steps': []
    }

    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    best_score = 0
    best_avg_score = 0

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Save interval: {save_interval}, Print interval: {print_interval}")
    print(f"Initial Q-table size: {len(agent.q_table)}")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        total_reward, score, steps = game.play_episode()

        agent.decay_epsilon()
        stats = agent.get_stats()

        history['scores'].append(score)
        history['rewards'].append(total_reward)
        history['epsilons'].append(stats['epsilon'])
        history['qtable_sizes'].append(stats['q_table_size'])
        history['steps'].append(steps)

        recent_scores.append(score)
        recent_rewards.append(total_reward)

        if score > best_score:
            best_score = score
            agent.save_model(Q_LEARNING_BEST_MODEL_PATH)

        avg_score = np.mean(recent_scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        if episode % print_interval == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode:5d} | "
                  f"Score: {score:3d} | "
                  f"Avg(100): {avg_score:6.1f} | "
                  f"Best: {best_score:3d} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Q-table: {stats['q_table_size']:4d} states | "
                  f"Reward Avg: {avg_reward:7.2f}")

        if episode % save_interval == 0:
            agent.save_model(Q_LEARNING_MODEL_PATH)
            print(f">>> Checkpoint saved at episode {episode}")

        if episode % plot_interval == 0:
            plot_live_update(history, Q_LEARNING_PROGRESS_PLOT_PATH)
            print(">>> Progress plot updated")

    agent.save_model(Q_LEARNING_MODEL_PATH)

    plot_training_results(history, Q_LEARNING_RESULTS_PLOT_PATH)

    np.savez(Q_LEARNING_HISTORY_PATH,
             scores=history['scores'],
             rewards=history['rewards'],
             epsilons=history['epsilons'],
             qtable_sizes=history['qtable_sizes'],
             steps=history['steps'])
    print(f"Training history saved to {Q_LEARNING_HISTORY_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Score: {best_score}")
    print(f"Best Average Score: {best_avg_score:.1f}")
    print(f"Final Average (last 100): {np.mean(recent_scores):.1f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Final Q-Table Size: {len(agent.q_table)} states")
    print(f"Models saved: {Q_LEARNING_MODEL_PATH.name}, {Q_LEARNING_BEST_MODEL_PATH.name}")
    print(f"Plots saved: {Q_LEARNING_RESULTS_PLOT_PATH.name}, {Q_LEARNING_PROGRESS_PLOT_PATH.name}")
    print("=" * 60)

    return agent, history


def main():
    """Main entry point with training presets"""
    print("\n" + "=" * 60)
    print("Q-LEARNING TRAINING PRESETS")
    print("=" * 60)
    print("1. Quick train (500 episodes) - ~5 minutes")
    print("2. Standard train (1000 episodes) - ~10 minutes")
    print("3. Extended train (2000 episodes) - ~20 minutes")
    print("4. Full train (5000 episodes) - ~50 minutes")
    print("5. Custom")

    choice = input("\nChoose (1-5): ")

    if choice == '1':
        train(500)
    elif choice == '2':
        train(1000)
    elif choice == '3':
        train(2000)
    elif choice == '4':
        train(5000)
    elif choice == '5':
        episodes = int(input("Number of episodes: "))
        train(episodes)
    else:
        print("Invalid choice, running standard training.")
        train(1000)
