# train_qlearning.py
"""
Q-Learning Training for Snake with Visualization

Features:
- Training progress tracking
- Live plotting during training
- Comprehensive final plots
- Training history saved to file
- Similar structure to train_dqn.py for easy comparison
"""
from snake import SnakeGame
from ai_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def plot_training_results(history, save_path='qlearning_training_results.png'):
    """
    Plot Q-Learning training metrics

    Creates a 2x2 grid of plots:
    1. Score per episode (with moving average)
    2. Total reward per episode
    3. Epsilon decay
    4. Q-table growth
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning Snake Training Results', fontsize=16, fontweight='bold')

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

    # --- Plot 4: Q-Table Growth ---
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


def plot_live_update(history, save_path='qlearning_training_progress.png'):
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

    Q-Learning typically needs fewer episodes than DQN since:
    - Simpler algorithm (table-based)
    - Faster convergence for small state spaces
    - No neural network training overhead

    Expected performance:
    Episodes 1-200: Random exploration (0-20 score)
    Episodes 200-500: Learning basics (20-50 score)
    Episodes 500-1000: Improving (50-100 score)
    Episodes 1000-2000: Refined strategy (80-150 score)
    """
    print("=" * 60)
    print("Q-LEARNING SNAKE TRAINING")
    print("=" * 60)

    # Create Q-Learning agent
    agent = QLearningAgent(
        alpha=0.1,  # Learning rate: how much to update Q-values
        gamma=0.95,  # Discount factor: how much to value future rewards
        epsilon=1.0,  # Start fully random
        epsilon_decay=0.995,  # Decay per episode
        epsilon_min=0.01  # Minimum exploration
    )

    # Try to load existing model
    if agent.load_model('q_learning_model.pkl'):
        print("Loaded existing model - continuing training")
    else:
        print("Starting fresh training")

    # Create game in training mode (headless for speed)
    game = SnakeGame(agent, training_mode=True, fps=0)

    # Statistics tracking (for plotting)
    history = {
        'scores': [],
        'rewards': [],
        'epsilons': [],
        'qtable_sizes': [],
        'steps': []
    }

    # Rolling window for console output
    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    best_score = 0
    best_avg_score = 0

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Save interval: {save_interval}, Print interval: {print_interval}")
    print(f"Initial Q-table size: {len(agent.q_table)}")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        # Play one episode
        total_reward, score, steps = game.play_episode()

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Get current stats
        stats = agent.get_stats()

        # Track full history (for plotting)
        history['scores'].append(score)
        history['rewards'].append(total_reward)
        history['epsilons'].append(stats['epsilon'])
        history['qtable_sizes'].append(stats['q_table_size'])
        history['steps'].append(steps)

        # Track rolling stats (for console)
        recent_scores.append(score)
        recent_rewards.append(total_reward)

        # Update best scores
        if score > best_score:
            best_score = score
            agent.save_model('q_learning_model_best.pkl')

        avg_score = np.mean(recent_scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        # Print progress
        if episode % print_interval == 0:
            avg_reward = np.mean(recent_rewards)

            print(f"Episode {episode:5d} | "
                  f"Score: {score:3d} | "
                  f"Avg(100): {avg_score:6.1f} | "
                  f"Best: {best_score:3d} | "
                  f"ε: {stats['epsilon']:.3f} | "
                  f"Q-table: {stats['q_table_size']:4d} states")

        # Save model checkpoint
        if episode % save_interval == 0:
            agent.save_model('q_learning_model.pkl')
            print(f">>> Checkpoint saved at episode {episode}")

        # Update plots
        if episode % plot_interval == 0:
            plot_live_update(history, 'qlearning_training_progress.png')
            print(f">>> Progress plot updated")

    # Final save
    agent.save_model('q_learning_model.pkl')

    # Generate final comprehensive plot
    plot_training_results(history, 'qlearning_training_results.png')

    # Save training history to file
    np.savez('qlearning_training_history.npz',
             scores=history['scores'],
             rewards=history['rewards'],
             epsilons=history['epsilons'],
             qtable_sizes=history['qtable_sizes'],
             steps=history['steps'])
    print("Training history saved to qlearning_training_history.npz")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Score: {best_score}")
    print(f"Best Average Score: {best_avg_score:.1f}")
    print(f"Final Average (last 100): {np.mean(recent_scores):.1f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Final Q-Table Size: {len(agent.q_table)} states")
    print(f"Models saved: q_learning_model.pkl, q_learning_model_best.pkl")
    print(f"Plots saved: qlearning_training_results.png, qlearning_training_progress.png")
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
        num_episodes = 500
    elif choice == '2':
        num_episodes = 1000
    elif choice == '3':
        num_episodes = 2000
    elif choice == '4':
        num_episodes = 5000
    elif choice == '5':
        num_episodes = int(input("Enter number of episodes: "))
    else:
        print("Invalid choice, using default (1000 episodes)")
        num_episodes = 1000

    # Train with chosen parameters
    agent, history = train(
        num_episodes=num_episodes,
        save_interval=500,
        print_interval=50,
        plot_interval=100
    )

    return agent, history


if __name__ == '__main__':
    main()