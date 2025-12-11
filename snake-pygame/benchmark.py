# benchmark.py - Unified benchmark for Q-Learning and DQN
"""
Simple benchmark script that tests agents and outputs raw data
User chooses agent and number of games
"""
from snake import SnakeGame
from ai_agent import QLearningAgent
import numpy as np
import json

# Try to import DQN
try:
    from dqn_agent import DQNAgent

    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False


def run_benchmark(agent, num_games, agent_name):
    """
    Run benchmark on an agent and return raw data

    Args:
        agent: Agent to test
        num_games: Number of games to play
        agent_name: Name for display

    Returns:
        List of scores, List of steps, Agent name
    """
    print(f"\nBenchmarking {agent_name}...")
    print(f"Running {num_games} games...\n")

    game = SnakeGame(agent, training_mode=True, fps=0)

    scores = []
    steps_list = []

    for i in range(num_games):
        total_reward, score, steps = game.play_episode()
        scores.append(score)
        steps_list.append(steps)

        # Show progress every 10 games
        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            print(f"Progress: {i + 1}/{num_games} games completed")

    return scores, steps_list, agent_name


def print_raw_data(scores, steps_list, agent_name):
    """Print raw benchmark data"""
    print("\n" + "=" * 60)
    print(f"RAW BENCHMARK DATA - {agent_name}")
    print("=" * 60)

    print(f"\nAgent: {agent_name}")
    print(f"Games Played: {len(scores)}")

    print("\n--- SCORES ---")
    print(f"All Scores: {scores}")
    print(f"\nMean: {np.mean(scores):.2f}")
    print(f"Median: {np.median(scores):.2f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Min: {np.min(scores)}")
    print(f"Max: {np.max(scores)}")
    print(f"Q1 (25th percentile): {np.percentile(scores, 25):.2f}")
    print(f"Q3 (75th percentile): {np.percentile(scores, 75):.2f}")

    print("\n--- SURVIVAL (STEPS) ---")
    print(f"All Steps: {steps_list}")
    print(f"\nMean: {np.mean(steps_list):.2f}")
    print(f"Median: {np.median(steps_list):.2f}")
    print(f"Std Dev: {np.std(steps_list):.2f}")
    print(f"Min: {np.min(steps_list)}")
    print(f"Max: {np.max(steps_list)}")

    print("\n--- DERIVED METRICS ---")
    foods_eaten = [s // 10 for s in scores]
    print(f"Mean Foods Eaten: {np.mean(foods_eaten):.2f}")
    survival_rate = np.sum(np.array(scores) > 0) / len(scores) * 100
    print(f"Survival Rate (Score > 0): {survival_rate:.1f}%")
    high_score_rate = np.sum(np.array(scores) >= 100) / len(scores) * 100
    print(f"High Score Rate (≥100): {high_score_rate:.1f}%")
    very_high_score_rate = np.sum(np.array(scores) >= 200) / len(scores) * 100
    print(f"Very High Score Rate (≥200): {very_high_score_rate:.1f}%")

    print("\n" + "=" * 60)


def save_to_json(scores, steps_list, agent_name, filename):
    """Save results to JSON file"""

    def convert_to_native(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    foods_eaten = [s // 10 for s in scores]

    data = {
        'agent_name': agent_name,
        'num_games': len(scores),
        'raw_data': {
            'scores': convert_to_native(scores),
            'steps': convert_to_native(steps_list),
            'foods_eaten': convert_to_native(foods_eaten)
        },
        'statistics': {
            'scores': {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': int(np.min(scores)),
                'max': int(np.max(scores)),
                'q1': float(np.percentile(scores, 25)),
                'q3': float(np.percentile(scores, 75))
            },
            'steps': {
                'mean': float(np.mean(steps_list)),
                'median': float(np.median(steps_list)),
                'std': float(np.std(steps_list)),
                'min': int(np.min(steps_list)),
                'max': int(np.max(steps_list))
            },
            'foods': {
                'mean': float(np.mean(foods_eaten))
            },
            'rates': {
                'survival_rate': float(np.sum(np.array(scores) > 0) / len(scores) * 100),
                'high_score_rate': float(np.sum(np.array(scores) >= 100) / len(scores) * 100),
                'very_high_score_rate': float(np.sum(np.array(scores) >= 200) / len(scores) * 100)
            }
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Data saved to {filename}")


def main():
    """Main benchmark function"""
    print("\n" + "=" * 60)
    print("SNAKE REINFORCEMENT LEARNING - BENCHMARK")
    print("=" * 60)

    # Choose agent
    print("\nWhich agent to benchmark?")
    print("1. Q-Learning")
    if DQN_AVAILABLE:
        print("2. DQN")

    agent_choice = input("\nChoice (1-2): " if DQN_AVAILABLE else "Choice (1): ")

    # Choose number of games
    print("\nHow many games?")
    print("Recommended: 50-100 games for good statistics")
    num_games = int(input("Number of games: "))

    # Load and test agent
    if agent_choice == '1':
        # Q-Learning
        agent = QLearningAgent(epsilon=0.0)
        if not agent.load_model('q_learning_model.pkl'):
            print("\n❌ ERROR: No trained Q-Learning model found!")
            print("Train first: python train.py")
            return
        agent_name = "Q-Learning"
        filename = "benchmark_qlearning.json"

    elif agent_choice == '2' and DQN_AVAILABLE:
        # DQN
        agent = DQNAgent(epsilon=0.0)
        if not agent.load_model('dqn_model.pth'):
            print("\n❌ ERROR: No trained DQN model found!")
            print("Train first: python train_dqn.py")
            return
        agent_name = "DQN"
        filename = "benchmark_dqn.json"

    else:
        print("Invalid choice!")
        return

    # Run benchmark
    scores, steps_list, agent_name = run_benchmark(agent, num_games, agent_name)

    # Output results
    print_raw_data(scores, steps_list, agent_name)
    save_to_json(scores, steps_list, agent_name, filename)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\n✓ Tested: {agent_name}")
    print(f"✓ Games: {num_games}")
    print(f"✓ Data saved: {filename}")
    print("\nUse this data for your calculations and presentation!")


if __name__ == '__main__':
    main()