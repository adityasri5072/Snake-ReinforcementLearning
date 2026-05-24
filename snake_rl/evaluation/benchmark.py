import json

import numpy as np

from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.game.snake_game import SnakeGame
from snake_rl.paths import (
    BENCHMARK_DQN_PATH,
    BENCHMARK_QLEARNING_PATH,
    DQN_BEST_MODEL_PATH,
    Q_LEARNING_MODEL_PATH,
)

try:
    from snake_rl.agents.dqn_agent import DQNAgent

    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False


def run_benchmark(agent, num_games, agent_name):
    """
    Run benchmark on an agent and return raw data
    """
    print(f"\nBenchmarking {agent_name}...")
    print(f"Running {num_games} games...\n")

    game = SnakeGame(agent, training_mode=True, fps=0)

    scores = []
    steps_list = []

    for index in range(num_games):
        total_reward, score, steps = game.play_episode()
        scores.append(score)
        steps_list.append(steps)

        if (index + 1) % 10 == 0 or (index + 1) == num_games:
            print(f"Progress: {index + 1}/{num_games} games completed")

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
    foods_eaten = [score // 10 for score in scores]
    print(f"Mean Foods Eaten: {np.mean(foods_eaten):.2f}")
    survival_rate = np.sum(np.array(scores) > 0) / len(scores) * 100
    print(f"Survival Rate (Score > 0): {survival_rate:.1f}%")
    high_score_rate = np.sum(np.array(scores) >= 100) / len(scores) * 100
    print(f"High Score Rate (>=100): {high_score_rate:.1f}%")
    very_high_score_rate = np.sum(np.array(scores) >= 200) / len(scores) * 100
    print(f"Very High Score Rate (>=200): {very_high_score_rate:.1f}%")

    print("\n" + "=" * 60)


def save_to_json(scores, steps_list, agent_name, filename):
    """Save results to JSON file"""

    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    foods_eaten = [score // 10 for score in scores]

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

    with open(filename, 'w', encoding='utf-8') as file_handle:
        json.dump(data, file_handle, indent=2)

    print(f"\nSaved data to {filename}")


def main():
    """Main benchmark function"""
    print("\n" + "=" * 60)
    print("SNAKE REINFORCEMENT LEARNING - BENCHMARK")
    print("=" * 60)

    print("\nWhich agent to benchmark?")
    print("1. Q-Learning")
    if DQN_AVAILABLE:
        print("2. DQN")

    agent_choice = input("\nChoice (1-2): " if DQN_AVAILABLE else "Choice (1): ")

    print("\nHow many games?")
    print("Recommended: 50-100 games for good statistics")
    num_games = int(input("Number of games: "))

    if agent_choice == '1':
        agent = QLearningAgent(epsilon=0.0)
        if not agent.load_model(Q_LEARNING_MODEL_PATH):
            print("\nERROR: No trained Q-Learning model found!")
            print("Train first: python train_qlearning.py")
            return
        agent_name = "Q-Learning"
        filename = BENCHMARK_QLEARNING_PATH

    elif agent_choice == '2' and DQN_AVAILABLE:
        agent = DQNAgent(epsilon=0.0)
        if not agent.load_model(DQN_BEST_MODEL_PATH):
            print("\nERROR: No trained DQN model found!")
            print("Train first: python train_dqn.py")
            return
        agent_name = "DQN"
        filename = BENCHMARK_DQN_PATH

    else:
        print("Invalid choice!")
        return

    scores, steps_list, agent_name = run_benchmark(agent, num_games, agent_name)
    print_raw_data(scores, steps_list, agent_name)
    save_to_json(scores, steps_list, agent_name, filename)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\nTested: {agent_name}")
    print(f"Games: {num_games}")
    print(f"Data saved: {filename}")
