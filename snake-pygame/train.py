# train.py - Training script
from snake import SnakeGame
from ai_agent import QLearningAgent


def train_agent(num_episodes=1000):
    """
    Train Q-Learning agent headless (no graphics = fast!)

    Args:
        num_episodes: Number of games to train on
    """
    print(f"Training Q-Learning agent for {num_episodes} episodes...")
    print("(Running headless - no graphics for speed)")
    print("-" * 50)

    # Create agent
    agent = QLearningAgent()

    # Try to load existing model to continue training
    agent.load_model('q_learning_model.pkl')

    # Create game in training mode (headless)
    game = SnakeGame(agent, training_mode=True, fps=0)

    # Training loop
    for episode in range(1, num_episodes + 1):
        # Play one episode
        total_reward, score, steps = game.play_episode()

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Print progress
        if episode % 50 == 0:
            print(f"Episode {episode}/{num_episodes} - Score: {score}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}, Q-table: {len(agent.q_table)}")

        # Save periodically
        if episode % 500 == 0:
            agent.save_model('q_learning_model.pkl')
            print(f"✓ Model saved at episode {episode}")

    # Final save
    agent.save_model('q_learning_model.pkl')
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Final Q-table size: {len(agent.q_table)} states")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print("Model saved to: q_learning_model.pkl")
    print("\nNow run 'python snake.py' and choose option 2 to watch your trained agent!")
    print("=" * 50)


if __name__ == '__main__':
    print("\n=== Q-Learning Training ===")
    print("1. Quick train (500 episodes) ")
    print("2. Medium train (1000 episodes) ")
    print("3. Full train (2000 episodes) ")
    print("4. Custom")

    choice = input("\nChoose (1-4): ")

    if choice == '1':
        train_agent(500)
    elif choice == '2':
        train_agent(1000)
    elif choice == '3':
        train_agent(2000)
    elif choice == '4':
        episodes = int(input("Number of episodes: "))
        train_agent(episodes)
    else:
        print("Invalid choice, running default (1000 episodes)")
        train_agent(1000)