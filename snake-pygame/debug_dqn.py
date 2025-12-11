# debug_dqn.py - Debug DQN training
"""
This script adds debug prints to figure out why replay buffer stays at 0
"""
from snake import SnakeGame
from dqn_agent import DQNAgent

print("=" * 60)
print("DQN DEBUG SCRIPT")
print("=" * 60)

# Create agent
agent = DQNAgent(
    state_size=11,
    action_size=4,
    learning_rate=0.001,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.01,
    batch_size=64,
    buffer_size=10000,
    target_update=1000
)

# Create game
game = SnakeGame(agent, training_mode=True, fps=0)

print(f"\nInitial buffer size: {len(agent.memory)}")
print(f"Agent type: {type(agent).__name__}")
print(f"Has learn method: {hasattr(agent, 'learn')}")

# Play 5 episodes and check buffer growth
for episode in range(1, 6):
    print(f"\n--- Episode {episode} ---")
    print(f"Buffer size before episode: {len(agent.memory)}")
    print(f"Agent last_state before: {agent.last_state}")

    total_reward, score, steps = game.play_episode()

    print(f"Episode finished:")
    print(f"  Steps: {steps}")
    print(f"  Score: {score}")
    print(f"  Buffer size after: {len(agent.memory)}")
    print(f"  Expected buffer additions: {steps - 1}")  # -1 for first step where last_state is None

    if len(agent.memory) == 0:
        print("  ⚠️  WARNING: Buffer is still empty!")
        print("  This means learn() is not storing experiences")
        break
    else:
        print(f"  ✓  Buffer growing! Added {len(agent.memory)} experiences")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)

# Additional checks
print("\nChecking agent methods:")
print(f"  get_action: {hasattr(agent, 'get_action')}")
print(f"  learn: {hasattr(agent, 'learn')}")
print(f"  reset_episode: {hasattr(agent, 'reset_episode')}")
print(f"  memory (replay buffer): {hasattr(agent, 'memory')}")
print(f"  memory.push: {hasattr(agent.memory, 'push')}")
print(f"  memory.__len__: {hasattr(agent.memory, '__len__')}")