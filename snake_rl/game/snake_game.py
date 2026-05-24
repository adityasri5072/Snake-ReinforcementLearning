import random
import sys

import pygame

from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.paths import DQN_BEST_MODEL_PATH, Q_LEARNING_MODEL_PATH

try:
    from snake_rl.agents.dqn_agent import DQNAgent

    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    print("Note: DQN not available (install PyTorch: pip install torch)")


pygame.init()

WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLACK = pygame.Color(0, 0, 0)
WINDOW_X = 720
WINDOW_Y = 480


class SnakeGame:
    def __init__(self, agent, training_mode=False, fps=10):
        """
        Snake Game with RL Agent

        Args:
            agent: AI agent
            training_mode: If True, runs headless for faster training
            fps: Frames per second (higher = faster game)
        """
        self.agent = agent
        self.training_mode = training_mode
        self.fps = fps

        if not training_mode:
            pygame.display.set_caption('Snake AI')
            self.game_window = pygame.display.set_mode((WINDOW_X, WINDOW_Y))
        else:
            self.game_window = None

        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        """Reset game state for new episode"""
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = self._spawn_food()
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.steps = 0
        self.last_distance_to_food = self._get_distance_to_food()

        agent_class_name = type(self.agent).__name__
        if agent_class_name in ['QLearningAgent', 'DQNAgent']:
            self.agent.reset_episode()

    def _spawn_food(self):
        """Spawn food at random position not on snake"""
        while True:
            food_pos = [
                random.randrange(1, (WINDOW_X // 10)) * 10,
                random.randrange(1, (WINDOW_Y // 10)) * 10
            ]
            if food_pos not in self.snake_body:
                return food_pos

    def _get_distance_to_food(self):
        """Calculate Manhattan distance to food"""
        return abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])

    def _get_game_state(self):
        """Get current game state for agent"""
        return {
            'snake_pos': self.snake_pos.copy(),
            'snake_body': [segment.copy() for segment in self.snake_body],
            'food_pos': self.food_pos.copy(),
            'direction': self.direction,
            'score': self.score
        }

    def _calculate_reward(self, food_eaten, game_over):
        """
        Calculate reward for the agent
        """
        if game_over:
            return -100

        if food_eaten:
            return 50

        current_distance = self._get_distance_to_food()
        if current_distance < self.last_distance_to_food:
            reward = 1
        else:
            reward = -1.5

        reward -= 0.1
        self.last_distance_to_food = current_distance
        return reward

    def step(self):
        """
        Execute one game step

        Returns:
            reward: Reward for this step
            game_over: Boolean indicating if game ended
            score: Current score
        """
        self.steps += 1

        game_state = self._get_game_state()
        action = self.agent.get_action(game_state)
        change_to = action

        if change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10

        food_eaten = False
        self.snake_body.insert(0, list(self.snake_pos))
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 10
            food_eaten = True
            self.food_pos = self._spawn_food()
        else:
            self.snake_body.pop()

        game_over = False
        if (self.snake_pos[0] < 0 or self.snake_pos[0] >= WINDOW_X or
                self.snake_pos[1] < 0 or self.snake_pos[1] >= WINDOW_Y):
            game_over = True

        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                game_over = True

        if self.steps > 1000:
            game_over = True

        reward = self._calculate_reward(food_eaten, game_over)

        agent_class_name = type(self.agent).__name__
        if agent_class_name in ['QLearningAgent', 'DQNAgent']:
            new_game_state = self._get_game_state()
            self.agent.learn(reward, new_game_state, game_over)

        if not self.training_mode and self.game_window is not None:
            self._render()

        return reward, game_over, self.score

    def _render(self):
        """Render the game (only in non-training mode)"""
        self.game_window.fill(BLACK)

        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))

        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

        self._show_score()

        agent_class_name = type(self.agent).__name__
        if agent_class_name in ['QLearningAgent', 'DQNAgent']:
            self._show_agent_stats()

        pygame.display.update()
        self.clock.tick(self.fps)

    def _show_score(self):
        """Display score on screen"""
        font = pygame.font.SysFont('consolas', 20)
        score_surface = font.render(f'Score: {self.score}', True, WHITE)
        score_rect = score_surface.get_rect()
        score_rect.topleft = (10, 10)
        self.game_window.blit(score_surface, score_rect)

    def _show_agent_stats(self):
        """Display agent statistics"""
        stats = self.agent.get_stats()
        font = pygame.font.SysFont('consolas', 16)

        epsilon_text = f"Epsilon: {stats['epsilon']:.3f}"
        epsilon_surface = font.render(epsilon_text, True, WHITE)
        self.game_window.blit(epsilon_surface, (10, 40))

        if 'q_table_size' in stats:
            qtable_text = f"Q-table: {stats['q_table_size']}"
            qtable_surface = font.render(qtable_text, True, WHITE)
            self.game_window.blit(qtable_surface, (10, 60))
        elif 'buffer_size' in stats:
            buffer_text = f"Buffer: {stats['buffer_size']}"
            buffer_surface = font.render(buffer_text, True, WHITE)
            self.game_window.blit(buffer_surface, (10, 60))

    def _show_game_over(self):
        """Display game over screen"""
        font = pygame.font.SysFont('times new roman', 50)
        game_over_surface = font.render(f'Score: {self.score}', True, RED)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (WINDOW_X / 2, WINDOW_Y / 4)
        self.game_window.blit(game_over_surface, game_over_rect)
        pygame.display.flip()
        pygame.time.wait(2000)

    def play_episode(self):
        """
        Play one complete episode

        Returns:
            total_reward: Total reward for episode
            score: Final score
            steps: Number of steps taken
        """
        self.reset_game()
        total_reward = 0
        game_over = False

        while not game_over:
            if not self.training_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            reward, game_over, score = self.step()
            total_reward += reward

        if not self.training_mode and self.game_window is not None:
            self._show_game_over()

        return total_reward, self.score, self.steps


def main():
    """Main function for playing or visualizing the game"""
    print("Snake AI Game")
    print("1. Watch Trained Q-Learning Agent")
    print("2. Train Q-Learning Agent")
    if DQN_AVAILABLE:
        print("3. Watch Trained DQN Agent")
        print("4. Train DQN Agent")

    choice = input("Enter choice (1-4): " if DQN_AVAILABLE else "Enter choice (1-2): ")

    if choice == '1':
        agent = QLearningAgent(epsilon=0.0)
        if agent.load_model(Q_LEARNING_MODEL_PATH):
            print("Loaded trained Q-Learning model")
            game = SnakeGame(agent, training_mode=False, fps=15)
            print("Watching trained Q-Learning agent play...")
            print("Close window to exit")

            while True:
                total_reward, score, steps = game.play_episode()
                print(f"Episode finished - Score: {score}, Steps: {steps}")
        else:
            print("No trained model found. Train first using: python train_qlearning.py")

    elif choice == '2':
        print("Please use: python train_qlearning.py")
        print("Then come back here and choose option 1 to watch the trained agent.")

    elif choice == '3' and DQN_AVAILABLE:
        agent = DQNAgent(epsilon=0.0)
        if agent.load_model(DQN_BEST_MODEL_PATH):
            print("Loaded trained DQN model")
            game = SnakeGame(agent, training_mode=False, fps=15)
            print("Watching trained DQN agent play...")
            print("Close window to exit")

            while True:
                total_reward, score, steps = game.play_episode()
                print(f"Episode finished - Score: {score}, Steps: {steps}")
        else:
            print("No trained model found. Train first using: python train_dqn.py")

    elif choice == '4' and DQN_AVAILABLE:
        print("Please use: python train_dqn.py")
        print("Then come back here and choose option 3 to watch the trained agent.")

    else:
        print("Invalid choice")
