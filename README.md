# Snake Reinforcement Learning

This project compares two reinforcement learning approaches for the Snake game:

- Q-Learning
- Deep Q-Network (DQN)

The game logic and learning logic were kept separate from saved models, plots, and benchmark outputs so the project is easier to read and submit.

## Project Structure

```text
Snake-ReinforcementLearning/
├── artifacts/
│   ├── data/          # Training history and benchmark JSON files
│   ├── models/        # Saved Q-learning and DQN models
│   └── plots/         # Training graphs and progress plots
├── snake_rl/
│   ├── agents/        # Q-learning, DQN, and replay buffer code
│   ├── evaluation/    # Benchmark script logic
│   ├── game/          # Snake game environment
│   ├── training/      # Training loops for Q-learning and DQN
│   └── paths.py       # Centralized artifact paths
├── benchmark.py       # Run benchmark from repo root
├── play.py            # Watch trained agents play
├── snake.py           # Compatibility launcher for the game menu
├── train.py           # Compatibility launcher for Q-learning training
├── train_dqn.py       # Train DQN
├── train_qlearning.py # Train Q-learning
└── requirements.txt
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Watch trained agents:

```bash
python play.py
```

You can also use the older launcher name:

```bash
python snake.py
```

Train Q-learning:

```bash
python train_qlearning.py
```

You can also use:

```bash
python train.py
```

Train DQN:

```bash
python train_dqn.py
```

Run benchmark:

```bash
python benchmark.py
```

## Notes

- `RandomAgent` was removed so the project focuses only on Q-Learning and DQN.
- Saved outputs now go into `artifacts/` instead of being mixed with source code.
- The DQN and Q-learning algorithms were preserved; the cleanup here is mainly organization, imports, and file paths.
