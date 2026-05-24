from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
DATA_DIR = ARTIFACTS_DIR / "data"

Q_LEARNING_MODEL_PATH = MODELS_DIR / "q_learning_model.pkl"
Q_LEARNING_BEST_MODEL_PATH = MODELS_DIR / "q_learning_model_best.pkl"
DQN_MODEL_PATH = MODELS_DIR / "dqn_model_improved.pth"
DQN_BEST_MODEL_PATH = MODELS_DIR / "dqn_model_best.pth"

Q_LEARNING_PROGRESS_PLOT_PATH = PLOTS_DIR / "qlearning_training_progress.png"
Q_LEARNING_RESULTS_PLOT_PATH = PLOTS_DIR / "qlearning_training_results.png"
DQN_PROGRESS_PLOT_PATH = PLOTS_DIR / "training_progress.png"
DQN_RESULTS_PLOT_PATH = PLOTS_DIR / "training_results.png"

Q_LEARNING_HISTORY_PATH = DATA_DIR / "qlearning_training_history.npz"
DQN_HISTORY_PATH = DATA_DIR / "training_history.npz"
BENCHMARK_QLEARNING_PATH = DATA_DIR / "benchmark_qlearning.json"
BENCHMARK_DQN_PATH = DATA_DIR / "benchmark_dqn.json"


def ensure_artifact_directories():
    """Create artifact folders if they do not already exist."""
    for directory in (MODELS_DIR, PLOTS_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
