"""Train Baseline / Random Forest / XGBoost and save the best model (Phases 8–11).

Usage:
    python -m scripts.train
"""

from src.config import ensure_directories, setup_logging
from src.models.train import run_training


def main() -> None:
    setup_logging()
    ensure_directories()
    run_training()


if __name__ == "__main__":
    main()
