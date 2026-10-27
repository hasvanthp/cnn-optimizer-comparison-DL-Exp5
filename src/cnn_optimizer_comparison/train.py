"""Train CNN models using seven optimization algorithms."""

import json
import time
from pathlib import Path

import tensorflow as tf

from cnn_optimizer_comparison.data import load_mnist
from cnn_optimizer_comparison.model import build_cnn
from cnn_optimizer_comparison.optimizers import get_optimizers


BATCH_SIZE = 128
EPOCHS = 10
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
HISTORIES_DIR = RESULTS_DIR / "histories"


def train_all(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train one fresh CNN for each optimizer and save its history."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_val, y_val), _ = load_mnist()

    results = {}
    optimizers = get_optimizers()

    for name in optimizers:
        print(f"\n{'=' * 60}")
        print(f"Training optimizer: {name}")
        print(f"{'=' * 60}")

        # Reset model initialization for a fairer comparison.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = build_cnn(seed=SEED)
        optimizer = optimizers[name]

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        start_time = time.perf_counter()

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
        )

        elapsed = time.perf_counter() - start_time

        history_data = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }

        history_path = HISTORIES_DIR / (
            f"{name.lower().replace(' ', '_').replace('+', 'plus')}.json"
        )

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history_data, file, indent=2)

        results[name] = {
            "training_time_seconds": round(elapsed, 2),
            "final_train_accuracy": history_data["accuracy"][-1],
            "final_validation_accuracy": history_data["val_accuracy"][-1],
            "final_train_loss": history_data["loss"][-1],
            "final_validation_loss": history_data["val_loss"][-1],
            "history_path": str(history_path),
        }

        print(f"\nOptimizer: {name}")
        print(f"Training time: {elapsed:.2f} seconds")
        print(
            "Final validation accuracy: "
            f"{history_data['val_accuracy'][-1]:.4f}"
        )

        # Release the model before training the next optimizer.
        del model
        tf.keras.backend.clear_session()

    summary_path = RESULTS_DIR / "training_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nTraining summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    train_all()