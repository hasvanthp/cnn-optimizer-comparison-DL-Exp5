"""Optimizer configurations for the CNN comparison experiment."""

import tensorflow as tf


def get_optimizers() -> dict[str, tf.keras.optimizers.Optimizer]:
    """Return fresh instances of the seven optimizers."""

    return {
        "SGD": tf.keras.optimizers.SGD(
            learning_rate=0.01,
        ),
        "SGD + Momentum": tf.keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9,
        ),
        "Adagrad": tf.keras.optimizers.Adagrad(
            learning_rate=0.01,
        ),
        "RMSprop": tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
        ),
        "Adam": tf.keras.optimizers.Adam(
            learning_rate=0.001,
        ),
        "AdamW": tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001,
        ),
        "Nadam": tf.keras.optimizers.Nadam(
            learning_rate=0.001,
        ),
    }