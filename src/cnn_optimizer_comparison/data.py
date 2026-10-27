```python
"""Dataset loading and preprocessing for MNIST classification."""

import numpy as np
import tensorflow as tf


SEED = 42
VALIDATION_SIZE = 6_000


def load_mnist():
    """Load MNIST and return normalized train, validation, and test data."""

    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values to [0, 1].
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Add the grayscale channel dimension.
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Shuffle deterministically before creating the validation split.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x_train_full))

    val_indices = indices[:VALIDATION_SIZE]
    train_indices = indices[VALIDATION_SIZE:]

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]

    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]

    return (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
    )
```
