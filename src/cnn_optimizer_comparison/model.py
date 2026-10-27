```python
"""CNN architecture for MNIST image classification."""

import tensorflow as tf


def build_cnn(seed: int = 42) -> tf.keras.Model:
    """Build a CNN for classifying MNIST digits from 0 to 9."""

    inputs = tf.keras.layers.Input(shape=(28, 28, 1))

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        name="conv1",
    )(inputs)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool1",
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        name="conv2",
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="pool2",
    )(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)

    x = tf.keras.layers.Dense(
        units=128,
        activation="relu",
        name="dense1",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=0.3,
        seed=seed,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=10,
        activation="softmax",
        name="output",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mnist_cnn",
    )