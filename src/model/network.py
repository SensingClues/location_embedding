import tensorflow as tf


def get_model():
    """Create model."""
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Conv2D(64, (1, 1), activation='relu')) # sentiance paper has two of these layers.
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='linear'))

    return model
