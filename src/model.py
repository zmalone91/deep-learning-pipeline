# src/model.py

import tensorflow as tf
from scikeras.wrappers import KerasClassifier

def create_model(num_neurons=8, learning_rate=1e-3):
    """
    A simple feed-forward neural network for classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_neurons, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_base_classifier(epochs=10, batch_size=8, verbose=0):
    """
    Returns a scikeras KerasClassifier with default arguments.
    """
    return KerasClassifier(
        model=create_model,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
