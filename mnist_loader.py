"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data without using cPickle. Uses
TensorFlow's built-in dataset loader. Returns data in the format
expected by our neural network implementations.
"""

import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    """Return the MNIST data as a tuple (training_data, validation_data, test_data).

    - training_data: (X_train, y_train) -- first 50,000 samples of MNIST training set
    - validation_data: (X_val, y_val) -- remaining 10,000 samples of MNIST training set
    - test_data: (X_test, y_test) -- 10,000 MNIST test images
    """
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # Normalize and reshape
    X_train_full = X_train_full.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    # Split 50k train / 10k val
    X_train, X_val = X_train_full[:50000], X_train_full[50000:]
    y_train, y_val = y_train_full[:50000], y_train_full[50000:]

    training_data = (X_train, y_train)
    validation_data = (X_val, y_val)
    test_data = (X_test, y_test)

    return training_data, validation_data, test_data


def load_data_wrapper():
    """Returns training_data, validation_data, test_data in a format ready
    for neural network training with one-hot labels for training data.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [x.reshape(784, 1) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [x.reshape(784, 1) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [x.reshape(784, 1) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 10-dimensional one-hot vector for digit j (0 through 9)."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e