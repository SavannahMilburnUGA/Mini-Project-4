import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils, datasets, regularizers
#from tensorflow.keras.callbacks import History
import os

# Load MNIST data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Split validation data
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Set batch size and epochs
BATCH_SIZE = 10
EPOCHS = 60


def build_model(activation='sigmoid', dropout=False, l2_lambda=0.0):
    model = models.Sequential()
    model.add(layers.Conv2D(20, (5, 5), activation=activation, input_shape=(28, 28, 1),
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(40, (5, 5), activation=activation,
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation=activation,
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def train_and_evaluate(activation='sigmoid', dropout=False, l2_lambda=0.0):
    print(f"Training model with activation={activation}, dropout={dropout}, l2_lambda={l2_lambda}")
    model = build_model(activation, dropout, l2_lambda)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_val, y_val),
                        verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.2%}")
    return model, history


if __name__ == '__main__':
    # Double conv net with sigmoid
    train_and_evaluate(activation='sigmoid', dropout=False)

    # Double conv net with relu
    train_and_evaluate(activation='relu', dropout=False)

    # Double conv with relu + dropout
    train_and_evaluate(activation='relu', dropout=True)

    # Regularized double conv with relu
    for lmbda in [0.0001, 0.001, 0.01, 0.1]:
        train_and_evaluate(activation='relu', dropout=False, l2_lambda=lmbda)