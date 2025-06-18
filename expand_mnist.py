"""
expand_mnist.py
~~~~~~~~~~~~~~~

Expands the 50,000 MNIST training images to 250,000 by shifting each
image up, down, left, and right by one pixel. Saves result as a .npz file.

"""

import os
import random
import numpy as np
from tensorflow.keras.datasets import mnist

SAVE_PATH = "../data/mnist_expanded.npz"

print("Expanding the MNIST training set")

if os.path.exists(SAVE_PATH):
    print("The expanded training set already exists. Exiting.")
else:
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # Use only first 50k for training, next 10k for validation
    X_train, y_train = X_train_full[:50000], y_train_full[:50000]
    X_val, y_val = X_train_full[50000:], y_train_full[50000:]

    expanded_images = []
    expanded_labels = []

    for j, (x, y) in enumerate(zip(X_train, y_train)):
        expanded_images.append(x)
        expanded_labels.append(y)
        if j % 1000 == 0:
            print(f"Expanding image number {j}")

        for d, axis, index_position, index in [
                (1,  0, "first", 0),   # down
                (-1, 0, "first", 27), # up
                (1,  1, "last",  0),  # right
                (-1, 1, "last",  27)  # left
            ]:
            new_img = np.roll(x, d, axis)
            if index_position == "first":
                if axis == 0:
                    new_img[index, :] = 0
                else:
                    new_img[:, index] = 0
            else:
                if axis == 0:
                    new_img[index, :] = 0
                else:
                    new_img[:, index] = 0
            expanded_images.append(new_img)
            expanded_labels.append(y)

    # Shuffle dataset
    combined = list(zip(expanded_images, expanded_labels))
    random.shuffle(combined)
    expanded_images, expanded_labels = zip(*combined)
    expanded_images = np.array(expanded_images).reshape(-1, 784) / 255.0
    expanded_labels = np.array(expanded_labels)

    # Also normalize validation/test sets
    X_val = X_val.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    print("Saving expanded dataset to", SAVE_PATH)
    np.savez_compressed(SAVE_PATH,
                        train_images=expanded_images,
                        train_labels=expanded_labels,
                        val_images=X_val,
                        val_labels=y_val,
                        test_images=X_test,
                        test_labels=y_test)