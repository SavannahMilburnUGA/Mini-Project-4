import numpy as np
import csv

def load_data_wrapper(train_path="sign_mnist_train.csv", test_path="sign_mnist_test.csv"):
    def load_csv(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            data = [row for row in reader]
        data = np.array(data, dtype=np.float32)
        labels = data[:, 0].astype(np.int32)
        images = data[:, 1:] / 255.0  # normalize pixels
        return images, labels

    train_images, train_labels = load_csv(train_path)
    test_images, test_labels = load_csv(test_path)

    # Wrap like Nielsen's mnist_loader: (image, label) tuples
    training_data = list(zip([x.reshape(784, 1) for x in train_images],
                             [vectorized_result(y) for y in train_labels]))
    test_data = list(zip([x.reshape(784, 1) for x in test_images],
                         test_labels))

    return training_data, test_data

def vectorized_result(j):
    e = np.zeros((24, 1))
    if j >= 9:  # skip 'J' (label 9) if using 24 letters
        j -= 1
    e[j] = 1.0
    return e