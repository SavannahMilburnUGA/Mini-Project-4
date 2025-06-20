import os
import pandas as pd
import numpy as np

# Create mapping from original labels to consecutive indices (skip 9)
def map_label(original_label):
    if original_label < 9:
        return original_label
    else:
        return original_label - 1

# Load data from local CSVs
def load_data(train_csv_path='sign_mnist_train.csv', test_csv_path='sign_mnist_test.csv'):
    # Load training data
    train_df = pd.read_csv(train_csv_path)
    train_labels = np.array([map_label(label) for label in train_df['label'].values])
    train_images = train_df.drop('label', axis=1).values / 255.0

    # Load test data
    test_df = pd.read_csv(test_csv_path)
    test_labels = np.array([map_label(label) for label in test_df['label'].values])
    test_images = test_df.drop('label', axis=1).values / 255.0

    # Split training into train/validation
    val_size = 5000
    validation_data = (train_images[:val_size], train_labels[:val_size])
    training_data = (train_images[val_size:], train_labels[val_size:])
    test_data = (test_images, test_labels)

    return (training_data, validation_data, test_data)

def load_data_wrapper(train_csv_path='sign_mnist_train.csv', test_csv_path='sign_mnist_test.csv'):
    print("Loading from CSVs...")
    tr_d, va_d, te_d = load_data(train_csv_path, test_csv_path)

    print(f"Train size: {len(tr_d[0])}, Val size: {len(va_d[0])}, Test size: {len(te_d[0])}")

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    print("Finished preparing data.")
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((24, 1))
    e[j] = 1.0
    return e