# Load ASL fingerspelling dataset from Kaggle
# sign_language_loader.py
# explore_dataset.py
import kagglehub
import os
import pandas as pd
import numpy as np
# Return Kaggle Sign Language MNIST data as a tuple containing training data, validation data, test data
def load_data():
    # Download dataset
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    
    # Load training data
    train_df = pd.read_csv(os.path.join(path, 'sign_mnist_train.csv'))
    train_labels = train_df['label'].values
    train_images = train_df.drop('label', axis=1).values / 255.0  # Normalize to 0-1
    
    # Load test data  
    test_df = pd.read_csv(os.path.join(path, 'sign_mnist_test.csv'))
    test_labels = test_df['label'].values
    test_images = test_df.drop('label', axis=1).values / 255.0
    
    # Split training into train/validation (similar to original MNIST)
    val_size = 5000
    validation_data = (train_images[:val_size], train_labels[:val_size])
    training_data = (train_images[val_size:], train_labels[val_size:])
    test_data = (test_images, test_labels)
    
    return (training_data, validation_data, test_data)

# Return data in format for NN
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

# Return 24-D vector w/ 1.0 in jth position
def vectorized_result(j):
    e = np.zeros((24, 1))
    e[j] = 1.0
    return e