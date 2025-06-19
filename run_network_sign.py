import numpy as np
from tqdm import tqdm
import sign_language_loader
from network import Network

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = sign_language_loader.load_data_wrapper()

# Format data for Network 1: x is (784,1), y is (24,1) for training, and label index for test
training_data = [(x.reshape(784, 1), y.reshape(24, 1)) for x, y in zip(x_train, y_train)]
test_data = [(x.reshape(784, 1), np.argmax(y)) for x, y in zip(x_test, y_test)]

# Define the network
net = Network([784, 64, 24])

# Patch Network.SGD to include tqdm and percentage output
from types import MethodType

def patched_SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in tqdm(mini_batches, desc=f"Epoch {j+1}/{epochs}", leave=False):
            self.update_mini_batch(mini_batch, eta)
        if test_data:
            acc = self.evaluate(test_data)
            print(f"Epoch {j+1}: {acc / n_test:.2%} ({acc} / {n_test})")
        else:
            print(f"Epoch {j+1} complete")

# Patch the method
net.SGD = MethodType(patched_SGD, net)

# Train the network
net.SGD(training_data, epochs=10, mini_batch_size=20, eta=3.0, test_data=test_data)