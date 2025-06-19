import numpy as np
from tqdm import tqdm
import sign_language_loader
from network2 import Network, CrossEntropyCost

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = sign_language_loader.load_data_wrapper()

# Format for network2: x is (784,1), y is (24,1) for training, and label index for test
training_data = [(x.reshape(784, 1), y.reshape(24, 1)) for x, y in zip(x_train, y_train)]
test_data = [(x.reshape(784, 1), np.argmax(y)) for x, y in zip(x_test, y_test)]

# Define the network
net = Network([784, 64, 24], cost=CrossEntropyCost)

# Patch Network.SGD to include tqdm in mini-batch loop
from types import MethodType

def patched_SGD(self, training_data, epochs, mini_batch_size, eta,
                lmbda=0.0,
                evaluation_data=None,
                monitor_evaluation_cost=False,
                monitor_evaluation_accuracy=False,
                monitor_training_cost=False,
                monitor_training_accuracy=False):
    if evaluation_data: n_data = len(evaluation_data)
    n = len(training_data)
    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []
    for j in range(epochs):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in tqdm(mini_batches, desc=f"Epoch {j+1}/{epochs}", leave=False):
            self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
        print(f"Epoch {j+1} training complete")
        if monitor_training_cost:
            cost = self.total_cost(training_data, lmbda)
            training_cost.append(cost)
            print("Training cost:", cost)
        if monitor_training_accuracy:
            acc = self.accuracy(training_data, convert=True)
            training_accuracy.append(acc)
            print(f"Training accuracy: {acc / n:.2%} ({acc} / {n})")
        if monitor_evaluation_cost:
            cost = self.total_cost(evaluation_data, lmbda, convert=True)
            evaluation_cost.append(cost)
            print("Eval cost:", cost)
        if monitor_evaluation_accuracy:
            acc = self.accuracy(evaluation_data)
            evaluation_accuracy.append(acc)
            print(f"Eval accuracy: {acc / n_data:.2%} ({acc} / {n_data})")

# Patch the method
net.SGD = MethodType(patched_SGD, net)

# Train the network with live batch progress
net.SGD(training_data, epochs=10, mini_batch_size=20, eta=3.0,
        evaluation_data=test_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=True)
