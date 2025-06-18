"""
mnist_svm
~~~~~~~~~~~~~~~~

A handwritten digit classifier using a manually implemented
linear SVM-like model trained with hinge loss and gradient descent.
"""

import mnist_loader
import numpy as np


class LinearSVM:
    def __init__(self, input_dim, num_classes, lr=0.01, reg=0.01):
        self.W = 0.001 * np.random.randn(input_dim, num_classes)
        self.b = np.zeros((1, num_classes))
        self.lr = lr
        self.reg = reg

    def hinge_loss(self, X, y):
        num_samples = X.shape[0]
        scores = X.dot(self.W) + self.b
        correct_class_scores = scores[np.arange(num_samples), y].reshape(-1, 1)
        margins = np.maximum(0, scores - correct_class_scores + 1)
        margins[np.arange(num_samples), y] = 0

        loss = np.sum(margins) / num_samples
        loss += 0.5 * self.reg * np.sum(self.W * self.W)
        return loss, margins

    def fit(self, X, y, epochs=10, batch_size=100):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                loss, margins = self.hinge_loss(X_batch, y_batch)

                # Gradient
                binary = margins > 0
                row_sum = np.sum(binary, axis=1)
                binary[np.arange(batch_size), y_batch] = -row_sum

                dW = X_batch.T.dot(binary) / batch_size + self.reg * self.W
                db = np.sum(binary, axis=0, keepdims=True) / batch_size

                self.W -= self.lr * dW
                self.b -= self.lr * db

            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    def predict(self, X):
        scores = X.dot(self.W) + self.b
        return np.argmax(scores, axis=1)


def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    X_train, y_train = training_data
    X_test, y_test = test_data

    svm = LinearSVM(input_dim=784, num_classes=10, lr=0.01, reg=0.01)
    svm.fit(X_train, y_train, epochs=10)

    predictions = svm.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Baseline classifier using a manual SVM.")
    print(f"Accuracy: {accuracy * 100:.2f}% ({np.sum(predictions == y_test)} of {len(y_test)} correct)")


if __name__ == "__main__":
    svm_baseline()