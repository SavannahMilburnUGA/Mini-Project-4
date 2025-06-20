# Neural network implementation using network.py from book
# Implements stochastic gradient descent learning algorithm for a feedforward NN. 
# Gradients calculated using backpropagation
import random
import numpy as np

class Network(object):
    # List sizes: contains # of neurons in each NN layer
    # NN biases/weights initialized randomly using Gaussian distribution w/ mean 0 & variance 1
    # First layer assumed to be input layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # Return output of NN if a is input
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # Train NN using mini-batch stochastic gradient descent
    # training_data: list of tuples (x, y) of training inputs & desired outputs 
    # if test_data provided then NN evaluated against test data after each epoch w/ partial progress printed - slows things down
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    
    # Update NN's weights/biases via gradient descent using backpropagation to a  single mini batch
    # mini_batch: list of tuples (x, y) & eta: learning rate
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # Returns tuple (nabla_b, nabla_w): gradient for cost function C_x
    # nabla_b/nabla_a: layer by layer lists of numpy arrays (like self.biases/self.weights)
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means last layer of neurons
        # l = 2: second to last layer, etc. to take advantage of Python's negative indices in lists
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # Returns # of test inputs for which NN outputs correct result
    # NN's output assumed to be index of whichever neuron in final layer has highest activation
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # Returns vector of partial derivatives of partial C_x / partial a for output activations
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

# Miscellaneous functions
# Sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))