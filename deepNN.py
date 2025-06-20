# Neural network implementation using debugged network2.py from book
# Implements stochastic gradient descent learning algorithm for a feedforward NN. 
# Includes cross-entropy cost function, regularization, better network weight initialization.
import json
import random
import sys
import numpy as np


# Define the quadratic and cross-entropy cost functions
class QuadraticCost(object):
    # Returns cost associated w/ output a & desired output y
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    # Return error change from output layer of NN
    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)

# Return cost of output a and desired output y
class CrossEntropyCost(object):
    # np.nan_to_num ensured numerical stability (if a & y have 1.0 in same slot, 
    # (1-y)*np.log(1-a) returns nan). Ensures converted to correct 0.0
    @staticmethod
    def fn(a, y): 
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    # Return error change from output layer - z is not used - only for consistency
    @staticmethod
    def delta(z, a, y):
        return (a-y)


# Main NN class
class Network(object):
    # List sizes: contains # of neurons in each NN layer
    # self.default_weight_initializer: NN biases/weights initialized randomly
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    # Initialize weights using Gaussian distribution w/ mean 0 & SD 1 over sq. root of # connected weights.
    # Initialize biases using Gaussian distribution w/ mean 0 & SD 1
    # Don't set biases for first layer (input layer)
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    # Initialize weights using Gaussian distribution w/ mean 0 & SD 1 over sq. root of # connected weights.
    # Initialize biases using Gaussian distribution w/ mean 0 & SD 1
    # Don't set biases for first layer (input layer)
    # Better to use default weight initializer usually 
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    # Return output of NN if input is a
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    # Train NN using mini-batch stochastic gradient descent
    # training_data: list of tuples (x, y) of training inputs & desired outputs evaluation_data: validation/test data - can set flags to monitor cost/accuracy 
    # Returns tuple of 4 lists: per-epoch costs on evaluation data, accuracies on " ", costs on training data, accuracies on " "
    # All values evaluated at end of training epoch
    # Ex: if we train 30 epochs, first element of tuple is 30-element list containg cost on evaluation data at end of each epoch
    # Lists are empy if flag not set
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
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
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    # Update NN weights/biases via gradient descent using backpropagation to a single mini batch
    # mini_batch: list of tuples (x, y)
    # eta: learning rate
    # lmbda: regularization param
    # n: total size of training data set
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # Return tuple (nabla_b, nabla_w): gradient for cost function C_x
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
        delta = (self.cost).delta(zs[-1], activations[-1], y)
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

    # Return # of inputs in data for which NN outputs correct result
    # NN's output assumed to be index of neuron in final layer w/ highest activation
    def accuracy(self, data, convert=False):
        # flag convert set to F if data is validation/test data (usual case)
        # convert set to T if data is training data
        # Use diff. representations for diff. data sets bc. more efficient
        # Program usually evaluates cost of training data & accuracy on other data sets
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    # Returns total cost for data
    # flag convert set to F if data is training data (usual case)
    # convert set to T if data is validation/test data 
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    # Save NN to file filename
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

# Loading a NN from filename
# Return instance of NN
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

# Miscellaneous functions
# Return 24-D unit vector w/ 1.0 in jth position & 0s else
# Used to convert a sign to corresponding desired output from NN
def vectorized_result(j):
    e = np.zeros((24, 1))
    e[j] = 1.0
    return e

# Sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))