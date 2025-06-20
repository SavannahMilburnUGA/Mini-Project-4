# Trains NN to recognize ASL finger-spelling letters (except J, Z) for 3 different NN architectures
# Mimics interactive Python shell training from textbook digit classifier 
import ASL_loader
import network
import time

# Load the ASL data once
training_data, validation_data, test_data = ASL_loader.load_data_wrapper()

# Storage for results of 3 different NNs
results = {}

# Train simple NN
print("Training simple ASL NN...")

# Train deep NN 
print("\nTraining deep ASL NN...")
start_time = time.time()
# Create deep neural network (3 hidden layers w/ 100, 50, 25 neurons)
deepNN = network.Network([784, 100, 50, 25, 24])
# Train using SGD - same parameters as textbook digit classifier
deepNN.SGD(training_data, 30, 10, 3.0, test_data=test_data)
deepNN_time = time.time() - start_time
deepNN_accuracy = deepNN.accuracy(test_data)
results['deep'] = {'accuracy': deepNN_accuracy, 'time': deepNN_time}

# Train CNN 

