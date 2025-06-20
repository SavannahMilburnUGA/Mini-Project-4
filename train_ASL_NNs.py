# Trains NN to recognize ASL finger-spelling letters (except J, Z) for 3 different NN architectures
# Mimics interactive Python shell training from textbook digit classifier 
import ASL_loader
import simpleNN
import deepNN
import CNN
import time

# Load the ASL data once
training_data, validation_data, test_data = ASL_loader.load_data_wrapper()

# Storage for results of 3 different NNs
results = {}

# Train simple NN
print("Training simple ASL NN...")
start_time = time.time()
# Create simple neural network w/ 30 hidden neurons
simple_net = simpleNN.Network([784, 30, 24])
# Train using SGD - same parameters as textbook digit classifier
simple_net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
simple_net_time = time.time() - start_time
simple_net_accuracy = simple_net.evaluate(test_data)  # Use evaluate() not accuracy()
results['simple'] = {'accuracy': simple_net_accuracy, 'time': simple_net_time}

# Train deep NN 
print("\nTraining deep ASL NN...")
start_time = time.time()
# Create deep neural network (3 hidden layers w/ 100, 50, 25 neurons)
deep_net = deepNN.Network([784, 100, 50, 25, 24])
# Train using SGD - same parameters as textbook digit classifier
deep_net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
deep_net_time = time.time() - start_time
deep_net_accuracy = deep_net.accuracy(test_data)
results['deep'] = {'accuracy': deep_net_accuracy, 'time': deep_net_time}

# # Train CNN - I cannot get the TensorFlow to work on my machine so scratch this CNN
# print("\nTraining CNN...")
# start_time = time.time()
# # Create CNN 
# cnn_net = CNN.CNNNetwork()
# # Train using same parameters as other networks
# cnn_net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# cnn_net_time = time.time() - start_time
# cnn_net_accuracy = cnn_net.accuracy(test_data)
# results['CNN'] = {'accuracy': cnn_net_accuracy, 'time': cnn_net_time}

# Print all results
print("\nTraining Complete! Results:")
for network_type, data in results.items():
    total_test = len(test_data)
    percentage = (data['accuracy'] / total_test) * 100
    print(f"{network_type}: {data['accuracy']}/{total_test} = {percentage:.1f}% in {data['time']:.1f}s")