# Trains NN to recognize ASL finger-spelling letters (except J, Z)
# Mimics interactive Python shell training from textbook digit classifier 
import ASL_loader
import network

# Load the ASL data
training_data, validation_data, test_data = ASL_loader.load_data_wrapper()

# Create network with 30 hidden neurons (same as textbook digit classifier)
net = network.Network([784, 30, 24])

# Train using SGD - same parameters as textbook digit classifier
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)