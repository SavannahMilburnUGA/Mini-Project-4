# Trains NN to recognize ASL finger-spelling letters (except J, Z) for 3 different NN architectures
# Fixed to handle data format compatibility
import ASL_loader
import simpleNN
import deepNN
import time
import numpy as np

# Load the ASL data
training_data, validation_data, test_data = ASL_loader.load_data_wrapper()

# Also load raw data for simple network evaluation (needs scalar labels)
raw_training, raw_validation, raw_test = ASL_loader.load_data()

# Convert raw data to format simple network can use
simple_training_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(raw_training[0], raw_training[1])]
simple_validation_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(raw_validation[0], raw_validation[1])]
simple_test_data = [(np.reshape(x, (784, 1)), y) for x, y in zip(raw_test[0], raw_test[1])]

# Storage for results of 3 different NNs
results = {}

# Train simple NN with manual tracking
print("Training simple ASL NN...")
start_time = time.time()
simple_net = simpleNN.Network([784, 30, 24])

simple_training_accuracy = []
simple_validation_accuracy = []

# Run training with manual tracking
for epoch in range(30):
    # Train one epoch at a time to track progress
    simple_net.SGD(training_data, 1, 10, 0.5, test_data=None)
    
    # Calculate accuracies after each epoch using compatible data
    train_acc = simple_net.evaluate(simple_training_data)
    val_acc = simple_net.evaluate(simple_validation_data) 
    
    simple_training_accuracy.append(train_acc)
    simple_validation_accuracy.append(val_acc)
    
    print(f"Simple - Epoch {epoch}: Train {train_acc}/{len(simple_training_data)}, Val {val_acc}/{len(simple_validation_data)}")

simple_net_time = time.time() - start_time
simple_net_accuracy = simple_net.evaluate(simple_test_data)
results['simple'] = {
    'accuracy': simple_net_accuracy, 
    'time': simple_net_time,
    'train_accuracy': simple_training_accuracy,
    'val_accuracy': simple_validation_accuracy,
    'train_cost': [0] * 30,  # Simple network doesn't track cost
    'val_cost': [0] * 30
}

# Train improved NN with full monitoring
print("\nTraining improved simple ASL NN...")
start_time = time.time()
improved_net = deepNN.Network([784, 30, 24])

# Enable ALL monitoring flags to get complete learning curves
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = improved_net.SGD(
    training_data, 30, 10, 0.5, 
    lmbda=0.0,  # No regularization for fair comparison
    evaluation_data=validation_data,  # Use validation_data, not test_data
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True
)

improved_net_time = time.time() - start_time
improved_net_accuracy = improved_net.accuracy(test_data)
results['improved'] = {
    'accuracy': improved_net_accuracy, 
    'time': improved_net_time,
    'train_accuracy': training_accuracy,
    'val_accuracy': evaluation_accuracy,
    'train_cost': training_cost,
    'val_cost': evaluation_cost
}

# Train deep NN with full monitoring
print("\nTraining deep ASL NN...")
start_time = time.time()
deep_net = deepNN.Network([784, 100, 50, 25, 24])

# Enable ALL monitoring flags
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = deep_net.SGD(
    training_data, 30, 10, 0.5,
    lmbda=0.0,  # No regularization for fair comparison
    evaluation_data=validation_data,  # Use validation_data, not test_data
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True
)

deep_net_time = time.time() - start_time
deep_net_accuracy = deep_net.accuracy(test_data)
results['deep'] = {
    'accuracy': deep_net_accuracy, 
    'time': deep_net_time,
    'train_accuracy': training_accuracy,
    'val_accuracy': evaluation_accuracy,
    'train_cost': training_cost,
    'val_cost': evaluation_cost
}

# Print all results
print("\nTraining Complete! Results:")
for network_type, data in results.items():
    if network_type == 'simple':
        total_test = len(simple_test_data)
    else:
        total_test = len(test_data)
    percentage = (data['accuracy'] / total_test) * 100
    print(f"{network_type}: {data['accuracy']}/{total_test} = {percentage:.1f}% in {data['time']:.1f}s")

# Save results for visualization
import pickle
with open('asl_training_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nTraining curves saved to 'asl_training_results.pkl'")
print("You can now run the visualization script to plot learning curves!")