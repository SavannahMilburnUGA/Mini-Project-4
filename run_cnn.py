from CNN import CNNNetwork  # Assuming your CNN class is in CNN.py
import ASL_loader

# Load your data
training_data, validation_data, test_data = ASL_loader.load_data_wrapper(
    train_csv_path='sign_mnist_train.csv',
    test_csv_path='sign_mnist_test.csv'
)

# Create and train CNN
cnn = CNNNetwork()
cnn.SGD(training_data, epochs=5, mini_batch_size=32, eta=0.1, test_data=validation_data)

# Evaluate accuracy
correct = cnn.accuracy(test_data)
print(f"Correct classifications on test set: {correct} / {len(test_data)}")

print("Loading data...")
training_data, validation_data, test_data = ASL_loader.load_data_wrapper(
    train_csv_path='sign_mnist_train.csv',
    test_csv_path='sign_mnist_test.csv'
)
print("Training finshed!")