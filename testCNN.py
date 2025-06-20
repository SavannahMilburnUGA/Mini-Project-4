# Was testing the CNN model to see if it was correct with the ASL data but I literally cannot get the TensorFlow to work right on my computer
import CNN
import ASL_loader

# Load data
training_data, validation_data, test_data = ASL_loader.load_data_wrapper()

# Create CNN
cnn_net = CNN.CNNNetwork()
print("✅ CNN created successfully")

# Build model
model = cnn_net.build_model()
print(f"✅ Model architecture: {len(model.layers)} layers")
print(f"✅ Input shape: {model.input_shape}")
print(f"✅ Output shape: {model.output_shape}")
model.summary()  # Shows detailed architecture

# Test data conversion
x_train, y_train = cnn_net._convert_data_format(training_data[:100])  # Small sample
x_test, y_test = cnn_net._convert_data_format(test_data[:50], is_test=True)

print(f"✅ Training data shape: {x_train.shape}, {y_train.shape}")  # Should be (100, 28, 28, 1), (100, 24)
print(f"✅ Test data shape: {x_test.shape}, {y_test.shape}")        # Should be (50, 28, 28, 1), (50, 24)
print(f"✅ Pixel range: {x_train.min():.3f} to {x_train.max():.3f}")  # Should be 0.0 to 1.0

# Test with small dataset for speed
small_training = training_data[:1000]  # Just 1000 samples
small_test = test_data[:200]           # Just 200 samples

print("Testing CNN training...")
start_time = time.time()
cnn_net.SGD(small_training, 2, 10, 3.0, test_data=small_test)  # Just 2 epochs
test_time = time.time() - start_time
test_accuracy = cnn_net.accuracy(small_test)

print(f"✅ Training completed in {test_time:.1f}s")
print(f"✅ Test accuracy: {test_accuracy}/{len(small_test)} = {test_accuracy/len(small_test)*100:.1f}%")