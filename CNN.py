# Convolutional neural network implementation using Ch. 6 from book ?
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
import ASL_loader

class CNNNetwork:
    def __init__(self):
        self.model = None # Will store TensorFlow/Keras model
        self.history = None # Will store training history
    
    # Build CNN for ASL classification
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(20, (5, 5), activation='sigmoid', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(40, (5, 5), activation='sigmoid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='sigmoid'))
        model.add(layers.Dense(24, activation='softmax'))  # 24 classes for ASL
        return model
    
    # Train CNN to match other NNs
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # Convert your ASL data format to CNN format
        x_train, y_train = self._convert_data_format(training_data)
        x_test, y_test = self._convert_data_format(test_data, is_test=True)
        
        # Build and compile model
        self.model = self.build_model()
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train model
        print("Training CNN...")
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=mini_batch_size,
            validation_data=(x_test, y_test),
            verbose=1
        )
    
    # Evaluate CNN & return # of correct predictions
    def accuracy(self, test_data):
        x_test, y_test = self._convert_data_format(test_data, is_test=True)
        predictions = self.model.predict(x_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(y_test, axis=1)
        correct_predictions = np.sum(predicted_classes == actual_classes)
        return correct_predictions
    
    # Convert ASL_loader format to CNN format
    def _convert_data_format(self, data, is_test=False):
        if is_test:
            # Test data format: [(x, y), (x, y), ...]
            x_data = np.array([x.flatten() for x, y in data])
            y_data = np.array([y for x, y in data])
            # Convert to one-hot for test data
            y_data = utils.to_categorical(y_data, 24)
        else:
            # Training data format: [(x, one_hot_y), (x, one_hot_y), ...]
            x_data = np.array([x.flatten() for x, y in data])
            y_data = np.array([y.flatten() for x, y in data])
        
        # Reshape to CNN format (28, 28, 1)
        x_data = x_data.reshape(-1, 28, 28, 1)
        return x_data, y_data