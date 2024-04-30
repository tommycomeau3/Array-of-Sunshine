#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Load the Fashion MNIST dataset into training and testing sets
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Normalize the image data to 0-1 range by dividing by 255
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Return the processed data sets
    return x_train, y_train, x_test, y_test

# Initialize and compile the neural network model
def create_model():
    # Create a sequential model
    model = Sequential([
        # First layer to flatten the input 28x28 image into a vector of 784 pixels
        Flatten(input_shape=(28, 28)),
        # Dense layer with 128 neurons and ReLU activation function
        Dense(128, activation='relu'),
        # Output dense layer with 10 neurons for each class, using softmax for probabilistic output
        Dense(10, activation='softmax')
    ])
    # Compile the model with Adam optimizer, cross-entropy loss, and accuracy metrics
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Display the image with its prediction
def display_prediction(index, predictions, labels, images):
    # Get the index of the highest probability class from the prediction
    pred_label = np.argmax(predictions[index])
    # Get the actual label for the image
    actual_label = labels[index]
    # Display the image
    plt.imshow(images[index], cmap=plt.cm.binary)
    # Add a title to the image showing predicted and actual labels and prediction confidence
    plt.title("Predicted: {} ({:.0f}%), Actual: {}".format(
        class_names[pred_label], 100 * np.max(predictions[index]), class_names[actual_label]))
    plt.colorbar()
    plt.grid(False)
    plt.show()

# Main function
def main():
    # Load and preprocess the data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Initialize the model
    model = create_model()
    
    # Train the model using the provided training data for 10 epochs
    model.fit(x_train, y_train, epochs=10)
    
    # Evaluate the model using the test data to calculate loss and accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    
    # Predict classes using the model on the test data
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Print a classification report comparing predictions to actual labels
    print(classification_report(y_test, predicted_classes, target_names=class_names))
    
    # Display a random prediction from the test data
    random_index = random.randint(0, len(x_test) - 1)
    display_prediction(random_index, predictions, y_test, x_test)

# Class names from the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Run the main function if the script is executed as the main program
if __name__ == "__main__":
    main()
