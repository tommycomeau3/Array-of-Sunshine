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


#load and preprocess the dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

#initialize and compile the neural network model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#display the image with its prediction
def display_prediction(index, predictions, labels, images):
    pred_label = np.argmax(predictions[index])
    actual_label = labels[index]
    plt.imshow(images[index], cmap=plt.cm.binary)
    plt.title("Predicted: {} ({:.0f}%), Actual: {}".format(
        class_names[pred_label], 100 * np.max(predictions[index]), class_names[actual_label]))
    plt.colorbar()
    plt.grid(False)
    plt.show()

#main function
def main():
    # split data in to training and testing.
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # initialize the model.
    model = create_model()

    # train the model using the training data provided from load_and_preprocess_data.
    model.fit(x_train, y_train, epochs=10)

    # now that we trained the model; calculate loss / accuracy using testing data from load_and_preprocess_data.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))

    # get result from model.
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    print(classification_report(y_test, predicted_classes, target_names=class_names))

    random_index = random.randint(0, len(x_test) - 1)
    display_prediction(random_index, predictions, y_test, x_test)


#class names from dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if __name__ == "__main__":
    main()


