from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np


class Model(object):
    """Keras Model wrapper."""

    def __init__(self, training_data, test_data, params = {}):
        self.training_data = training_data
        self.test_data = test_data
        self.params = {"lr": 0.01,
                       "momentum": 0.5,
                       "decay": 1e-6}

        self.params.update(params)

    def build(self):
        self.model = Sequential()

        # First layer takes 900 pixels of a 30 x 30 image
        self.model.add(Dense(300, input_dim=900, activation='linear'))

        # Hidden layers
        self.model.add(Dense(30))
        self.model.add(LeakyReLU(alpha=0.5))

        # Final layer outputs one of the 7 emotions
        self.model.add(Dense(7, activation='softmax'))

<<<<<<< HEAD
=======
        plot_model(self.model, show_shapes=True, to_file='model.png')

    def train(self, epochs=100, batch_size=128):
>>>>>>> hidden-layers
        self.model.compile(
            # Stochastic gradient descent
            # Learning rate, momentum, learning rate decay
            optimizer=SGD(**self.params),

            # Objective function which we wish to minimise
            loss='categorical_crossentropy',

            # Metrics used to judge the effectiveness of our model
            # Accuracy is used for classification problems
            metrics=['accuracy']
        )

        plot_model(self.model, show_shapes=True, to_file='model.png')

    def train(self, epochs=10, batch_size=32):
        history = self.model.fit(self.training_data.data,
                                 self.training_data.targets,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=0)

        return history

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_data.data,
                                             self.test_data.targets)

        print("Loss:{}\nAccuracy:{}".format(loss, accuracy))

        return loss, accuracy


def plot_history(history):
    # Plot accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy.png')

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss.png')
