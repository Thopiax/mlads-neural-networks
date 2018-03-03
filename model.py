from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt


class Model(object):
    """Keras Model wrapper."""

    def __init__(self, training_data, test_data, params):
        self.training_data = training_data
        self.test_data = test_data
        self.params = {"lr": 0.01,
                       "momentum": 0.0,
                       "decay": 0.0}

        self.params.update(params)

    def build(self):
        self.model = Sequential()

        # First layer takes 900 pixels of a 30 x 30 image
        self.model.add(Dense(300, input_dim=900, activation='relu'))

        # Hidden layers
        self.model.add(Dense(30, activation='relu'))

        # Final layer outputs one of the 7 emotions
        self.model.add(Dense(7, activation='relu'))

        plot_model(self.model, show_shapes=True, to_file='model.png')

    def train(self, epochs=10, batch_size=32):
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

        history = self.model.fit(self.training_data.data,
                                 self.training_data.targets,
                                 epochs=epochs,
                                 batch_size=batch_size)

        return history

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_data.data,
                                             self.test_data.targets)

        print("LOSS:{}\nACCURACY:{}".format(loss, accuracy))


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
