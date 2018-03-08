from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
import math


class Model(object):
    """Keras Model wrapper."""

    def __init__(self, training_data, validation_data, params={}):
        self.training_data = training_data
        self.validation_data = validation_data
        self.params = params

    def build(self):
        self.model = Sequential()

        # First layer takes 900 pixels of a 30 x 30 image
        self.model.add(Dense(self.params.hidden_layer_neurons[0],
                             input_dim=900,
                             activation='linear',
                             kernel_initializer=self.params.weight_initialisation))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.346))

        # Hidden layers
        for neurons in self.params.hidden_layer_neurons[1:]:
            self.model.add(Dense(neurons,
                                 activation=self.params.hidden_activation,
                                 kernel_initializer=self.params.weight_initialisation))

            self.model.add(Dropout(0.4))
        # Final layer outputs one of the 7 emotions
        self.model.add(Dense(7,
                             activation=self.params.output_activation,
                             kernel_initializer=self.params.weight_initialisation))

        self.model.compile(
            # Stochastic gradient descent
            # Learning rate, momentum, learning rate decay
            optimizer=SGD(lr=self.params.lr,
                          momentum=self.params.momentum,
                          decay=self.params.lr_decay),

            # Objective function which we wish to minimise
            loss=self.params.loss,



            # Metrics used to judge the effectiveness of our model
            # Accuracy is used for classification problems
            metrics=['categorical_accuracy']
        )

        print(self.model.model)
        plot_model(self.model.model, to_file="model.png")

    def train(self, epochs=20, batch_size=32):
        step_decay = self.build_step_decay()

        history = self.model.fit(self.training_data.data,
                                 self.training_data.targets,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.validation_data.data,
                                                  self.validation_data.targets),
                                 callbacks=[
                                    EarlyStopping(monitor='val_loss',
                                                  min_delta=0,
                                                  patience=self.params.early_stopping_patience,
                                                  verbose=0,
                                                  mode='auto'),
                                    LearningRateScheduler(step_decay, verbose=1)
                                 ],
                                 verbose=1)

        return history

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.validation_data.data,
                                             self.validation_data.targets)

        print("Loss: {}\n Accuracy: {}".format(loss, accuracy))

        return loss, accuracy

    def build_step_decay(self):
        return lambda epoch, lr: self.params.lr * math.pow((1/2), math.floor((1+epoch)/5))


def plot_history(history):
    import matplotlib.pyplot as plt

    # Plot accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
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
