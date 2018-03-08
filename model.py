from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, SGD
from keras.layers import LeakyReLU
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
import math


class Model(object):
    """Keras Model wrapper."""
    input_shape = (30, 30, 1)

    def __init__(self, training_data, validation_data, params={}):
        self.training_data = training_data
        self.validation_data = validation_data
        self.params = params

    def build(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation=self.params.hidden_activation,
                              input_shape=Model.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation=self.params.hidden_activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(self.params.dropout_first))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=self.params.hidden_activation))
        self.model.add(Dropout(self.params.dropout_second))
        self.model.add(Dense(7, activation=self.params.output_activation))

        self.model.summary()

        self.model.compile(
            # Stochastic gradient descent
            # Learning rate, momentum, learning rate decay
            optimizer=SGD(lr=self.params.lr,
                          momentum=self.params.momentum),


            # Objective function which we wish to minimise
            loss=self.params.loss,

            # Metrics used to judge the effectiveness of our model
            # Accuracy is used for classification problems
            metrics=['categorical_accuracy']
        )

        #print(self.model.model)
        #plot_model(self.model.model, to_file="model.png")

    def train(self, epochs=20, batch_size=32):
        decay = eval("self.build_{}()".format(self.params.lr_scheduler))

        history = self.model.fit(self.training_data.data,
                                 self.training_data.targets,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.validation_data.data,
                                                  self.validation_data.targets),
                                 callbacks=[
                                    EarlyStopping(monitor='val_loss',
                                                  min_delta=0.01,
                                                  patience=self.params.early_stopping_patience,
                                                  verbose=0,
                                                  mode='auto'),
                                    LearningRateScheduler(decay, verbose=1),
                                    MomentumRateScheduler(verbose=1)
                                 ],
                                 verbose=1)

        return history

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.validation_data.data,
                                             self.validation_data.targets)

        print("Loss: {}\n Accuracy: {}".format(loss, accuracy))

        return loss, accuracy

    def build_step_decay(self):
        return lambda epoch, lr: self.params.lr * math.pow((1/2), math.floor((1+epoch)/self.params.decay_rate))

    def build_exponential_decay(self):
        return lambda epoch, lr: self.params.lr * math.exp(-self.params.decay_rate*epoch)

    def build_inverse_decay(self):
        return lambda epoch, lr: self.params.lr/(1 + self.params.decay_rate*epoch)


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
    plt.legend(['train', 'validation', 'test'], loc='upper left')
    plt.savefig('loss.png')

class MomentumRateScheduler(Callback):
    def __init__(self, verbose=0):
        super(MomentumRateScheduler, self).__init__()
        self.momentum = None
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs={}):
        if self.momentum == None:
            self.momentum = float(K.get_value(self.model.optimizer.momentum))
        elif self.momentum < 0.9:
            self.momentum += 0.0125
            self.model.optimizer.momentum = self.momentum
            if self.verbose > 0:
                print('\nEpoch %05d: MomentumRateScheduler increasing momentum '
                           'rate to %s.' % (epoch + 1, self.momentum))
        else:
            if self.verbose > 0:
                print('\nEpoch %05d: MomentumRateScheduler limit reached')
