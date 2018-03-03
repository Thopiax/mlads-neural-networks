from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class Model(object):
    """Keras Model wrapper."""

    def __init__(self, training_data, test_data, params):
        super(Model, self).__init__()
        self.training_data = training_data
        self.test_data     = test_data
        self.params        = { "lr":0.01, "momentum":0.0, "decay":0.0, "nesterov":False }

        self.params.update(params)

    def build(self):
        self.model = Sequential()
        # First layer takes 900 pixels of a 30 x 30 image
        self.model.add(Dense(300, input_dim=900, activation='relu'))
        # Final layer outputs one of the 7 emotions
        self.model.add(Dense(7, activation='relu'))

    def train(self, epochs=10):
        self.model.compile(
            # Stochastic gradient descent
            # Learning rate, momentum, learning rate decay, Nesterov momentum mode
            optimizer=SGD(**self.params),

            # Objective function which we wish to minimise
            loss='categorical_crossentropy',

            # Metrics used to judge the effectiveness of our model
            # Accuracy is used for classification problems
            metrics=['accuracy']
        )

        self.model.fit(self.training_data.data,
                       self.training_data.targets,
                       epochs=epochs,
                       batch_size=None)

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.test_data.data, self.test_data.targets)

        print("LOSS:{}\nACCURACY:{}".format(loss, accuracy))
