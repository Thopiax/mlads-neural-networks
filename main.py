from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt
from scipy.io import loadmat

# import numpy
# numpy.set_printoptions(threshold=numpy.nan)


class InputData:
    def __init__(self, data, targets):
        assert len(data) == len(targets)

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)


def load_data():
    data = loadmat("data4students.mat")
    inputs = data["datasetInputs"][0]
    targets = data["datasetTargets"][0]

    training_data = InputData(inputs[0], targets[0])
    test_data = InputData(inputs[1], targets[1])
    validation_data = InputData(inputs[2], targets[2])

    assert len(training_data) == 25120
    assert len(test_data) == 3589
    assert len(validation_data) == 3589

    return training_data, test_data, validation_data


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


def main():
    training_data, test_data, validation_data = load_data()

    model = Sequential()

    # First layer takes 900 pixels of a 30 x 30 image
    model.add(Dense(300, input_dim=900, activation='relu'))

    # Hidden layers
    model.add(Dense(30, activation='relu'))

    # Final layer outputs one of the 7 emotions
    model.add(Dense(7, activation='relu'))

    model.summary()
    plot_model(model, show_shapes=True, to_file='model.png')

    model.compile(
        # Learning rate, momentum, learning rate decay, Nesterov momentum mode
        optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),

        # Objective function which we wish to minimise
        loss='categorical_crossentropy',

        # Metrics used to judge the effectiveness of our model
        # Accuracy is used for classification problems
        metrics=['accuracy']
    )

    history = model.fit(training_data.data,
                        training_data.targets,
                        epochs=10,
                        batch_size=32,
                        validation_data=(validation_data.data,
                                         validation_data.targets))

    plot_history(history)

    loss_and_accuracy = model.evaluate(
        validation_data.data,
        validation_data.targets,
        batch_size=32)

    loss = loss_and_accuracy[0]
    accuracy = loss_and_accuracy[1]

    print("Final loss: {:f}".format(loss))
    print("Final accuracy: {:f}".format(accuracy))


if __name__ == "__main__":
    main()
