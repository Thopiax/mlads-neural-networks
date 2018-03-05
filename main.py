from model import Model
from scipy.io import loadmat
from random import randint
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import plot_model

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


def main():
    training_data, test_data, validation_data = load_data()

    # def test_param_values(name, low, high, step=1, ratio=1):
    #     for i in range(low, high, step):
    #         print("TESTING PARAM VALUES FOR {}={}".format(name, i/ratio))
    #         model = Model(training_data, test_data, {name: i/ratio})
    #         model.build()
    #         model.train(epochs=500, batch_size=128)
    #         model.evaluate()

    with open("nn-history.csv", "a") as csvfile:
        writer = csv.writer(csvfile)

        for i in range(100):
            model = Model(training_data, test_data)
            model.model = Sequential()

            hidden_layers = randint(3, 8)
            layer_neurons = [randint(0, 2000) for x in range(hidden_layers + 1)]

            print("Number of hidden layers: {}".format(hidden_layers))
            print("Neuron layout: {}".format(layer_neurons))

            model.model.add(Dense(layer_neurons[0], input_dim=900))
            model.model.add(LeakyReLU(alpha=0.5))
            for i in range(0, hidden_layers):
                model.model.add(Dense(layer_neurons[i + 1]))
                model.model.add(LeakyReLU(alpha=0.5))

            model.model.add(Dense(7, activation='softmax'))

            plot_model(model.model, show_shapes=True, to_file='models/model-{}.png'.format(i))

            model.train(epochs=1000, batch_size=128)
            loss, accuracy = model.evaluate()

            writer.writerow([i, hidden_layers, layer_neurons, accuracy, loss])


if __name__ == "__main__":
    main()
