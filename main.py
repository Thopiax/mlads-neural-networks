import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.utils import plot_model
from model import Model
from random import randint


def main():

    training_data = np.load("data/training_data.npy").tolist()
    test_data = np.load("data/test_data.npy").tolist()
    validation_data = np.load("data/validation_data.npy").tolist()


    def test_param_values(name, low, high, step=1, ratio=1):
        for i in range(low, high, step):
            print("Testing param values for {}={}".format(name, i/ratio))
            model = Model(training_data, test_data, {name: i/ratio})
            model.build()
            model.train(epochs=1)
            model.evaluate()
            print()

    for i in range(100):
        print("Creating model...")
        model = Model(training_data, test_data)
        print("Building model...")
        model.model = Sequential()

        hidden_layers = randint(3, 8)
        layer_neurons = [randint(0, 2000) for x in range(hidden_layers + 1)]

        print("Number of hidden layers: {}".format(hidden_layers))
        print("Neuron layout: {}".format(layer_neurons))

        model.model.add(Dense(900, input_dim=900, activation='linear'))

        for i in range(0, hidden_layers):
            model.model.add(Dense(layer_neurons[i + 1]))
            model.model.add(LeakyReLU(alpha=0.5))

        model.model.add(Dense(7, activation='softmax'))

        plot_model(model.model, show_shapes=True,
                   to_file='models/model-{}.png'.format(i))

        print("Training model...")
        model.train(epochs=100, batch_size=128)
        print("Evaluating model...")
        loss, accuracy = model.evaluate()

        writer.writerow([i, hidden_layers, layer_neurons, accuracy, loss])


if __name__ == "__main__":
    main()
