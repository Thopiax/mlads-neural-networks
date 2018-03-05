from data import load_data
from model import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from random import randint


def main():
    training_data, test_data, validation_data = load_data("data4students.mat")

    for i in range(100):
        model = Model(training_data, test_data)
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

        model.train(epochs=100, batch_size=128)
        loss, accuracy = model.evaluate()


if __name__ == "__main__":
    main()
