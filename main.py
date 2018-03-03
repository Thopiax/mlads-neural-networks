from model import Model
from scipy.io import loadmat


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

    def test_param_values(name, low, high, step=1, ratio=1):
        for i in range(low, high, step):
            print("TESTING PARAM VALUES FOR {}={}".format(name, i/ratio))
            model = Model(training_data, test_data, {name: i/ratio})
            model.build()
            model.train(epochs=1)
            model.evaluate()

    test_param_values("lr", 1, 10, 1, 100)


if __name__ == "__main__":
    main()
