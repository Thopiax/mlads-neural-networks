from scipy.io import loadmat
import numpy as np
import pickle
import os
from keras.utils import normalize as norm

PIXELS_IN_IMAGE = 900
TRAINING_COUNT = 25120
TESTING_COUNT = 3589
VALIDATION_COUNT = 3589


class InputData:
    def __init__(self, data, targets, mean=None, std=None):
        assert len(data) == len(targets)

        if mean is None and std is None:
            mean = np.mean(data)
            std = np.std(data)

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)


def load_data(matlab_file):
    data = loadmat(matlab_file)
    inputs = data["datasetInputs"][0]
    targets = data["datasetTargets"][0]

    # Assign data
    training_data = InputData(norm(inputs[0]), targets[0])
    testing_data = InputData(norm(inputs[1]), targets[1])
    validation_data = InputData(norm(inputs[2]), targets[2])

    # Check data size
    assert len(training_data) == TRAINING_COUNT
    assert len(testing_data) == TESTING_COUNT
    assert len(validation_data) == VALIDATION_COUNT

    training_data.data = training_data.data.reshape(training_data.data.shape[0], 30, 30, 1)
    validation_data.data = validation_data.data.reshape(validation_data.data.shape[0], 30, 30, 1)
    testing_data.data = test_data.data.reshape(testing_data.data.shape[0], 30, 30, 1)

    return training_data, testing_data, validation_data
