from scipy.io import loadmat
import numpy as np
import pickle
import os

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

    def normalize(self, mean_image):
        self.data = np.subtract(self.data, mean_image)


def calculate_mean_image(data):
    culmulator = np.zeros(PIXELS_IN_IMAGE)

    for image in data:
        for i, pixel in enumerate(image):
            culmulator[i] += pixel

    mean_image = np.divide(culmulator, TRAINING_COUNT)

    return mean_image

def load_data(matlab_file):
    if os.path.isfile("{}.pkl"):
        with open("{}.pkl".format(matlab_file), "rb") as pkl_file:
            return pickle.dump(pkl_file)
    else:
        data = loadmat(matlab_file)
        inputs = data["datasetInputs"][0]
        targets = data["datasetTargets"][0]

        # Assign data
        training_data   = InputData(inputs[0], targets[0])
        testing_data    = InputData(inputs[1], targets[1])
        validation_data = InputData(inputs[2], targets[2])

        # Check data size
        assert len(training_data) == TRAINING_COUNT
        assert len(testing_data) == TESTING_COUNT
        assert len(validation_data) == VALIDATION_COUNT

        # Get mean image to normalize data
        mean_image = calculate_mean_image(training_data.data)

        # Normalize data
        training_data.normalize(mean_image)
        testing_data.normalize(mean_image)
        validation_data.normalize(mean_image)

        # Saved normalized data
        with open("{}.pkl".format(matlab_file), "wb") as pkl_file:
            pickle.dump((training_data, testing_data, validation_data), pkl_file)


        return training_data, testing_data, validation_data
