import numpy as np
from data import load_data


def main():

    training_data, test_data, validation_data = load_data("data4students.mat")

    np.save('data/training_data',np.asarray(training_data))
    np.save('data/test_data',np.asarray(test_data))
    np.save('data/validation_data',np.asarray(validation_data))


if __name__ == "__main__":
    main()
