from main import train_and_report, get_parser
from data import load_data
import sys
import numpy as np


def main():
    name = sys.argv[1]
    low = float(sys.argv[2])
    high = float(sys.argv[3])
    samples = int(sys.argv[4])
    training_data, testing_data, validation_data = load_data("data4students.mat")

    for value in np.linspace(low, high, num=samples, endpoint=False):
        print("Testing param values for {}={}".format(name, value))

        parser = get_parser()
        params = parser.parse_args(['--' + name, str(value)])
        train_and_report(training_data, validation_data, params)


if __name__ == '__main__':
    main()
