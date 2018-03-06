from main import train_and_report, load_normalized_data, get_parser
import sys
import numpy as np


def main():
    name = sys.argv[1]
    low = float(sys.argv[2])
    high = float(sys.argv[3])
    samples = int(sys.argv[4])
    training_data, validation_data = load_normalized_data()

    for value in np.linspace(low, high, num=samples, endpoint=False):
        print("Testing param values for {}={}".format(name, value))

        parser = get_parser()
        params = parser.parse_args(['--' + name, str(value)])
        train_and_report(training_data, validation_data, params)


if __name__ == '__main__':
    main()
