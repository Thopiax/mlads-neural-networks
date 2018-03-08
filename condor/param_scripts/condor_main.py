from main import train_and_report, get_parser
from data import load_data
import sys
import numpy as np


def main():
    name    = sys.argv[1]
    t       = sys.argv[2]
    low     = float(sys.argv[3])
    high    = float(sys.argv[4])
    samples = int(sys.argv[5])
    is_int  = bool(sys.argv[6])
    training_data, testing_data, validation_data = load_data("data4students.mat")

    np.random.seed(42)

    if is_int:
       rand = np.random.random_integers(low, high, samples)
    else:
       rand = (high - low) * np.random.random_sample(samples) + low

    for value in rand:
        print("Testing param values for {}={}".format(name, value))
        
        if is_int: 
            params_in = [str(np.random.randint(2000)) for i in range(value)]
        else:
            params_in = str(value)
            

        parser = get_parser()
        params = parser.parse_args(["--timestamp", str(t), "--{}".format(name)] + params_in)
        train_and_report(training_data, validation_data, params)


if __name__ == '__main__':
    main()
