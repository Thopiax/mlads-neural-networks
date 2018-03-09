import sys
from keras.models import load_model
from main import get_parser
from model import Model
from scipy.io import loadmat
from data import calculate_mean_image
from data import InputData

def test_network(model, testing_data):
    predictions = model.predict(testing_data).argmax(-1)

    return predictions


def load_data(matlab_file):
    loaded = loadmat(matlab_file)
    data = loaded["hiddenInputs"][0]
    targets = loaded["hiddenTargets"][0]

    # Get mean image to normalize data
    mean_image = calculate_mean_image(data)

    data = InputData(data, targets) #[0 for x in data])

    # Normalize data
    data.normalize(mean_image)

    data.data = data.data.reshape(data.data.shape[0], 30, 30, 1)

    return data


def main():
    if len(sys.argv) != 2:
        print("Please specify a path to a .mat file containing the testing data")
        return

    data = load_data(sys.argv[1])

    parser = get_parser()
    params = parser.parse_args([])

    model = Model(None, None, params)
    model.build()
    model.model.load_weights('trained_model.h5')

    predictions = test_network(model.model, data.data)

    print(predictions)
    correct = 0

    for i, datum in enumerate(data.targets):
        datum = list(datum).index(1)

        print('Prediction for image ' + str(i) + ': ' + str(predictions[i]) + ', expected ' + str(datum))
        correct += 1 if predictions[i] == datum else 0

    print('Overall CR: ' + str(correct / len(predictions) * 100) + '%')


if __name__ == "__main__":
    main()
