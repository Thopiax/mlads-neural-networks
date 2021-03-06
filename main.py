import numpy as np

np.random.seed(1337)

import keras
from model import Model, plot_accuracy, plot_loss
import argparse
from argparse import Namespace
import os
import csv
import httplib2
from datetime import datetime
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from data import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Machine Learning Coursework 2'


def get_credentials(flags):
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        flags.noauth_local_webserver = True
        credentials = tools.run_flow(flow, store, flags)
        print('Storing credentials to ' + credential_path)

    return credentials


def report_local(params, loss, accuracy):
    if loss == "nan":
        return

    print("reporting:\n\tparams={}\n\tloss:{}\n\taccuracy".format(params, loss, accuracy))

    with open("./results/results-{}.csv".format(params.timestamp), "a+") as csvfile:
        writer = csv.writer(csvfile)
        values = [
            str(datetime.now()),
            str(params.loss),
            str(params.hidden_activation),
            str(params.output_activation),
            str(params.weight_initialisation),
            str(params.epochs),
            str(params.batch_size),
            str(params.lr),
            str(params.lr_scheduler),
            str(params.decay_rate),
            str(params.early_stopping_patience),
            str(params.dropout_first),
            str(params.dropout_second),
            str(params.momentum),
            str(params.l1),
            str(params.l2),
            str(loss),
            str(accuracy)
        ]

        writer.writerow(values)
        print(", ".join(values))


def report_run(params, loss, accuracy):
    credentials = get_credentials(params)
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                    'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discoveryUrl)

    spreadsheet_id = '1tx0n4QN-tzjZNfqvqi74hs3RHHkRb5tZ2AxS4xMIxY0'
    range_name = 'Sheet1'

    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        body={
            'values': [[
                str(datetime.now()),
                ', '.join(map(str, params.hidden_layer_neurons)),
                params.loss,
                params.hidden_activation,
                params.output_activation,
                params.weight_initialisation,
                params.epochs,
                params.batch_size,
                params.lr,
                params.lr_decay,
                params.momentum,
                loss,
                accuracy
            ]]
        }).execute()


def train_and_report(training_data, validation_data, params):
    model = Model(training_data, validation_data, params)
    model.build()
    history = model.train(epochs=params.epochs, batch_size=params.batch_size)

    loss, accuracy = model.evaluate()
    report_local(params, loss, accuracy)

    return history, model


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run model with given arguments.',
        parents=[tools.argparser])

    parser.add_argument('--data', type=str, default='data4students.mat')
    parser.add_argument('--lr', type=float, default=0.207)
    parser.add_argument('--lr_scheduler', type=str, default='exponential_decay')
    parser.add_argument('--decay_rate', type=float, default=0.37293)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_activation', type=str, default='relu')
    parser.add_argument('--output_activation', type=str, default='softmax')
    parser.add_argument('--weight_initialisation', type=str, default='random_uniform')
    parser.add_argument('--loss', type=str, default='categorical_crossentropy')
    parser.add_argument('--timestamp', type=str, default='test')
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--dropout_first', type=float, default=0.08077)
    parser.add_argument('--dropout_second', type=float, default=0.24986)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)

    return parser


def confusion_matrix(testing_data, predictions):
    matrix = np.zeros((7, 7), dtype=int)

    # Compute confusion matrix
    for i, predicted_label in enumerate(predictions):
        expected_label = testing_data.targets[i].argmax(-1)
        matrix[expected_label, predicted_label] += 1

    # Rows are actual labels, columns are predicted labels
    predicted_sums = np.sum(matrix, axis=0)
    actual_sums = np.sum(matrix, axis=1)

    print("Confusion matrix:")
    print(matrix)

    for i in range(7):
        true_positives = matrix[i, i]
        false_positives = predicted_sums[i] - true_positives
        false_negatives = actual_sums[i] - true_positives

        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        f1 = 2 * (precision * recall) / (precision + recall)

        print("Statistics for emotion " + str(i) + ":")
        print("Recall: " + str(recall))
        print("Precision: " + str(precision))
        print("F1: " + str(f1))
        print()


def main():
    parser = get_parser()
    params = parser.parse_args()

    training_data, testing_data, validation_data = load_data(params.data)

    history, model = train_and_report(training_data, validation_data, params)
    plot_accuracy(history)
    plot_loss(history)

    print(testing_data.data[0])
    predictions = model.model.predict(testing_data.data).argmax(-1)

    confusion_matrix(testing_data, predictions)


if __name__ == "__main__":
    main()
