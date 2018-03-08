from model import Model, plot_history
import argparse
from argparse import Namespace
import os
import csv
import httplib2
import numpy as np
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
    if loss == "nan": return

    print("reporting:\n\tparams={}\n\tloss:{}\n\taccuracy".format(params, loss, accuracy))
    with open("results-{}.csv".format(params.timestamp), "a") as csvfile:
        writer = csv.writer(csvfile)
        values = [str(datetime.now()), '| '.join(map(str, params.hidden_layer_neurons)), str(params.loss), str(params.hidden_activation), str(params.output_activation), str(params.weight_initialisation), str(params.epochs), str(params.batch_size), str(params.lr), str(params.lr_decay), str(params.momentum), str(loss), str(accuracy)]

        writer.writerow(values)


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
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_activation', type=str, default='relu')
    parser.add_argument('--output_activation', type=str, default='softmax')
    parser.add_argument('--weight_initialisation', type=str, default='glorot_uniform')
    parser.add_argument('--hidden_layer_neurons', nargs='+', type=int, default=[1128])
    parser.add_argument('--loss', type=str, default='categorical_crossentropy')
    parser.add_argument('--timestamp', type=str)
    parser.add_argument('--early_stopping_patience', type=int, default=10)

    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    training_data, testing_data, validation_data = load_data(params.data)

    history, model = train_and_report(training_data, validation_data, params)
    # plot_history(history)


if __name__ == "__main__":
    main()
