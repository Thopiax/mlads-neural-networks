from model import Model
from data import load_data
import argparse
import os
import httplib2
from datetime import datetime
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

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
        credentials = tools.run_flow(flow, store, flags)
        print('Storing credentials to ' + credential_path)

    return credentials


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
                params.lr,
                params.lrd,
                params.momentum,
                params.epochs,
                params.batch_size,
                loss,
                accuracy
            ]]
        }).execute()


def main():
    parser = argparse.ArgumentParser(
        description='Run model with given arguments.',
        parents=[tools.argparser])

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrd', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    params = parser.parse_args()

    training_data, test_data, validation_data = load_data("data4students.mat")

    model = Model(training_data, test_data, params)
    model.build()
    model.train(epochs=params.epochs, batch_size=params.batch_size)
    loss, accuracy = model.evaluate()

    report_run(params, loss, accuracy)


if __name__ == "__main__":
    main()
