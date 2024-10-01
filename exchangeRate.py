import requests
import json
from datetime import datetime
import pandas as pd

# Reemplaza 'TU_TOKEN' con tu token de acceso de Banxico
token = 'a2ec0793c85a491fad39859267092512d7f2badd31b205f079432f0f41d5c27f'
url = 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43718/datos/oportuno'
JSON_FILE = 'exchange_rate_dollar.json'


def exchangeRateUsd():
    ''' Function to know the actual price of the dollar'''
    headers = {'Bmx-Token': token}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        exchangeRate = data['bmx']['series'][0]['datos'][0]['dato']
        date = pd.to_datetime(data['bmx']['series'][0]['datos'][0]['fecha'], format='%d/%m/%Y').strftime('%Y-%m-%d')
        return {'date':date, 'exchangeRate':exchangeRate}
    else:
        print("Unable to access exchange rates data")
        return None

def loadSavedER():
    """ Opens Exchange rate Json (if it already exists) """
    try:
        with open(JSON_FILE, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def saveDataER(exchangeRateJSON):
    """ Function to save the exchange rate data in a Json """
    with open(JSON_FILE, 'w') as file:
        json.dump(exchangeRateJSON, file)
        
def ExchangeUsdToMxn():
    ''' Main function of exchange rate of USD in MXN. Validates date and whether use saved file or consults API '''
    savedData = loadSavedER()
    hoy = datetime.now().strftime('%Y-%m-%d')
    
    if savedData and savedData['date'] == hoy:
        return savedData['exchangeRate']
    else:
        newData = exchangeRateUsd()
        if newData:
            saveDataER(newData)
            return newData['exchangeRate']