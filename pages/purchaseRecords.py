from dash import html
import dash_bootstrap_components as dbc
import dash
import pandas as pd
import yfinance as yf
import dash_ag_grid as dag

dash.register_page(__name__, path='/purchaseRecords', title='Optimization')

exchange_names = {
    # Bolsas de México
    "MEX": "Bolsa Mexicana de Valores (BMV)",               # Exchange principal en México
    "BIVA": "Bolsa Institucional de Valores (BIVA)",       # Segunda bolsa de valores en México

    # Bolsas de Estados Unidos
    "NYQ": "New York Stock Exchange (NYSE)",                # Bolsa de Nueva York
    "NMS": "Nasdaq Global Select Market (NMS)",                   # Mercado Selecto Global del Nasdaq
    "NAS": "Nasdaq Stock Market (NAS)",                           # Nasdaq en general
    "ASE": "NYSE American (ASE)",                 # Bolsa Americana, anteriormente conocida como AMEX
    "ARCA": "NYSE Arca (ARCA)",                                   # Bolsa de intercambio de valores de NYSE Arca
    "BATS": "Bats Global Markets (BATS)",                         # Bolsa Bats, ahora parte de Cboe Global Markets
    "IEX": "Investors Exchange (IEX)",                            # Bolsa IEX, conocida por su enfoque en la equidad y transparencia en los intercambios
}

def columnsTable(dataFrame, format):
    ''' Function to display efficient frontier results '''
    ''' Input: efficient frontier results '''
    ''' Output: table with visual effects for a better understanding of the data '''
    
    valFormat = lambda x: {'function':"d3.format(',.2f')(params.value)"} if x=='float' else {'function':"d3.format(',.0f')(params.value)"} if x=='int' else {'function':"d3.format(',.0%')(params.value)"} if x=='percentage' else {"function": "d3.format('$,.2f')(params.value)"} if x=='currency' else None if x=='string' else None
    
    columnDefs = [{'field':i, 'headerName':i, 'minWidth': 100, "filter": "agSetColumnFilter", "valueFormatter":valFormat(f)} if f is not None else {'field':i, 'headerName':i, 'minWidth': 100, "filter": "agSetColumnFilter"} for i, f in zip(dataFrame, format)]
    
    grid = dag.AgGrid(
        id="efficientFrontierDataResults",
        rowData=dataFrame.to_dict('records'),
        columnDefs=columnDefs,
        columnSize='responsiveSizeToFit',
        defaultColDef={"sortable": True, "floatingFilter": True},
        className='ag-theme-balham',
    )
    return grid

# Predetermined Stocks for the Optimization
purchaseRecords = pd.read_csv('assets/purchaseRecords.csv')
purchaseRecords['date'] = pd.to_datetime(purchaseRecords['date']).dt.strftime('%Y-%m-%d')
purchaseRecords['market'] = [exchange_names.get(yf.Ticker(i).info.get('exchange'), 'Unknown').split(' ')[-1][1:-1] for i in purchaseRecords['stock']]
purchaseRecords.columns = ['Order ID', 'Date', 'Stock', 'Market', 'Purchase Qty (USD)', 'Echange Rates (MXN)', 'Stock Price (USD)', 'No Stocks']



# MAIN TITLE #
main_title = html.Div("Purchase Records", className='h1 text-center text-md-start')
sec_title = html.Div(className='h3 text-center text-md-start', children="All Time Records", style={'color':'#5a5a5a'})

table = columnsTable(purchaseRecords, ['int', 'string', 'string', 'string', 'currency', 'currency', 'currency', 'float'])

# Define el layout de tu aplicación
layout = html.Div([
    dbc.Container([
        main_title,
        sec_title,
        table
        
    ], style={'margin-top':'50px', 'padding-right':'0','padding-left':'0'})
])