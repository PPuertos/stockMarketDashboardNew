from dash import html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash
import optimizationModule as om
import pandas as pd
from datetime import datetime as dt
import yfinance as yf

dash.register_page(__name__, path='/page2', title='Optimization')

# Predetermined Stocks for the Optimization
portDf = pd.read_csv('src/assets/purchaseRecords.csv')

# MAIN TITLE #
main_title = html.Div("Portfolio Optimization", className='h1 text-center text-md-start')
sec_title = html.Div(id='sec_title_p2', className='h3 text-center text-md-start', children="1 Year S&P 500 Stocks Forecast", style={'color':'#5a5a5a'})

# Defining Searcher
stocks_searcher = pd.read_csv('stocks_list.csv')
stocks_searcher = stocks_searcher[stocks_searcher['status'] == 'Active']
stocks_searcher = stocks_searcher[stocks_searcher['assetType'] == 'Stock']
searcher = [f'{name} - {symbol}' for symbol, name in zip(stocks_searcher['symbol'], stocks_searcher['name'])]

# Defining Stocks Purchases depending on the market
market = '^GSPC'

stocksPurchased = portDf[portDf['market'] == market]['stock']
stocksPurchased = pd.Series([i[:-3] if 'MX' in i else i for i in stocksPurchased]).unique()

# Searching stocks in the Searcher to add them            
preDefStocks = []
for stock in stocksPurchased:
    for i in searcher:
        if stock == i.split(' ')[-1]:
            preDefStocks.append(i)

stocks_dropdown = dcc.Dropdown(searcher,preDefStocks, multi=True, id='selected_stocks_p2', style={'margin-bottom':'30px'})


# PORTFOLIO GRAPH #
# PORTFOLIO GRAPH #
# PORTFOLIO GRAPH #


### NEW FUNCTION ###
def efficientFrontier(stocks, nWeeks):
    start_date = dt.today() - pd.Timedelta(days=14*nWeeks)
    stocks_cp = om.extractingData(stocks, startDate=start_date, closePrices=True)
    stocks_daily_rr = om.dailyReturnRates(stocks_cp)
    stocks_mean_rr, stocks_cov_matrix = om.stocksStatistics(stocks_daily_rr)
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientResults = om.efficientFrontierData(stocks_mean_rr, stocks_cov_matrix, nWeeks, riskFreeRate=0.07)
    mc_sim = om.montecarloSimulation(stocks_mean_rr, stocks_cov_matrix, 5000, nWeeks)
    figure = om.efficientFrontierPlot(efficientResults, maxSR_std, maxSR_returns, minVol_std, minVol_returns, mc_sim, nWeeks)
    table = om.efficientFrontierTable(efficientResults)
    return figure, table

# PREDETERMINED FIGURE
pred_fig, pred_table = efficientFrontier([s.split(' ')[-1] for s in preDefStocks], 48)

# Dropdown menu item for date range
# Time lapse options
timelapse_dict = {'w2':'2 Weeks','w4':'1 Month', 'w12':'3 Months','w24':'6 Months', 'w48':'1 Year', 'w96':'2 Years', 'w240':'5 Years'}
timelapse_id = [i for i in timelapse_dict]
timelapse = [i for i in timelapse_dict.values()]

# Time lapse items
items_timelapse = [{'label':j, 'value':i} for i,j in zip(timelapse_id, timelapse)]
dropdown_timelapse = dbc.Select(id='dropdown_timelapse_p2',options=items_timelapse, value=items_timelapse[4]['value'], style={'margin-left':'10px'})

# Dropdown menu item for market 
# Market Options

mktn = ['^GSPC', '^MXX']
data = {mkt:yf.Ticker(mkt).info for mkt in mktn}
market = [data['^GSPC']['longName'], data['^MXX']['longName']]
market_id = ['^GSPC', '^MXX']

# market items
items_market = [{'label':j,'value':i} for i,j in zip(market_id, market)]
dropdown_market = dbc.Select(id='dropdown_market_p2',options=items_market, value=items_market[0]['value'])

computeButton = dbc.Button("Compute", className="me-1", outline=True, color='success', size='sm',id='compute_button')

plotButtons = dbc.Row([
        dbc.Col(
            computeButton
        ),
        dbc.Col(
            html.Div([
                dropdown_market, dropdown_timelapse
            ], className="d-inline-flex"),  # Alinea los dropdowns en línea
            width="auto",  # Ajusta el tamaño del contenedor de dropdowns
            className="ml-auto"  # Empuja los dropdowns hacia la derecha
        )
    ], justify="between") 

portfolio_graph = dbc.Row([
    # Column With Dropdowns
    plotButtons,
    # Column With Graph
    dbc.Col(dcc.Graph(id='portfolio_graph_p2', figure=pred_fig, config={'responsive': True}, style={'width': '100%', 'height': '50vw'}), width=12),
    
    ], className='text-black', style={'border-radius':'14px', 'margin-bottom':'50px', 'padding':'20px', 'background':'#fafafa', 'border':'1px solid #000'})

### PORTFOLIO EFFICIENT FRONTIER TABLE ###
### PORTFOLIO EFFICIENT FRONTIER TABLE ###
### PORTFOLIO EFFICIENT FRONTIER TABLE ###
efficient_frontier_table = html.Div(id='efficient_frontier_table', children=pred_table)

# Define el layout de tu aplicación
layout = html.Div([
    dbc.Container([
        main_title,
        sec_title,
        stocks_dropdown,
        portfolio_graph,
        efficient_frontier_table
    ], style={'margin-top':'50px', 'padding-right':'0','padding-left':'0'})
])

@callback(
    [Output('portfolio_graph_p2','figure'),
        Output('sec_title_p2','children'),
        Output('efficient_frontier_table','children')],
    [Input('compute_button', 'n_clicks'), Input('dropdown_market_p2','value'), Input('dropdown_timelapse_p2','value'), Input('selected_stocks_p2','value')]
)
def computeEfficientFrontier(computeButton, selectedMarket, selectedTimelapse, selectedStocks):
    ctx = dash.callback_context
    stocks = [i.split(' ')[-1] for i in selectedStocks]
    nWeeks = int(selectedTimelapse[1:])
    market = selectedMarket
    
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        period = lambda x: x[:-1] if x[-1] == 's' else x
        subtitle = f"{period(timelapse_dict[selectedTimelapse])} {data[market]['longName']} Stocks Forecast"
    
        if triggered_id == 'compute_button':
            if market == '^MXX':
                stocks = [i+'.MX' for i in stocks]
            
            figure, table = efficientFrontier(stocks, nWeeks)
            
            return figure, subtitle, table
    
    return dash.no_update