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
stocks_searcher = pd.read_csv('src/assets/stocks_list.csv')
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
timelapse_dict = {'w2':'2 Weeks', 'w4':'1 Month', 'w12':'3 Months', 'w24':'6 Months', 'w48':'1 Year', 'w96':'2 Years', 'w240':'5 Years'}
timelapse_items = [{'label':timelapse_dict[label],'value':label} for label in timelapse_dict]

markets = ['^GSPC', '^MXX']
markets_dict = {mkt:yf.Ticker(mkt).info['longName'] for mkt in markets}
markets_items = [{'label':markets_dict[label], 'value':label} for label in markets_dict]

dropdown_timelapse = om.dropdownMaker('dropdown_timelapse_p2', 'Select Timelapse', timelapse_items, 'w48', '100px')
dropdown_market = om.dropdownMaker('dropdown_market_p2', 'Select Market', markets_items, '^GSPC', '116px')


bgImage = '''url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"%3E%3Cpath d="M73 39c-14.8-9.1-33.4-9.4-48.5-.9S0 62.6 0 80L0 432c0 17.4 9.4 33.4 24.5 41.9s33.7 8.1 48.5-.9L361 297c14.3-8.7 23-24.2 23-41s-8.7-32.2-23-41L73 39z"%3E%3C/path%3E%3C/svg%3E');'''
bgPosition = 'right 8px center'
computeButton = dbc.Button("Compute", outline=True, class_name='text-start', size='sm',id='compute_button', style={'width':'98px','border-radius':'8px', 'font-size':'16px', 'font-width':'100','padding':'5px 8px', 'border':'none', 'color':'rgb(100,100,100)','background-size': '10px', 'background-position':bgPosition, 'background-repeat':'no-repeat'})


plotButtons = html.Div([
        html.Span(
            computeButton
        ),
        
        html.Div([
            html.Div(dropdown_market), html.Div(dropdown_timelapse, style={'margin-left':'5px'})  
        ], className='d-inline-flex')
        
    ], className='d-flex justify-content-between',style={'border-radius':'14px 14px 0 0', 'border':'1px solid rgb(190,187,187)', 'padding':'15px', 'background-color':'rgb(243,244,244)'})

config1 = {
    'staticPlot': False,  # Hace que el gráfico sea estático
    'displayModeBar': False,  # Oculta la barra de herramientas
    'scrollZoom': False,  # Desactiva el zoom con la rueda del mouse
    'editable': False,  # Desactiva la edición de los gráficos
    'responsive':True
}

portfolio_graph = html.Div([
    # Column With Dropdowns
    plotButtons,
    # Column With Graph
    html.Div([
       html.Div(dcc.Graph(id='portfolio_graph_p2', figure=pred_fig, config={'responsive': True}, style={'width':'100%', 'height':'100%', 'background-color':'transparent', 'padding':'14px'}), style={'width':'100%', 'height':'100%', 'background-color':'transparent'}), 
    ], style={'width': '100%', 'height': '50vw','padding':'0', 'border-radius':'0 0 14px 14px', 'border-bottom':'1px solid rgb(190,187,187)', 'border-left':'1px solid rgb(190,187,187)', 'border-right':'1px solid rgb(190,187,187)', 'background-color':'transparent'}),
    
    ], className='text-black', style={'margin-bottom':'50px', 'padding':'0'})

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
        
        subtitle = f"{timelapse_dict[selectedTimelapse]} {markets_dict[market]} Stocks Forecast"
    
        if triggered_id == 'compute_button':
            if market == '^MXX':
                stocks = [i+'.MX' for i in stocks]
            
            figure, table = efficientFrontier(stocks, nWeeks)
            
            return figure, subtitle, table
    
    return dash.no_update