from dash import Dash, html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash
import optimizationModule as om
import pandas as pd
from datetime import datetime as dt
import yfinance as yf
from plotly import graph_objects as go
import numpy as np
import optimizationModule as om

dash.register_page(__name__, path='/page1', title='Portfolio')

# MAIN TITLE #
# MAIN TITLE #
# MAIN TITLE #

main_title = html.Div(id='main_title', className="h1 text-center text-md-start")
secondary_title = html.Div(id='sec_title', className="h3 text-center text-md-start", style={'color':'#5a5a5a'})

# PORTFOLIO GRAPH #
# PORTFOLIO GRAPH #
# PORTFOLIO GRAPH #

# Dropdown menu item for date range
# Time lapse options
timelapse_dict = {'15 Days':15,'1 Month':30, '3 Months':90,'6 Months':180, '1 Year':252, 'All Time':'All Time'}
timelapse_label = [i for i in timelapse_dict]
timelapse_value = [i for i in timelapse_dict.values()]

## CONFIGURATION FOR FIGURES ##
config1 = {
    'staticPlot': False,  # Hace que el gráfico sea estático
    'displayModeBar': False,  # Oculta la barra de herramientas
    'scrollZoom': False,  # Desactiva el zoom con la rueda del mouse
    'editable': False,  # Desactiva la edición de los gráficos
    'responsive':True
}

# Dropdown menu item for market 
# Market Options

market_id = ['^GSPC', '^MXX'] 
market = ['S&P 500', 'IPC MEXICO']

### NEW DROPDOWN PROPOSITION
items_market = [{'label':i, 'value':j} for i,j in zip(market, market_id)]
dropdown_market = dbc.Select(id='dropdown_market_p1', placeholder='Select Market', options=items_market, value='^GSPC', style={'width':'auto','border-radius':'20px', 'font-size':'15px', 'font-width':'100', 'background-color':'transparent', 'border':None})

items_timelapse = [{'label':i, 'value':j} for i,j in zip(timelapse_label, timelapse_value)]

dropdown_timelapse = om.dropdownMaker('dropdown_timelapse_p1', 'Select Timelapse', items_timelapse, 'All Time','98px')
dropdown_market = om.dropdownMaker('dropdown_market_p1', 'Select Market', items_market, '^GSPC', '114px')

portfolio_graph = dbc.Row([
    # Column With Dropdowns
    dbc.Col([
        html.Span(dropdown_market, style={'display':'inline-block', 'margin-right':'5px'}),
        html.Span(dropdown_timelapse, style={'display':'inline-block'})
        ], class_name='text-end', width=12),
    # Column With Data
    dbc.Col([html.Span(id='portfolio_actual_worth', className='text-center h4 mb-0', style={'font-weight':'100', 'margin-right':'10px'}), 
                html.Span(id='portfolio_return_rate', className='text-center align-center small-text'), 
                ], style={'margin-bottom':'20px', 'display':'flex', 'align-items':'center'}, width=12),
    # Column With Graph
    dbc.Col(dcc.Graph(id='portfolio_graph', config=config1, style={'width': '100%', 'height': '50vw'}), width=12),
    
    ], className='text-black', style={'border-radius':'14px', 'margin-bottom':'50px', 'padding':'20px', 'background':'#fafafa', 'border':'1px solid #000'})

### PORTFOLIO INFO ###
### PORTFOLIO INFO ###
### PORTFOLIO INFO ###
### PORTFOLIO INFO ###

portfolio_info = dbc.Row([
    dbc.Col([
        html.Div([
            html.Div("Hist. Inv.", className='h3'),
            html.Div(id='market_investment', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
    dbc.Col([
        html.Div([
            html.Div("Period Inv.", className='h3'),
            html.Div(id='market_period_investment', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
    dbc.Col([
        html.Div([
            html.Div("Returns", className='h3'),
            html.Div(id='portfolio_returns', className='h5'),
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
    dbc.Col([
        html.Div([
            html.Div("Daily VaR", className='h3'),
            html.Div(id='VaR', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
    dbc.Col([
        html.Div([
            html.Div("Daily CVaR", className='h3'),
            html.Div(id='CVaR', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
    dbc.Col([
        html.Div([
            html.Div("Volatility", className='h3'),
            html.Div(id='portfolio_risk', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
    dbc.Col([
        html.Div([
            html.Div("Alpha", className='h3'),
            html.Div(id='portfolio_alpha', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
            
    dbc.Col([
        html.Div([
            html.Div("Beta", className='h3'),
            html.Div(id='portfolio_beta', className='h5'),  
        ])
    ], width=12, sm=6, md=4, lg=3, class_name='info-container'),
    
], class_name='text-center', id='info-row', style={'margin-bottom':'50px'})


### STOCK AND MARKET SLIDERS ROW ###
### STOCK AND MARKET SLIDERS ROW ###
### STOCK AND MARKET SLIDERS ROW ###
### STOCK AND MARKET SLIDERS ROW ###

### PORTFOLIO STOCKS AN MARKETS ROW ###  
stock_and_market_row = dbc.Row([
    # Stocks Title
    dbc.Col([
        html.Div(id='market_stocks_page1', className='h2 text-center', style={'font-size':'35px', 'font-weight':'300','margin-bottom':'30px'})
    ]),
    # Stocks Slider
    dbc.Col(id='stock_boxes_page1', width=12, class_name='px-0', style={'margin-bottom':'50px'}),
    # Markets Title
    dbc.Col([
        html.Div("Markets", className='h2 text-center', style={'font-size':'35px', 'font-weight':'300','margin-bottom':'30px'})
    ]),
    dbc.Col(id="market_boxes_page1", width=12, class_name='px-0')
], style={'margin-bottom':'50px'})


# APP LAYOUT #
layout = html.Div([
    dbc.Container([
        main_title,
        secondary_title,
        portfolio_graph,
        portfolio_info,
        stock_and_market_row
    ], style={'margin-top':'50px', 'padding-right':'0','padding-left':'0'}, fluid=True)
])

@callback(
    [
        Output('portfolio_graph','figure'),
        Output('portfolio_actual_worth', 'children'),
        Output('portfolio_return_rate','children'),
        Output('portfolio_return_rate', 'style'),
        Output('portfolio_returns','children'),
        Output('portfolio_returns','style'),
        Output('main_title', 'children'),
        Output('sec_title', 'children'),
        Output('market_investment', 'children'),
        Output('market_period_investment', 'children'),
        Output('VaR','children'),
        Output('CVaR','children'),
        Output('portfolio_risk','children'),
        Output('portfolio_alpha','children'),
        Output('portfolio_beta','children'),
        Output('market_stocks_page1', 'children'),
        Output('stock_boxes_page1', 'children'),
        Output('market_boxes_page1','children')
        ],
    [Input('dropdown_market_p1', 'value'), Input('dropdown_timelapse_p1', 'value'), Input('portfolio_main_data', 'data')]
)
def market_portoflio_graph(selected_market, selected_timelapse, data):
    ## THIS PART IS FOR VARIABLE OUTPUTS ##
    ## THIS PART IS FOR VARIABLE OUTPUTS ##
    ## THIS PART IS FOR VARIABLE OUTPUTS ##

    # Conditionals for formatting
    # String values
    format_returns = lambda returns: f"+${returns:,.2f}" if returns > 0 else f"-${-returns:,.2f}"
    # Format for return rate
    format_return_rate = lambda returns: f"+{returns*100:,.2f}%" if returns > 0 else f"-{-returns*100:,.2f}%"
    # Apply color styling based on the value
    style_colors = lambda returns: ['rgba(72, 200, 0, 1)', 'rgba(72, 200, 0, .3)'] if returns > 0 else ['rgba(255, 0, 0, 1)', 'rgba(255, 0, 0,.3)']
    
    # Portfolio Actual Worth
    portfolio_actual_worth = data[selected_market]['Other Results']['actual_worth']
    # Portfolio Total Investment
    portfolio_total_inv = data[selected_market]['Other Results']['total_investment']
    # Market Purchase Records
    market_records = pd.DataFrame(data['records_by_market'][selected_market])
 
    if selected_timelapse == 'All Time':
        # Portfolio Historical Evolution
        data_range = pd.Series(data[selected_market]['behaviour'])
        # Inversion in a period, in this case historically
        period_inv = portfolio_total_inv
        # Portfolio Returns
        portfolio_returns = portfolio_actual_worth - period_inv
        # Portfolio Return Rate
        portfolio_return_rate = data[selected_market]['Other Results']['return_rate']
        
        # Market daily returns (data frame with stocks returns)
        market_daily_r_rates = pd.DataFrame(data[selected_market]['stocks_daily_return_rates'])
        # Market daily return rates
        market_rr = market_daily_r_rates[selected_market]
        # Stocks daily return rates without its market
        market_daily_r_rates = market_daily_r_rates.drop(columns=selected_market)
        
    else:
        # Selecting Range (Portfolio Evolution in a determined period)
        data_range = pd.Series(data[selected_market]['behaviour']).iloc[-int(selected_timelapse):]
        # FOR ADJUSTED INITIAL VALUE
        # Investment made in that period
        period_inv = sum([inv for date, inv in zip(market_records.index, market_records['qty_bought_usd']) if date in data_range.index])
        # Initial Value (Adjusted). First date value in the period + investments in the period
        adj_initial_value = data_range.iloc[0] + period_inv
        # Potfolio Returns
        portfolio_returns = portfolio_actual_worth - adj_initial_value
        # Return Rate
        portfolio_return_rate = portfolio_returns/adj_initial_value
        
        # Market daily return rates (data frame with stocks returns) (This one is for VaR and CVaR calculation)
        market_daily_r_rates = pd.DataFrame(data[selected_market]['stocks_daily_return_rates']).iloc[-int(selected_timelapse):,:]
        # Market daily Return rates
        market_rr = market_daily_r_rates[selected_market]
        # Stocks daily return rates without its markets
        market_daily_r_rates = market_daily_r_rates.drop(columns=selected_market)
    
    
    # FOR PORTFOLIO RISK #
    # FOR PORTFOLIO RISK #
    # FOR PORTFOLIO RISK #
    
    # Investment by stock, from the selected market
    stocksInvestments = pd.Series(data['investment_by_market'][selected_market])
    # Total investment from the selected market
    totalInvestment = stocksInvestments.sum()
    # Portfolio Weights
    stocksWeights = stocksInvestments/totalInvestment
    # Portfolio mean returns and covariance matrix
    meanRet, covMat = om.stocksStatistics(market_daily_r_rates)
    # Number of weeks. We will convert the volatility
    nWeeks = len(market_daily_r_rates)/5
    # Portfolio volatility
    portfolioV = om.portfolioStd(stocksWeights, meanRet, covMat, nWeeks)
    
    
    # String Formatting
    # Format for returns
    portfolio_returns = format_returns(portfolio_returns)
    # Color format for returns and return rates
    color_a, color_b = style_colors(portfolio_return_rate)
    # Format for return rates
    portfolio_return_rate = format_return_rate(portfolio_return_rate)
    # Styles to apply to portfolio return rate
    return_rate_style = {'color':color_a, 'border':f'1px solid {color_a}', 'border-radius':'10px', 'padding':'3px'}
    # Styles to apply to potfolio return
    returns_style = {'color':color_a}
    
    # Portfolio Graph
    
    figure = om.portfolioWorthPlot(data_range, color_b, color_a)
    
    # VaR and CVaR
    # Dataframe with daily return rates of each stock of the market, taking into account their weights, which are the total investment in the stock divided by the total investment in the market
    # Also taking into account the selected period
    stocks_weights = pd.Series(data['stock_weights'][selected_market])
    daily_r = market_daily_r_rates @ stocks_weights
    
    # Value at Risk 95% Confidence
    VaR_95 = np.percentile(daily_r, 5)
    # Conditional Value at Risk 95% Confidence
    CVaR_95 = daily_r[daily_r.values <= VaR_95].mean()
    
    # FOR BETA AND ALPHA
    beta_alpha = om.ols(market_rr, daily_r)
    
    # FOR SLIDERS STOCKS
    # Stocks of the market
    market_stocks = market_records['stock'].unique()
    
    # Returns
    # Investment in stock
    stock_returns = []
    stock_ret_rates = []
    stock_behaviour = []
    for i in market_stocks:
        if selected_timelapse == 'All Time':
            start_price = data['investment_by_market'][selected_market][i]
            stock_beh = pd.Series(data[selected_market][i]['behaviour'])
        else:
            ## PARA SABER CUANTO HEMOS INVERTIDO EN EL STOCK
            # Comportamiento histórico del stock "n" años atrás
            stock_beh = pd.Series(data[selected_market][i]['behaviour']).iloc[-int(selected_timelapse):]
            # Records de las inversiones hechas en el stock
            stock_inv_record = market_records[market_records['stock'] == i]
            # Inversion hecha a "n" años atrás
            inv_made = sum([inv for date, inv in zip(stock_inv_record.index, stock_inv_record['qty_bought_usd']) if date in stock_beh.index])
            start_price = stock_beh.iloc[0] + inv_made
        
        ret = stock_beh.iloc[-1] - start_price
        ret_rate = ret/start_price
        stock_returns.append(ret)
        stock_ret_rates.append(ret_rate)
        stock_behaviour.append(stock_beh)
    
    # FOR SLIDERS MARKETS
    markets = data['records_by_market'].keys()
    
    market_ret_rates = []
    market_behaviour = []
    actual_prices = []
    ## SLIDER MARKETS ###
    ## SLIDER MARKETS ###
    ## SLIDER MARKETS ###
    for i in markets:
        if selected_timelapse == 'All Time':
            mkt_beh = pd.Series(data[i]['close_prices'])
            st_price = mkt_beh.iloc[0]
            act_price = mkt_beh.iloc[-1]
        else:
            mkt_beh = pd.Series(data[i]['close_prices'])[-int(selected_timelapse):]
            st_price = mkt_beh.iloc[0]
            act_price = mkt_beh.iloc[-1]
        
        ret = act_price - st_price
        ret_rate = ret/st_price
        market_ret_rates.append(ret_rate)
        market_behaviour.append(mkt_beh)
        actual_prices.append(act_price)
        
    
    subtitleFormat = lambda x: f"{x} Returns" if x != 'All Time' else f"{x} Returns (≈ {round(len(data[selected_market]['behaviour'])/5/4/12)} years)"
    
    def selected_timelapse_name(selected_timelapse, timelapse_dict):
        for label, value in timelapse_dict.items():
            if selected_timelapse == 'All Time':
                return selected_timelapse
            else:
                if value != 'All Time':
                    if int(selected_timelapse) == value:
                        return label
        
    return figure, f"${portfolio_actual_worth:,.2f} USD", portfolio_return_rate, return_rate_style, portfolio_returns, returns_style, f"{data[selected_market]['longName']} Portfolio", subtitleFormat(selected_timelapse_name(selected_timelapse, timelapse_dict)), f"${portfolio_total_inv:,.2f} USD", f"${period_inv:,.2f} USD", f"${-VaR_95*portfolio_total_inv:,.2f}", f"${-CVaR_95*portfolio_total_inv:,.2f}", f"{portfolioV*100:,.2f}%",f"{beta_alpha['alpha']*100:,.2f}%", f"{beta_alpha['beta']*100:,.2f}%", f"{data[selected_market]['longName']} Stocks", om.container_with_slider(market_stocks, stock_returns, stock_ret_rates, stock_behaviour, 'stocks'), om.container_with_slider(markets, actual_prices, market_ret_rates, market_behaviour, 'markets')

'''
OUTPUT
main_title
sec_title

portfolio_actual_worth
portfolio_return_rate (for children and for style)
portfolio_graph

market_investment
market_period_investment
portfolio_returns (for children and for style)
VaR
CVaR
portfolio_risk
portfolio_alpha
portfolio_beta

market_stocks
stock_boxes
market_boxes

INPUT
timelapse_id
market_id
selected_market
'''

'''
[
        Output('portfolio_graph','figure'),
        Output('portfolio_actual_worth', 'children'),
        Output('portfolio_return_rate','children'),
        Output('portfolio_return_rate', 'style'),
        Output('portfolio_returns','children'),
        Output('portfolio_returns','style'),
        Output('main_title', 'children'),
        Output('sec_title', 'children'),
        Output('market_investment', 'children'),
        Output('market_period_investment', 'children'),
        Output('VaR','children'),
        Output('CVaR','children'),
        Output('portfolio_risk','children'),
        Output('portfolio_alpha','children'),
        Output('portfolio_beta','children'),
        Output('market_stocks', 'children'),
        Output('stock_boxes', 'children'),
        Output('market_boxes','children')
        ]
'''