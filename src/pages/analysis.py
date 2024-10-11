from dash import Output, Input, callback_context, html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import ALL
import optimizationModule as om
from dash_extensions import EventListener
import pandas as pd
import dash

# Registrar pagina
dash.register_page(__name__, path='/analysis', title='Home')

# Dataframe with information of all stocks of USA
allStocksDF = pd.read_csv('src/assets/stocks_list.csv')

# SCREENERS
popularStocks, mostActiveSymbols, dayGainerSymbols, symbols, closePrices = om.screeners()

# Recomendation system function for trending ticker's news
# newData is the dictionary with the news by stock, selectedLongNames is de list of longNames of the selected stocks for the news
newData, selectedLongNames = om.recomendationSystemNews(symbols)

# Pandas series with the currency of each stock
currencies = pd.concat([pd.Series([i['currency'] for i in popularStocks['day_gainers']['quotes']], index=[i['symbol'] for i in popularStocks['day_gainers']['quotes']]), pd.Series([i['currency'] for i in popularStocks['most_actives']['quotes']], index=[i['symbol'] for i in popularStocks['most_actives']['quotes']])])

# In this function we update the currencies serie to have unique stocks.
currenciesList = []
for i in closePrices:
    if len(currencies[i]) == 2:
        currenciesList.append(currencies[i].iloc[0])
    else:
        currenciesList.append(currencies[i])
currencies = pd.Series(currenciesList, index=closePrices.columns)

config1 = {
    'staticPlot': False,  # Hace que el gráfico sea estático
    'displayModeBar': False,  # Oculta la barra de herramientas
    'scrollZoom': False,  # Desactiva el zoom con la rueda del mouse
    'editable': False,  # Desactiva la edición de los gráficos
}

# MAIN TITLE #
mainTitle = html.Div("Stocks Overview", className='h1')

# TICKER TAPE #
stocksActualPrice, stocksRR, stocksR, stocksBehaviour = om.tickersCalcs(closePrices.iloc[-2:,:])
tickerTape = om.tickerTapeStocks(stocksActualPrice, stocksRR, stocksR, currencies, symbols)

# POPULAR STOCKS HEATMAP AND FINANCIAL INFO IN GRID #
popularStocksTreeMap = {}
popularStocksDf = {}

# TRENDING TICKERS HEATMAP AND GRID #
for i in popularStocks:
    popularStocksList = popularStocks[i]['quotes']
    popularStocksTreeMap[i], popularStocksDf[i] = om.treeMapAndPopularDf(popularStocksList)

# GRID
tableViewMostActive = om.generalTableFormat(popularStocksDf['most_actives'], ['str', 'str', 'percentage', 'float', 'int', 'int', 'percentage'])
tableViewDayGainers = om.generalTableFormat(popularStocksDf['day_gainers'], ['str', 'str', 'percentage', 'float', 'int', 'int', 'percentage'])

# ROW WITH GRID AND HEATMAP
testRow = dbc.Container([
        dbc.Row([
        dbc.Col("Heatmap View", className='small-text text-center fw-bolder px-0 order-1 order-md-1', width=12, md=6),
        dbc.Col("Table View", className='small-text text-center fw-bolder px-0 order-3 order-md-2', width=12, md=6),
        dbc.Col([
            html.Div([
                html.Div(id="treemap-view-analysis", style={'height':'100%', 'border-radius':'10px'})
                ], style={'background-color':'#fafafa', 'height':'100%', 'border-radius':'16px', 'border':'1px solid black', 'padding':'15px'})
            ], width=12, md=6, style={'height':'100%'}, class_name='px-0 pe-md-2 mb-4 mb-md-0 order-2 order-md-3'),
        dbc.Col([
            html.Div([
                html.Div(id="table-view-analysis", style={'height':'100%', 'border-radius':'10px'})
                ], style={'background-color':'#fafafa', 'height':'100%', 'border-radius':'16px', 'border':'1px solid black', 'padding':'15px'}, className='height-450')
            ], width=12, md=6, class_name='px-0 ps-md-2 order-4 order-md-4')
        ])
    ], fluid=True, style={'margin-bottom':'15px'})


timelapseValues = {'1 Week':7, '2 Weeks':14, '1 Month':30, '2 Months': 60, '3 Months':90, '4 Months':120, '6 Months':30*6, '1 Year':30*12}

popularStocksTreeMapsRow = dbc.Row([
    html.Div(id='trending-tickers-title-analysis', className='h2', style={'margin-bottom':'20px'}),
    html.Div(dcc.Dropdown(['Most Active','Day Gainers'], id='select-view-analysis', placeholder='Select View', clearable=False, searchable=False, value='Most Active', style={'width':'150px','border-radius':'20px', 'font-size':'15px', 'font-width':'100'}), style={'margin-bottom':'5px'}, className='justify-content-end d-flex'),
    testRow,
    html.Div([html.Span(id='timelapse-subtitle-analysis', className='text-small fw-bolder'),dcc.Dropdown([i for i in timelapseValues], id='select-timelapse-analysis', placeholder='Select View', clearable=False, searchable=False, value='1 Month', style={'width':'150px','border-radius':'20px', 'font-size':'15px'})], style={'margin-bottom':'5px'}, className='justify-content-between align-items-end d-flex'),
    dbc.Col(id="stock_boxes_analysis", width=12, class_name='px-0')
], style={'margin-bottom':'50px'})  

# STOCK NEWS #
newsRow = dbc.Row([
    html.Div("Recent News", className='h2', style={'margin-bottom':'20px'}),
    html.Div(id='news_list_analysis', style={'border':'1px solid black', 'background-color':'#fafafa', 'padding':'20px', 'border-radius':'14px'}),
    html.Div(html.Button("View More", id='view_more_button_analysis', n_clicks=0), className='text-center mt-3'),
    dcc.Store(id='n_new_analysis')
], style={'margin-bottom':'50px'})


### APP LAYOUT ###
### APP LAYOUT ###
### APP LAYOUT ###
layout = dbc.Container([
    html.Div(id="output-test-button"),
    tickerTape,
    mainTitle,
    popularStocksTreeMapsRow,
    html.Div(id="stock-analysis"),
    newsRow,
    EventListener(id="listener", events=[{"event": "search-input-click"}])
    ])


### PAGE CALLBACKS ###
### PAGE CALLBACKS ###
### PAGE CALLBACKS ###

@callback(
    Output('output-test-button', 'children'),
    Input('listener', 'n_events'),
    prevent_initial_call=True
)

def searchBar(nClicks):
    if nClicks is not None and nClicks > 0:
        return f"clicked times: {nClicks}"
    else:
        return "Haven't clicked searchbar"

# CALLBACK FOR STOCK SELECTION IN SLIDER #
@callback(
    Output('stock-analysis', 'children'),
    Input({'type': 'stock-button', 'index':ALL}, 'n_clicks')
)

def analysisStockPage(nClicks):
    
    ctx = callback_context
    
    if not any(nClicks):
        None
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        return eval(triggered_id)['index'].split('-')[0]

# CALLBACK FOR SELECTING TYPE OF TRENDING STOCKS, EITHER DAY GAINERS OR MOST ACTIVES #
@callback(
    [Output('treemap-view-analysis', 'children'),
     Output('table-view-analysis', 'children'),
     Output('stock_boxes_analysis', 'children'),
     Output('trending-tickers-title-analysis', 'children'),
     Output('timelapse-subtitle-analysis', 'children')],
    [Input('select-view-analysis', 'value'), Input('select-timelapse-analysis', 'value')]
)
def update_output(viewType, timeLapse):
    timeframe = timelapseValues[timeLapse]
    
    if viewType == 'Most Active':
        treemapView = dcc.Graph(figure=popularStocksTreeMap['most_actives'],config=config1, style={'width': '100%', 'overflow':'visible'})
        tableView = tableViewMostActive
        
        cp = closePrices[mostActiveSymbols].iloc[-timeframe:]
        
        stocksActualPrice, stocksRR, stocksR, stocksBehaviour = om.tickersCalcs(cp)
        slider = om.container_with_slider(mostActiveSymbols, stocksActualPrice, stocksRR, stocksBehaviour, 'mostActive')
        title = f"Trending Tickers: {popularStocks['most_actives']['title']}"
        
    elif viewType == 'Day Gainers':
        treemapView = dcc.Graph(figure=popularStocksTreeMap['day_gainers'],config=config1, style={'width': '100%', 'overflow':'visible'})
        tableView = tableViewDayGainers
        
        cp = closePrices[dayGainerSymbols].iloc[-timeframe:,:]
        stocksActualPrice, stocksRR, stocksR, stocksBehaviour = om.tickersCalcs(cp)
        slider = om.container_with_slider(dayGainerSymbols, stocksActualPrice, stocksRR, stocksBehaviour, 'dayGainers')
          
        title = f"Trending Tickers: {popularStocks['day_gainers']['title']}"
        
    return treemapView, tableView, slider, title, f"Tickers Info: {viewType}"

# CALLBACK FOR NEWS ALGORITHM #
@callback(
    [
        # News elements
        Output('news_list_analysis', 'children'),
        # Stored News
        Output('n_new_analysis', 'data'),
        Output('view_more_button_analysis', 'style')
     ],
    [
        # Stored News
        Input('n_new_analysis', 'data'),
        # View more button
        Input('view_more_button_analysis', 'n_clicks')
     ]
)

def newUpdatedAnalysis(data, nClicks):
    if data:
        if nClicks > 0:
            childrens = data[0]
            headlines = data[1]
            nNews = data[2]            
            newsCounter = data[3]
            
            for stock in newData:
                counter = 0
                
                for new in newData[stock]:
                    if counter == 2:
                        counter = 0
                        break
                    
                    if 'https' in new['image']:
                        if new['headline'] not in headlines:
                            childrens.append(om.newCreation(new, selectedLongNames[stock]))
                            headlines.append(new['headline'])
                            
                            counter += 1
                            newsCounter += 1
            
            if newsCounter == nNews:
                styleButton = {'display':'none'}
            else:
                styleButton = {'display':'inline'}
            
            return childrens, [childrens, headlines, nNews, newsCounter], styleButton
            
            
    else:
        # Creating new list of stock news
        childrens = []
        headlines = []
        
        newsCounter = 0
        # Cicle for every stock
        for stock in newData:
            # Starting counter of news
            counter = 0
            for new in newData[stock]:
                # We want two news per stock
                if counter == 2:
                    break
                
                # If image exists
                if 'https' in new['image']:
                    # Add new
                    childrens.append(om.newCreation(new, selectedLongNames[stock]))
                    headlines.append(new['headline'])
                    counter += 1
                    newsCounter += 1
        
        if newsCounter == 0:
            print("No available News")
        
        nNews = 0
        for i in newData:
            for new in newData[i]:
                if 'https' in new['image']:
                    nNews += 1
                
        
        return childrens, [childrens, headlines, nNews, newsCounter], {'display':'inline'}