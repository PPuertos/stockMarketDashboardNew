from dash import Dash, html, dcc, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash
import optimizationModule as om
import pandas as pd
from datetime import datetime as dt
from yahooquery import Screener
import random

dash.register_page(__name__, path='/', title='Home')

# Purchase Records
purchaseRecords = pd.read_csv('src/assets/purchaseRecords.csv')
portfolioStocks = purchaseRecords.groupby(['stock']).agg({'qty_bought_usd':'sum'}).reset_index()
portfolioWeights = pd.Series(data=[i/portfolioStocks['qty_bought_usd'].sum() for i in portfolioStocks['qty_bought_usd']], index=portfolioStocks['stock'])

stocksList = [i for i in purchaseRecords['stock'].unique() if '.MX' not in i] + ['^GSPC']
# Consulting news related to the stocks in the portfolio
newsDict = om.NewsForAllActives(stocksList)
screener = Screener()

# Consulting screeners of todays popular stocks in the market
popularStocks = screener.get_screeners(['most_actives', 'day_gainers'])

### PORTFOLIO MAIN INFO ###
### PORTFOLIO MAIN INFO ###
### PORTFOLIO MAIN INFO ###
portfolio_combined_balances_row = dbc.Row([
            dbc.Col("Wallet", className='h2 text-black text-start', width=12),
            dbc.Col(id='portfolio_combined_balances_home', className='h3 text-black text-start', width=12),
        ])

portfolio_main_row = dbc.Col([
            portfolio_combined_balances_row,
            html.Div(html.A("View History ->", className='p-0 bg-transparent border-0 text-black', style={'text-decoration':None}, href='/purchaseRecords'), className='text-end'),
        ], style={'background':'#fafafa', 'border':'1px solid black', 'border-radius':'12px', 'padding':'20px 50px', 'margin-bottom':'50px'})

### PORTFOLIO OTHER VALUES ###
### PORTFOLIO OTHER VALUES ###
### PORTFOLIO OTHER VALUES ###
{'font-size':'20px', 'font-weight':600, 'padding':'5px 15px'}
portfolio_values = dbc.Row([
    html.Div("Portfolio Performance", className='h2 text-start', style={'margin-bottom':'20px'}),
    dbc.Col(html.Div([html.Span(id='risk_score', className='h5 justify-center'), html.Div("Risk Score", className='h4', style={'color':'rgb(120,120,120)'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'height':'100%'}, className='p-3 d-flex flex-column align-items-center'), width=12, lg=3, sm=6, class_name='mb-4'),
    dbc.Col(html.Div([html.Div(id='total_invested_home', style={'font-size':'20px', 'font-weight':600, 'padding':'5px 15px'}, className='h5'), html.Div("Investment", className='h4', style={'color':'rgb(120,120,120)'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black','height':'100%'}, className='p-3'), width=12, lg=3, sm=6, className='mb-4'),
    dbc.Col(html.Div([html.Div(id='portfolio_returns_home', style={'font-size':'20px', 'font-weight':600, 'padding':'5px 15px'}, className='h5'), html.Div("Returns", className='h4', style={'color':'rgb(120,120,120)'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'height':'100%'}, className='p-3'), width=12, lg=3, sm=6, className='mb-4'),
    dbc.Col(html.Div([html.Div(id='profit_or_loss_home', style={'font-size':'20px', 'font-weight':600, 'padding':'5px 15px'}, className='h5'), html.Div("Profit/Loss", className='h4', style={'color':'rgb(120,120,120)'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'height':'100%'}, className='p-3'), width=12, lg=3, sm=6, className='mb-4'),
], class_name='text-center')

### PORTFOLIO PLOTS ROW ###
### PORTFOLIO PLOTS ROW ###
### PORTFOLIO PLOTS ROW ###

config1 = {
    'staticPlot': False,  # Hace que el gráfico sea estático
    'displayModeBar': False,  # Oculta la barra de herramientas
    'scrollZoom': False,  # Desactiva el zoom con la rueda del mouse
    'editable': False,  # Desactiva la edición de los gráficos
}

config2 = {
    'staticPlot': True,  # Hace que el gráfico sea estático
    'displayModeBar': False,  # Oculta la barra de herramientas
    'scrollZoom': False,  # Desactiva el zoom con la rueda del mouse
    'editable': True,  # Desactiva la edición de los gráficos
}

### PALETA DE COLORES ###
### PALETA DE COLORES ###
### PALETA DE COLORES ###
# Color base que diste
base_color = "#A4A091"
# Generar tonos desde el color base, grises y plateados
colors_base = om.generate_shades(base_color, 8)
colors_gray = om.generate_shades("#808080", 8)
colors_silver = om.generate_shades("#C0C0C0",8)  

# Intercalar las tres listas de colores
palettelist = [colors_base, colors_gray, colors_silver]

palette = []
for _ in range(8):
    for j in range(3):
        palette.append(random.choice(palettelist[j]))

goBackButton = dbc.Button("<- Go Back",id='go_back_button_home', className="m-0 text-black", style={'border':'0px solid black', 'background-color':'transparent', 'padding':'3px 0 0 8px'}, size='sm')
pieChartTitle = html.Div(id="pie_chart_title",className="text-center h5",style={'font-weight':'600', 'margin-bottom':'15px'})
returnChartTitle = html.Div(id="return_chart_title",className="text-center h5",style={'font-weight':'600', 'margin-bottom':'15px'})

plotsRow = dbc.Row([
    dbc.Col(html.Div([pieChartTitle, dcc.Graph(id='selecting_stock_home', config=config1, style={'width': '100%'}), goBackButton], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'padding':'15px'}), width=12, lg=6, className='mb-4 mb-lg-0'),
    dbc.Col(html.Div([returnChartTitle, dcc.Graph(id='six_month_returns', config=config2, style={'width': '100%'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'height':'100%', 'padding':'15px'}), width=12, lg=6)
], style={'margin-bottom':'50px'})

### STOCK NEWS ###
### STOCK NEWS ###
### STOCK NEWS ###
def newCreation(newData, activeName):
    today = dt.today()
    publishDateFunction = lambda x: f"{(today - x).total_seconds()/3600:.0f} hours ago" if (today - x).total_seconds()/3600 <= 24 else ("yesterday" if (today - x).total_seconds()/(3600*24) < 2 else (f"{(today - x).total_seconds()/(3600*24):.0f} days ago" if (today - x).total_seconds()/(3600*24) < 7 else x.strftime('%Y-%m-%d')))
    newDate = dt.fromtimestamp(newData['datetime'])
    newDate = publishDateFunction(newDate)
    newHeadLine = newData['headline']
    newImage = newData['image']
    newSource = newData['source']
    newSummary = newData['summary'].split('. ')[0] + '.' if len(newData['summary'].split('. ')) > 1 else newData['summary'].split('. ')[0]
    newUrl = newData['url']
    
    new = html.Div([
        dbc.Row([
            # Columna de la imagen
            dbc.Col(
                html.Div(
                    html.Img(src=newImage, style={'width': '100%', 'border-radius': '10px', 'height': '100%', 'object-fit': 'cover'}),
                    style={'width': '100%', 'overflow': 'hidden'},
                    className='height-for-image-new d-flex justify-content-center align-items-center p-0 mb-3 mb-lg-0'  # Centra la imagen verticalmente
                ), 
                width=12, xxl=3, xl=4, lg=5, 
                className='d-flex align-items-center'  # Alineación vertical de la columna de imagen
            ),
            # Columna de los textos
            dbc.Col([
                html.Div(f"Related to: {activeName}", className='h4 news-stock', style={'font-weight': '800'}),
                html.Div(newHeadLine, className='h3 mb-2 news-headline text-start', style={'font-weight': '700', 'font-size': '24px'}),
                html.Div(newSummary, className='small-text', style={'text-decoration': None, 'color': 'black'}),
                html.Div(f"{newSource} • {newDate}", style={'font-size': '12px', 'margin-top': '10px', 'text-decoration': None, 'color': 'black'}, className='mt-auto')
            ], 
                width=12, xxl=9, xl=8, lg=7, 
                class_name='d-flex flex-column justify-content-start'  # Alineación del texto al inicio de la columna
            )
        ], 
        className='align-items-stretch'  # Alinea las columnas según la altura de la más alta
    )], style={'padding': '10px', 'border-radius': '12px', 'border': '1px solid black', 'background-color':'#fff'}, className='mb-4')

    new = html.A(new, href=newUrl, target="_blank", className='news-container')

    return new
         
newsRow = dbc.Row([
    html.Div("Recent News", className='h2', style={'margin-bottom':'20px'}),
    html.Div(id='news_list_home', style={'border':'1px solid black', 'background-color':'#fafafa', 'padding':'20px', 'border-radius':'14px'}),
    html.Div(html.Button("View More", id='view_more_button_home', n_clicks=0), className='text-center mt-3'),
    dcc.Store(id='n_new_home')
], style={'margin-bottom':'50px'})



### Popular Stocks Tree Map ###
### Popular Stocks Tree Map ###
### Popular Stocks Tree Map ###
### Popular Stocks Tree Map ###
popularStocksTreeMap = {}
popularStocksDf = {}

for i in popularStocks:
    popularStocksList = popularStocks[i]['quotes']
    popularStocksTreeMap[i], popularStocksDf[i] = om.treeMapAndPopularDf(popularStocksList)


popularStocksTreeMapsRow = dbc.Row([
    html.Div("Popular Stocks", className='h2', style={'margin-bottom':'20px'}),
    dbc.Col(html.Div([html.Div(f"Stocks: {popularStocks['most_actives']['title']}", className='h5 text-center', style={'font-weight':'600'}),dcc.Graph(figure=popularStocksTreeMap['most_actives'],config=config1, style={'width': '100%'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'padding':'15px', 'overflow':'visible'}), width=12, lg=6, className='mb-4 mb-lg-0', style={'overflow':'visible'}),
    dbc.Col(html.Div([html.Div(f"Stocks: {popularStocks['day_gainers']['title']}", className='h5 text-center', style={'font-weight':'600'}), dcc.Graph(figure=popularStocksTreeMap['day_gainers'],config=config1, style={'width': '100%'})], style={'background-color':'#fafafa', 'border-radius':'10px', 'border':'1px solid black', 'padding':'15px'}), width=12, lg=6)
], style={'margin-bottom':'50px'})
    
    

layout = dbc.Container([
    html.Div("Welcome Home", className='h1'),
    portfolio_main_row,
    portfolio_values,
    plotsRow,
    popularStocksTreeMapsRow,
    newsRow,
    dcc.Store(id='store_pie_query')
])

@callback([
    Output('portfolio_combined_balances_home', 'children'),
    Output('risk_score', 'children'),
    Output('risk_score', 'style'),
    Output('total_invested_home', 'children'),
    Output('profit_or_loss_home', 'children'),
    Output('portfolio_returns_home', 'children')
    ],
    Input('portfolio_main_data', 'data')
)

def PortfolioMainData(data):
    if data == None:
        return "Nan", "Nan", "Nan"
    else:
        # All portfolio stock return rates
        stocksRetRates = pd.concat([pd.DataFrame(data['^GSPC']['stocks_daily_return_rates']).drop(columns='^GSPC').iloc[-60:,:], pd.DataFrame(data['^MXX']['stocks_daily_return_rates']).drop(columns='^MXX').iloc[-60:,:]], axis=1)
        # Portfolio Investment
        stocksInvestments = pd.concat([pd.Series(data['investment_by_market']['^GSPC']), pd.Series(data['investment_by_market']['^MXX'])], axis=0)
        totalInvestment = stocksInvestments.sum()
        # Portfolio Weights
        stocksWeights = stocksInvestments/stocksInvestments.sum()
        
        movVol = []
        for i in range(30):
            query = stocksRetRates.iloc[i:i+30,:]
            meanRet, covMat = om.stocksStatistics(query)
            portfolioV = om.portfolioStd(stocksWeights, meanRet, covMat, 1/5)
            movVol.append(portfolioV)
        portfolioVol = np.mean(movVol)
        riskScore, riskColor = om.riskScore(portfolioVol)
        
        newColor = ''
        for i in riskColor.split(' ')[:-1]:
            newColor += i
        newColor += '.4)'
        
        portfolioActualWorth = data['^GSPC']['Other Results']['actual_worth'] + data['^MXX']['Other Results']['actual_worth']
        returns = portfolioActualWorth - totalInvestment
        returnRate = portfolioActualWorth/totalInvestment - 1
        return f"${portfolioActualWorth:,.2f} USD", riskScore, {'font-size':'20px', 'font-weight':600, 'padding':'5px 15px', 'border':f'3px solid {riskColor}', 'border-radius':'10px', 'color':f'{riskColor}', 'background-color':newColor}, f"${totalInvestment/1000:,.0f}k USD", f"{returnRate*100:,.2f}%", f"${returns/1000:,.0f}k USD"

@callback(
    [Output('selecting_stock_home', 'figure'),
     Output('store_pie_query', 'data'),
     Output('six_month_returns','figure'),
     Output('pie_chart_title', 'children'),
     Output('return_chart_title', 'children')],
    [Input('selecting_stock_home', 'clickData'),
     Input('go_back_button_home','n_clicks'),
     Input('store_pie_query', 'data'),
     Input('portfolio_main_data','data')]
)

def ReturnsGraphUpdate(clickData, goBack, pieQuery, data):
    if data == None:
        return "Nan", "Nan", "Nan"
    else:
        ctx = dash.callback_context
        
        # DEFINING INVESTMENTS DATAFRAME
        sp500Inv = pd.DataFrame({'Stocks':data['investment_by_market']['^GSPC'].keys(), 'Investment':data['investment_by_market']['^GSPC'].values(), 'Market':'^GSPC'})
        MXInv = pd.DataFrame({'Stocks':data['investment_by_market']['^MXX'].keys(), 'Investment':data['investment_by_market']['^MXX'].values(), 'Market':'^MXX'})
        investmentsDf = pd.concat([sp500Inv, MXInv], axis=0)
        invByMarket = investmentsDf.groupby('Market').agg({'Investment':'sum'}).reset_index()
        # If the function is triggered
        
        if ctx.triggered:
            # Checking which was the triggered item
            triggeredItem = ctx.triggered[0]['prop_id'].split('.')[0]
            # We validate that the triggered item is not the store_pie_query, that way we can ensure that the triggered item is the selected area in the pie chart
            if triggeredItem != 'store_pie_query':
                invDf, selection = pieQuery
                invDf = pd.DataFrame(invDf)
                
                # Validating if the triggered item is go_back_button_home
                if triggeredItem == 'go_back_button_home':
                    if selection in ['^GSPC', '^MXX']:
                        time = 6
                        tableIncomes = pd.concat([pd.DataFrame({'date':data['^GSPC']['behaviour'].keys(), 'behaviour':data['^GSPC']['behaviour'].values()}), pd.DataFrame({'date':data['^MXX']['behaviour'].keys(), 'behaviour':data['^MXX']['behaviour'].values()})], axis=0)
                        # Going back to the initial pie chart
                        
                        return om.investmentPieChart(invByMarket, palette), [invByMarket.to_dict(), 'Portfolio'], om.returnsBarChart(tableIncomes, time), "Portfolio Investments", "Portfolio Returns"
                    else:
                        try:
                            time = 6
                            market = investmentsDf.loc[investmentsDf['Stocks'] == selection, 'Market'].reset_index(drop=True)[0]
                            tableIncomes = pd.DataFrame({'date':data[market]['behaviour'].keys(), 'behaviour':data[market]['behaviour'].values()})
                            return om.investmentPieChart(invDf, palette).update_traces(pull=[0 for _ in range(len(invDf))]), [invDf.to_dict(), market], om.returnsBarChart(tableIncomes, time), f"{market} Investments", f"{market} Returns"
                        except:
                            time = 6
                            tableIncomes = pd.concat([pd.DataFrame({'date':data['^GSPC']['behaviour'].keys(), 'behaviour':data['^GSPC']['behaviour'].values()}), pd.DataFrame({'date':data['^MXX']['behaviour'].keys(), 'behaviour':data['^MXX']['behaviour'].values()})], axis=0)
                            return om.investmentPieChart(invByMarket, palette), [invByMarket.to_dict(), 'Portfolio'], om.returnsBarChart(tableIncomes, time), "Portfolio Investments", "Portfolio Returns"
                
                # This can be the market query, or the stock query
                else:
                    # Extracting the selection
                    selection = clickData['points'][0]['label']
                    
                    # validating that the selection belongs to a market
                    if selection == '^GSPC' or selection == '^MXX':
                        # Df query, where we chose the stocks of the market
                        invDf = investmentsDf.loc[investmentsDf['Market'] == selection, ['Stocks','Investment']]
                        # Function to create a pie chart
                        queryPlot = om.investmentPieChart(invDf, palette)
                        
                        time = 6
                        tableIncomes = pd.DataFrame({'date':data[selection]['behaviour'].keys(), 'behaviour':data[selection]['behaviour'].values()})
                        # Returning pie chart, and dataframe used
                        return queryPlot, [invDf.to_dict(), selection], om.returnsBarChart(tableIncomes, time), f"{selection} Investments", f"{selection} Returns"
                    # In other cases, it will be the selection of a stock
                    else:
                        
                        # We pull to 0.1 the stock selected, the other stocks remain the same
                        pull = [0.1 if selection == i else 0 for i in invDf['Stocks']]
                        
                        time = 6
                        market = investmentsDf.loc[investmentsDf['Stocks'] == selection, 'Market'].reset_index(drop=True)[0]
                        tableIncomes = pd.DataFrame({'date':data[market][selection]['behaviour'].keys(), 'behaviour':data[market][selection]['behaviour'].values()})
                        
                        return om.investmentPieChart(invDf, palette).update_traces(pull=pull).update_layout(annotations=[dict(text=f"{invDf.loc[invDf['Stocks'] == selection, 'Investment'].reset_index(drop=True)[0]/1000:,.0f}k",x=0.5,y=0.5,showarrow=False,font=dict(size=18, color='black'))]), [invDf.to_dict(), selection], om.returnsBarChart(tableIncomes, time), f"{selection} Investments", f"{selection} Returns"
            
        else:
            time = 6
            tableIncomes = pd.concat([pd.DataFrame({'date':data['^GSPC']['behaviour'].keys(), 'behaviour':data['^GSPC']['behaviour'].values()}), pd.DataFrame({'date':data['^MXX']['behaviour'].keys(), 'behaviour':data['^MXX']['behaviour'].values()})], axis=0)
            return om.investmentPieChart(invByMarket, palette), [invByMarket.to_dict(), 'Portfolio'], om.returnsBarChart(tableIncomes, time), "Portfolio Investments", "Portfolio Returns"


@callback(
    [Output('news_list_home', 'children'),
     Output('n_new_home', 'data'),
     Output('view_more_button_home', 'style')],
    [Input('view_more_button_home', 'n_clicks'),
     Input('n_new_home', 'data'),
     Input('portfolio_main_data','data')]
)

def moreNews(nClicks, storedData, data):
    if data == None:
        return "Nan", "Nan", "Nan"
    else:
        determiningName = lambda x: data[x]['longName'] if x in ['^GSPC', '^MXX'] else data['^GSPC'][x]['longName']
        

        if nClicks > 0:
            newsList, j, newsHeadersLinks = storedData
            for i in stocksList:
                success = 0
                while success < 2:
                    try:
                        if 'https' in newsDict[i][j]['image'] and newsDict[i][j]['headline'] not in newsHeadersLinks: #and (determiningName(i).split()[0] in newsDict[i][j]['headline'] or determiningName(i).split()[0] in newsDict[i][j]['summary']):
                            newsList.append(newCreation(newsDict[i][j], determiningName(i)))
                            j += 1
                            success += 1
                    except:
                        return newsList, [newsList, j, newsHeadersLinks], {'display':'none'}
                    j += 1 
            
            return newsList, [newsList, j, newsHeadersLinks], {'display':'inline'}
            
        else:
            newsList = []
            newsHeadersList = []

            for i in stocksList:
                j = 0
                success = 0
                while success < 2:
                    if 'https' in newsDict[i][j]['image'] and newsDict[i][j]['headline'] not in newsHeadersList: #and (determiningName(i).split()[0] in newsDict[i][j]['headline'] or determiningName(i).split()[0] in newsDict[i][j]['summary']):
                        
                        newsList.append(newCreation(newsDict[i][j], determiningName(i)))
                        newsHeadersList.append(newsDict[i][j]['headline'])
                        j += 1
                        success += 1
                    j += 1 
            
            return newsList, [newsList, j, newsHeadersList], {'display':'inline'}