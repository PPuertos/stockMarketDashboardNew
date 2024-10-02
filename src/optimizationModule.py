import scipy.optimize as sc
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as dt
from plotly import graph_objects as go
import dash_ag_grid as dag
import matplotlib.colors as mcolors
import finnhub
import exchangeRate as er

def extractingData(stocks=list, startDate=None, closePrices=None):
    ''' Function to extract data from yfinance library '''
    ''' Input: stocks=[]; startDate=date; closePrices=bool '''
    ''' Output: data (either close prices or all the data) '''
    if startDate == None:
        period = '2y'
        data = yf.download(stocks, period=period).bfill()
    else:
        today = dt.today()
        data = yf.download(stocks, start=startDate, end=today).bfill()
    
    if closePrices == True:
        data = data['Close']
    return data

def dailyReturnRates(stocksClosePrices):
    ''' Function to calculate return rates '''
    ''' Input: data frame with close prices of each stock '''
    ''' Output: data frame with daily return rates for each stock '''
    dailyRR = stocksClosePrices.pct_change().dropna()
    return dailyRR

def expectedStocksReturns(marketReturns, stocksReturns, riskFreeRate):
    ''' Function to calculate expected returns using CAPM model '''
    ''' Input: Market and stocks return rates, risk free rate '''
    ''' Output: Expected Returns of the stocks '''
    marketVar = marketReturns.var(ddof=1)
    covariances = np.array([np.cov(stocksReturns[i], marketReturns, ddof=1)[0][1] for i in stocksReturns])
    stockBetas = np.array([cov/marketVar for cov in covariances])
    expectedMarketReturns = marketReturns.mean()
    expectedStocksReturns = riskFreeRate + stockBetas * (expectedMarketReturns - riskFreeRate)
    expectedStocksReturns = pd.Series(data=expectedStocksReturns, index=stocksReturns.columns)
    return expectedStocksReturns

def stocksStatistics(stocksDailyReturnRates, riskFreeRate=None, marketReturns=None):
    ''' Function to calculate the mean returns and covariance matrix for the stocks '''
    ''' Input: data frame with with daily returns of each stock '''
    ''' Output: list with the mean returns of each and covariance matrix '''
    if riskFreeRate is not None:
        stocksMeanReturns = expectedStocksReturns(marketReturns,stocksDailyReturnRates, riskFreeRate)
    else:
        stocksMeanReturns = stocksDailyReturnRates.mean()
    stocksCovMatrix = stocksDailyReturnRates.cov()
    return [stocksMeanReturns, stocksCovMatrix]

def portfolioPerformance(stocksMeanReturns, covMatrix, weights, nWeeks):
    ''' Function to calculate portfolio performance '''
    ''' Input: pandas series with mean return rate of each stock, data frame with covariance matrix of the stocks, and weights of each stock '''
    ''' Output: list with the portfolio returns, and the portfolio volatility '''
    weights = np.array(weights)
    
    portfolioReturns = np.sum(stocksMeanReturns * weights) * nWeeks*5
    portfolioStd = np.sqrt(weights.T @ (covMatrix @ weights)) * np.sqrt(nWeeks*5)
    return [portfolioReturns, portfolioStd]

def negativeSR(weights, stocksMeanReturns, covMatrix, nWeeks, riskFreeRate):
    ''' Objective function: calculates the sharp ratio of the portfolio '''
    ''' Input: weights and mean return rates of each stock, covariance matrix of the stocks and risk free rate '''
    ''' Output: Negative Risk Free Rate '''
    weights = np.array(weights)
    
    pReturns, pStd = portfolioPerformance(stocksMeanReturns, covMatrix, weights, nWeeks)
    
    return -(pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, nWeeks, riskFreeRate, constraintSet = (0,1)):
    ''' Function to minimize the negative sharp ratio function, which will lead to a maximization of the positive sharp ratio '''
    ''' Inputs: mean return rates of each stock, covariance matrix of the stocks, risk free rate and constraint set '''
    ''' Output: results of the minimization of the function '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, nWeeks, riskFreeRate)
    constraints = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for _ in range(numAssets))
    initialWeights = np.array(numAssets * [1. / numAssets])
    result = sc.minimize(negativeSR, initialWeights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioStd(weights, stocksMeanReturns, covMatrix, nWeeks):
    ''' Function to calculate the volatility of the portfolio '''
    ''' Input: weights and mean return rate of each stock, covariance matrix of the stocks '''
    ''' Output: Portfolio volatility '''
    return portfolioPerformance(stocksMeanReturns, covMatrix, weights, nWeeks)[1]

def minStd(meanReturns, covMartix, nWeeks, constraintSet= (0,1)):
    ''' Function to minimize the portfolioStd function '''
    ''' Input: mean raturn rate of each stock, covariance matrix of the stocks, risk free rate and constraint set '''
    ''' Output: results of the minimization of the function '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMartix, nWeeks)
    constraints = ({'type':'eq', 'fun':lambda x:np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for _ in range(numAssets))
    initialWeights = np.array(numAssets * [1. / numAssets])
    result = sc.minimize(portfolioStd, initialWeights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioReturns(weights, stocksMeanReturns, covMatrix, nWeeks):
    ''' Function to alculate the returns of the portfolio '''
    ''' Input: weights and mean return rate of each stock, covariance matrix of the stocks'''
    ''' Output: Returns of the portfolio '''
    return portfolioPerformance(stocksMeanReturns, covMatrix, weights, nWeeks)[0]

def efficientFrontierOpt(meanReturns, covMatrix, returnTarget, nWeeks, constraintSet = (0,1)):
    ''' Function to minimize the volatility of the portfolio on a given return target '''
    ''' Input: mean return rate of each stock, covariance matrix of the stocks, portofolio return target, constraint set '''
    ''' Output: Minimization of volatility of the portfolio, given a return target '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, nWeeks)
    
    ''' For the constraints, we have that the sum of the weights have to be 1, and the returns for the portfolio have to be equal
    or higher than the given target return'''
    constraints = ({'type':'eq', 'fun':lambda x:portfolioReturns(x, meanReturns, covMatrix, nWeeks) - returnTarget}, {'type':'eq', 'fun': lambda x:np.sum(x) - 1})
    bounds = (constraintSet for _ in range(numAssets))
    initialWeights = np.array(numAssets * [1./numAssets])
    effOpt = sc.minimize(portfolioStd, initialWeights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return effOpt

def efficientFrontierData(stocks_mean_rr, stocks_cov_matrix, nWeeks, riskFreeRate=0):
    ''' Function to calculate the efficient frontier '''
    ''' Input: mean return rate of each stock, covariance matrix of each stock '''
    ''' Output: max sharpe ratio, min volatility, efficient frontier '''
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(stocks_mean_rr, stocks_cov_matrix, nWeeks, riskFreeRate)
    maxSR_returns, maxSR_std = portfolioPerformance(stocks_mean_rr, stocks_cov_matrix, maxSR_Portfolio['x'], nWeeks)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=stocks_mean_rr.index, columns=['allocation'])
    # Min Volatility Portfolio
    minVol_Portfolio = minStd(stocks_mean_rr, stocks_cov_matrix, nWeeks)
    minVol_returns, minVol_std = portfolioPerformance(stocks_mean_rr, stocks_cov_matrix, minVol_Portfolio['x'], nWeeks)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=stocks_mean_rr.index, columns=['allocation'])
    
    # Efficient Frontier
    efficient_list = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    weights = []
    
    for target in targetReturns:
        optimization = efficientFrontierOpt(stocks_mean_rr, stocks_cov_matrix, target, nWeeks)
        efficient_list.append(optimization['fun'])
        weights.append(optimization['x'])
    
    efficientResults = pd.DataFrame(weights, columns=stocks_mean_rr.index)
    efficientResults['Volatility'] = efficient_list
    efficientResults['Returns'] = targetReturns
    efficientResults['Sharpe Ratio'] = (np.array(targetReturns) - riskFreeRate)/np.array(efficient_list)
    
    
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientResults

def montecarloSimulation(stocksMeanRR, stocksCovMatrix, mcSims, nWeeks):
    ''' Function to simulate the portfolio results by random sampling the weight of the stocks '''
    ''' Input: return rate of each stock, covariance matrix of the stocks and number of simulations '''
    ''' Output: Dataframe with volatility, return, and stock weights of each simulation '''
    mc_results = []
    for _ in range(mcSims):
        weights = np.random.random(size=len(stocksMeanRR))
        weights /= sum(weights)

        pReturns, pStd = portfolioPerformance(stocksMeanRR, stocksCovMatrix, weights, nWeeks)
        mc_results.append([pReturns, pStd] + weights.tolist())
    
    mc_results = pd.DataFrame(mc_results, columns=['Returns','Volatility']+stocksMeanRR.index.tolist())
    return mc_results

### STYLING FUNCTION ###
def get_color_from_value(values, colors, agGrid=True):
    ''' Function to get a scale of colors '''
    ''' Input: values to asign colors (list or pandas series), list or pandas series with the colors you want '''
    ''' Output: List with the colors '''
    min_value = values.min()
    max_value= values.max()
    
    # Normaliza el valor en el rango [0, 1]
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    
    # Crea un colormap lineal a partir de la lista de colores
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    # Obtén el color correspondiente al valor normalizado
    colors = []
    for value in values:
        normalized_value = norm(value)
        color = cmap(normalized_value)
        color = tuple([j*255 if i != 4 else j for i, j in enumerate(color)])
        
        if agGrid == True:
            styling = {
                'condition':f'params.value === {value}', 
                'style':{'background-color':f"rgba{color}"}
                }
            colors.append(styling)
        if agGrid == False:
            colors.append((value, color))
    
    return tuple(colors)

def efficientFrontierPlot(efficientResults, maxSR_std, maxSR_returns, minVol_std, minVol_returns, mc_sim, nWeeks):
    ''' Function to graph the efficient frontier '''
    ''' Input: '''
    ''' Output: '''
    font_fam = ['Arial', 'Balto', 'Courier New', 'Droid Sans', 'Droid Serif', 'Droid Sans Mono', 'Gravitas One', 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow', 'Raleway', 'Times New Roman']

    efficient_list = efficientResults['Volatility']
    targetReturns = efficientResults['Returns']
    minVarColor = 'rgba(74,229,74,1)'
    maxSPColor = 'rgba(171,0,255,1)'
    middleColor = 'rgba(0,120,255,1)'
    sec_color = 'rgba(150,150,150,.1)'


    MaxSharpeRatio = go.Scatter(
        name='Sharp Ratio Max.',
        x = [maxSR_std*100],
        y = [maxSR_returns*100],
        mode='markers',
        marker=dict(color=maxSPColor, size=18, symbol=300),
        hoverinfo='skip'
    )

    MinVolatility = go.Scatter(
        name='Volatility Min.',
        x = [minVol_std*100],
        y = [minVol_returns*100],
        mode='markers',
        marker=dict(color=minVarColor, size=20, symbol=300),
        hoverinfo='skip'
    )
    tod = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    
    # For hover template
    # Agregamos dinámicamente los pesos de cada acción a partir de customdata
    hovertemplate=(
            '<b>Results</b><br>'
            '   -   Volatility: %{x:,.2f}%<br>'  # Adjusts the format of Volatility
            '   -   Returns: %{y:,.2f}%<br>' # Adjusts the format of Returns
            '   -   Sharpe Ratio: %{customdata[0]:,.2f}<br><br>'
            '<b>Weights</b><br>'
        )
    
    customdata = [efficientResults['Sharpe Ratio']]
    for i, stock in enumerate(efficientResults.columns[:-3]):
        hovertemplate += f'   -   {stock}: %{{customdata[{i+1}]:.0f}}%<br>'
        customdata.append(efficientResults[stock]*100)
    hovertemplate += '<extra></extra>'
    
    customdata = np.stack(customdata, axis=1)
    
    # Asignar tamaño básico (5)) para todos los puntos
    sizes = np.full_like(efficient_list, 8)

    # Identificar el índice de los puntos con menor y mayor volatilidad
    min_vol_idx = np.argmin(efficient_list)
    max_vol_idx = np.argmax(efficient_list)

    # Cambiar el tamaño de los puntos con menor y mayor volatilidad a 15
    sizes[min_vol_idx] = 15
    sizes[max_vol_idx] = 15
    
    efficientFrontier = go.Scatter(
        name='Efficient Frontier',
        x = [i*100 for i in efficient_list],
        y = [i*100 for i in targetReturns],
        mode='lines+markers',
        line=dict(color='rgb(0,0,0,1)', dash=tod[0], width=1.5),
        hoverinfo=None,
        hoverlabel=dict(
            bgcolor='rgb(245,245,245)',  # Background color with transparency
            font_size=16,               # Font size (adjusted for better readability)
            font_color="black",         # Font color (white for contrast on dark background)
            bordercolor="rgb(0,0,0,1)",        # Border color
            namelength=-1,               # Full length of the name in the tooltip
            font_family=font_fam[10],
            align='left'
        ),
        marker=dict(color=efficient_list,colorscale=[[0,minVarColor],[0.5,middleColor],[1,maxSPColor]], size=sizes, line=dict(width=0)),
        customdata=customdata,
        hovertemplate=hovertemplate
    )
    
    mc_sim_plt = go.Scatter(
        name='Monte Carlo Sim.',
        x = [i*100 for i in mc_sim['Volatility']],
        y = [i*100 for i in mc_sim['Returns']],
        mode='markers',
        marker=dict(color=sec_color, size=8),
        hoverinfo='skip'
    )
    
    if nWeeks == 1:
        periodCalc = '1 Week'
    elif nWeeks == 2:
        periodCalc = '2 Week'
    elif nWeeks == 4:
        periodCalc = 'Monthly'
    elif nWeeks == 12:
        periodCalc = 'Quarterly'
    elif nWeeks == 24:
        periodCalc = '6 Month'
    elif nWeeks == 48:
        periodCalc = 'Annualized'
    elif nWeeks == 96:
        periodCalc = '2 Year'
    elif nWeeks == 240:
        periodCalc = '5 Year'

    fig = go.Figure(
        data=[mc_sim_plt, efficientFrontier, MaxSharpeRatio, MinVolatility],
        layout=dict(template='none', title=dict(text='<b>Efficient Frontier Optimization</b>', x=.5, font=dict(size=20)), xaxis=dict(title=f'{periodCalc} Volatility (%)'), yaxis=dict(title=f'{periodCalc} Returns (%)')),
    )

    fig.update_layout(hovermode='x', paper_bgcolor='#fafafa', plot_bgcolor='#fafafa')
    # Optionally, you can add vertical 'spike' lines on hover to make the nearest y-point clearer
    fig.update_xaxes(showspikes=True, spikecolor='rgb(0,0,0)', spikemode="across", spikethickness=1, spikedash=tod[1], spikesnap='hovered data')
    fig.update_yaxes(showspikes=True, spikecolor='rgb(0,0,0)', spikemode="across", spikethickness=1, spikedash=tod[1], spikesnap='hovered data')


    max_y = efficientResults[efficientResults['Returns'] == efficientResults['Returns'].max()].iloc[-1,-2]*100
    max_y = max_y*1.1
    max_x = efficientResults[efficientResults['Returns'] == efficientResults['Returns'].max()].iloc[-1,-3]*100
    max_x = max_x*1.03
    min_y = efficientResults[efficientResults['Volatility'] == efficientResults['Volatility'].min()].iloc[-1,-2]*100
    min_y = min_y*.9
    min_x = efficientResults[efficientResults['Volatility'] == efficientResults['Volatility'].min()].iloc[-1,-3]*100
    min_x = min_x*.98

    fig.update_layout(
        xaxis=dict(
            range=[min_x, max_x]  # Set the x-axis range
        ),
        yaxis=dict(
            range=[min_y, max_y]  # Set the y-axis range
        ),
        showlegend=False
    )
    
    return fig

def columnsTable(dataFrame):
    ''' Function to display efficient frontier results '''
    ''' Input: efficient frontier results '''
    ''' Output: table with visual effects for a better understanding of the data '''
    ['string', 'currency', 'number', 'percentage']
    valFormat = lambda x: "d3.format(',.2f')(params.value)" if x == 'Sharpe Ratio' else "d3.format(',.0%')(params.value)"

    columnDefs = [{'field':i, 'headerName':i, 'valueFormatter':{"function": valFormat(i)}, 'minWidth': 100, "filter": "agSetColumnFilter"} for i in dataFrame]

    grid = dag.AgGrid(
        id="efficientFrontierDataResults",
        rowData=dataFrame.to_dict('records'),
        columnDefs=columnDefs,
        columnSize='responsiveSizeToFit',
        defaultColDef={"sortable": True, "floatingFilter": True},
        className='ag-theme-balham',
    )
    return grid

def efficientFrontierTable(effRes):
    ''' Function to display efficient frontier results '''
    ''' Input: efficient frontier results '''
    ''' Output: table with visual effects for a better understanding of the data '''
    metricCols = list(effRes.iloc[:,-3:])
    stocksCols = [i.split('.')[0] + ' MX' if 'MX' in i else i for i in effRes.iloc[:,:-3]]
    
    efficientResults = pd.DataFrame(data=effRes.values, columns=stocksCols + metricCols)
    
    valFormat = lambda x: "d3.format(',.2f')(params.value)" if x == 'Sharpe Ratio' else "d3.format(',.0%')(params.value)"
    
    colors = ['#4ae54a', '#0078ff', '#ab00ff']  # Azul, Verde, Rojo
    styleConditionsVol = {'styleConditions':get_color_from_value(efficientResults['Volatility'], colors)}
    styleConditionsRet = {'styleConditions':get_color_from_value(efficientResults['Returns'], colors[::-1])}
    styleConditionsSR = {'styleConditions':get_color_from_value(efficientResults['Sharpe Ratio'], colors[::-1])}


    columnDefs = [{'field':'Returns', 'headerName':'Returns', 'valueFormatter':{"function": valFormat('Returns')}, 'minWidth': 100, "filter": "agSetColumnFilter", "cellStyle": styleConditionsRet}] + [{'field':'Volatility', 'headerName':'Volatility', 'valueFormatter':{"function": valFormat('Volatility')}, 'minWidth': 100, "filter": "agSetColumnFilter", "cellStyle": styleConditionsVol}] + [{'field':'Sharpe Ratio', 'headerName':'Sharpe Ratio', 'valueFormatter':{"function": valFormat('Sharpe Ratio')}, 'minWidth': 100, "filter": "agSetColumnFilter", "cellStyle": styleConditionsSR}]  + [{'field':i, 'headerName':i, 'valueFormatter':{"function": valFormat(i)}, 'minWidth': 100, "filter": "agSetColumnFilter"} for i in stocksCols]

    grid = dag.AgGrid(
        id="efficientFrontierDataResults",
        rowData=efficientResults.to_dict('records'),
        columnDefs=columnDefs,
        columnSize='responsiveSizeToFit',
        defaultColDef={"sortable": True, "floatingFilter": True},
        className='ag-theme-balham',
    )
    
    return grid


#### FOR PORTFOLIO SECTION ####
#### FOR PORTFOLIO SECTION ####
#### FOR PORTFOLIO SECTION ####
#### FOR PORTFOLIO SECTION ####
def ols(X, Y, b0=True):
    ''' Function to model a data sample, using Ordinary Least Squares method '''
    ''' Input: X matrix (n * k+1), Y matrix (n * 1) '''
    ''' Output: Portfolio b0, b1, ..., bk '''
    # If b0 == True, it means the model doesn't intersects with the origin of the model, so b0 value is going to be determined, and we add a column of ones to the X matrix
    if b0 == True:
        X =  np.column_stack([np.ones(X.shape[0]), X])
    # If b0 == False, it means the model intersects with the origin, so b0 value is 0, and we add a column of zeros to the X matrix
    else:
        X = np.column_stack([np.zeros(X.shape[0]), X])
    Xt = X.T
    XtX = Xt @ X
    XtX_inv = np.linalg.inv(XtX)
    bi = XtX_inv @ (Xt @ Y)
    return {'alpha':bi[0], 'beta':bi[1]}

def portfolioQueryResults(portfolioBehaviour, portfolioRecords):
    actualValue = portfolioBehaviour.loc[portfolioBehaviour.index.max()]
    investment = portfolioRecords.loc[portfolioRecords.index >= portfolioBehaviour.index.min(),'qty_bought_usd'].sum()
    initialValue = portfolioBehaviour.loc[portfolioBehaviour.index.min()] + investment
    returns = actualValue - initialValue
    returnRate = returns/initialValue
    
    return returns, returnRate

def valueAtRisk(stocksReturnRates, stocksWeights, portfolioInvestment):
    ''' Function to calculate value at risk and Conditional Value at Risk '''
    ''' Input: Stocks daily return rates, stock weights, portfolio investment '''
    ''' Output: Daily VaR and Conditional VaR, both with 95% confidence. '''
    portfolioReturnR = stocksReturnRates @ stocksWeights
    
    # Value at Risk 95% Confidence
    VaR_95 = np.percentile(portfolioReturnR, 5)
    # Conditional Value at Risk 95% Confidence
    CVaR_95 = portfolioReturnR[portfolioReturnR.values <= VaR_95].mean()
    
    
    return -VaR_95*portfolioInvestment, -CVaR_95*portfolioInvestment

def riskScore(volatility):
    # Risk Score Function
    risk_score = lambda x: 1 if x<=.005 else 2 if x<=.012 else 3 if x<=.02 else 4 if x<=.027 else 5 if x<=.039 else 6 if x<=.054 else 7 if x<=.077 else 8 if x<=.155 else 9 if x<=.233 else 10
    
    score = risk_score(volatility)
    
    # Risk Score Colors
    risk_posible_values = pd.Series([i+1 for i in range(10)])
    colors = get_color_from_value(risk_posible_values, ['#3DD508','#E3E008','#D20B0B'])
    
    color = [colors[i] for i in range(10) if score == i+1]
    color = color[0]['style']['background-color']
    
    return score, color

def portfolioActuals():
    ''' Function to get all the data from your investments '''
    ''' Input: Nothing '''
    ''' Output: financial results and metrics for a better understanding of your portfolio health '''
    purchaseRecords = pd.read_csv('src/assets/purchaseRecords.csv')
    purchaseRecords.index = pd.to_datetime(purchaseRecords['date']).dt.strftime('%Y-%m-%d')
    purchaseRecords = purchaseRecords.drop(columns='date')
    
    exchangeRate = float(er.ExchangeUsdToMxn())
    
    # Separed Data Frames for Each Market
    # Purchase records by market
    portfolio_by_market = {}
    # Total investment by market
    initial_investment = {}
    # Oldest purchase record by market
    oldest_date = {}
    # Stock weights by market
    weights = {}
    # Total investment in each stock by market
    investment_by_stock = {}
    
    # Cicle for each market
    for market in purchaseRecords['market'].unique():
        # Market purchase records
        market_records = purchaseRecords.loc[purchaseRecords['market'] == market,:]
        portfolio_by_market[market] = market_records
        # Market all time investment
        initial_investment[market] = market_records['qty_bought_usd'].sum()
        # Oldest purchase date (optimization purposes)
        oldest_date[market] = market_records.index.min()
        # Stocks historical investments from the market
        inv_by_stock = market_records.groupby('stock').agg({'qty_bought_usd':'sum'})['qty_bought_usd']
        investment_by_stock[market] = inv_by_stock
        # Portfolio weights in the market
        weights[market] = inv_by_stock/sum(inv_by_stock)
        

    # Main Dictionary
    portfolio_data = {}
    # Starting portfolio behaviour variable
    port_behaviour = pd.Series()
    # Cicle to evaluate all the markets
    for market in portfolio_by_market:
        # Stocks investments in the market
        stocks = portfolio_by_market[market]['stock'].unique().tolist()
        # behaviour of all the stock in a sepecific market of the portfolio
        market_behaviour = pd.Series()
        # Dicitonary for the specific market
        portfolio_data[market] = {'longName':yf.Ticker(market).info['longName']}
        # Stocks close prices
        stocks_close_prices = extractingData(stocks=stocks + [market], startDate=oldest_date[market], closePrices=True)
        stocks_close_prices.index = pd.Series(stocks_close_prices.index).dt.strftime('%Y-%m-%d')
        # Stocks Return Rates
        stocks_daily_r_rates = dailyReturnRates(stocks_close_prices)
        
        if market == '^MXX':
            stocks_close_prices = stocks_close_prices/exchangeRate
            stocks_daily_r_rates = stocks_daily_r_rates
        # Cicle to evaluate the stocks of each market
        for stock in portfolio_by_market[market]['stock'].unique():
            # Purchase record of the stock
            stock_info = portfolio_by_market[market][portfolio_by_market[market]['stock'] == stock]
            # Stock close prices
            stock_hist_data = stocks_close_prices[stock]
            
            # Making new dicionary were we will store the complete data of the stock (in df), and the returns of every inversion made of each stock
            portfolio_data[market][stock] = {'longName':yf.Ticker(stock).info['longName'], 'shortName':yf.Ticker(stock).info['shortName'],'data':stock_hist_data.to_dict()}
            # Cicle to evaluate returns of each record
            stock_behaviour = pd.Series()
            for date in stock_info.index:
                # Close Prices of the record
                close_prices = stock_hist_data[stock_hist_data.index >= date]
                # Record Info
                record_info = stock_info.loc[date, :]
                # Stock purchase price
                purchase_price = record_info['stock_price_usd']
                # Purchase Quantity
                total_bought = record_info['qty_bought_usd']
                # Purchase in stock units
                stock_units = record_info['n_stocks']
                # Record Behaviour
                record_behaviour = stock_units*close_prices
                # Adding the record to the stock behaviour
                stock_behaviour = stock_behaviour.add(record_behaviour, fill_value=0)
                # Adding the record behaviour to its section in the dictionary (conclusion of the record)
                portfolio_data[market][stock][date] = {'close_prices':close_prices, 'purchase_price':purchase_price, 'total_bought':total_bought, 'record_behaviour':record_behaviour.to_dict()}
            # Stock behaviour            
            portfolio_data[market][stock]['behaviour'] = stock_behaviour.to_dict()
            # Adding the stock behaviour to the market behaviour
            market_behaviour = market_behaviour.add(stock_behaviour, fill_value=0)
        # Concluding the market behaviour
        portfolio_data[market]['behaviour'] = market_behaviour.to_dict()
        # Market Stocks Daily Return Rates
        portfolio_data[market]['stocks_daily_return_rates'] = stocks_daily_r_rates.to_dict()
        # Actual Portfolio Worth (In that specific market)
        market_actual_worth = market_behaviour[market_behaviour.index.max()]
        # Capital invested in the market
        investment_in_market = initial_investment[market]
        # Return Rate
        market_return_rate = market_actual_worth/investment_in_market - 1
        # Adding to the portfolio behaviour
        port_behaviour = port_behaviour.add(market_behaviour, fill_value=0)
        
        # Adding all of this information to the dictionary
        portfolio_data[market]['Other Results'] = {'actual_worth':market_actual_worth,'total_investment':investment_in_market, 'return_rate':market_return_rate}
        portfolio_data[market]['close_prices'] = stocks_close_prices[market].to_dict()
        portfolio_data['records_by_market'] = {i:portfolio_by_market[i].to_dict() for i in portfolio_by_market}
        portfolio_data['stock_weights'] = {i:weights[i].to_dict() for i in weights}
        portfolio_data['investment_by_market'] = {i:investment_by_stock[i].to_dict() for i in investment_by_stock}
        
    # Concluding portfolio historical behaviour    
    portfolio_data['behaviour'] = port_behaviour.to_dict()
    
    return portfolio_data

def portfolioWorthPlot(data, fill_color, border_color):
    ''' Function to '''
    tod = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    font_fam = ['Arial', 'Balto', 'Courier New', 'Droid Sans', 'Droid Serif', 'Droid Sans Mono', 'Gravitas One', 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow', 'Raleway', 'Times New Roman']
    
    y = data
    
    linePlot = go.Scatter(
        name='',
        fill='tozeroy',
        x=data.index,
        y=y,
        mode='lines',
        line=dict(dash=tod[0], width=2),
        customdata=np.stack([data.pct_change().dropna()], axis=1),
        hoverinfo=None,
        hoverlabel=dict(
            bgcolor="rgba(230,230,230,.8)",  
            font_size=16,               
            font_color="black",         
            bordercolor="#000",        
            namelength=-1,               
            font_family=font_fam[10],
            align='left'
            ),
        hovertemplate=(
            '<b>Portfolio Worth</b><br>'
            '   -   Date: %{x}<br>'  
            '   -   Returns: $%{y:,.2f}<br>' 
            '   -   Return Rate: %{customdata[0]:,.2f}%<br><br>'
            )
    )



    figure = go.Figure(data=linePlot)
    figure.update_layout(hovermode='x', template='simple_white', showlegend=False, margin=dict(l=40, r=0, t=15, b=30), paper_bgcolor='#fafafa', plot_bgcolor='#fafafa', yaxis=dict(range=[y.min(), y.max()]))
    # Optionally, you can add vertical 'spike' lines on hover to make the nearest y-point clearer
    figure.update_xaxes(showspikes=True, spikecolor="black", spikemode="across", spikethickness=1, spikedash=tod[1], spikesnap='hovered data')
    figure.update_yaxes(showspikes=True, spikecolor="black", spikemode="across", spikethickness=1, spikedash=tod[1], spikesnap='hovered data')
    figure.update_traces(fillcolor=fill_color, line_color= border_color)
    
    return figure

### HOME FUNCTIONS ###
### HOME FUNCTIONS ###
### HOME FUNCTIONS ###

# Función para generar tonos variando la luminosidad
def generate_shades(color, n):
    color_rgb = mcolors.hex2color(color)  # Convertir color a RGB
    shades = []
    
    # Generar tonos más oscuros y claros
    for i in np.linspace(0.6, 1.4, n):  # Factor de ajuste
        shade = tuple(np.clip(np.array(color_rgb) * i, 0, 1))  # Clip para evitar desbordes
        shades.append(mcolors.to_hex(shade))
    
    return shades

def investmentPieChart(investments, palette):
    ### INVESTMENT PIE CHART ###
    ### INVESTMENT PIE CHART ###
    ### INVESTMENT PIE CHART ###

    investmentPie = go.Figure(data=[go.Pie(labels=investments.iloc[:,0],
                                values=investments['Investment'], hole=.35)])
    investmentPie.update_traces(hoverinfo='label+value+percent', textinfo='percent', textfont_size=11,
                    marker=dict(line=dict(color='#000000', width=1), colors=[palette[i] for i in range(len(investments))]))

    total = sum(investments['Investment'])
    investmentPie.add_annotation(text=f'${total/1000:.0f}k',  # Texto del total
                    x=0.5, y=0.5,  # Posición centrada
                    font_size=18, showarrow=False,  # Ajustes del texto
                    font=dict(color='black'))  # Color del texto

    investmentPie.update_layout(
        legend=dict(
            orientation="h",  # Orientación horizontal
            x=0.5,  # Posiciona la leyenda horizontalmente centrada
            y=-0.1,  # Posiciona la leyenda debajo del gráfico
            xanchor='center',  # Ancla la leyenda al centro horizontal
            yanchor='top',  # Ancla la leyenda al borde superior de su posición
        ),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select',
        margin=dict(l=0, r=0, t=1, b=0)
    )
    return investmentPie

def returnsBarChart(tableIncomes, time):
    tableIncomes['returns'] = tableIncomes['behaviour'].diff()

    tableIncomes['date'] = pd.to_datetime(tableIncomes['date'])
    tableIncomes['year'] = tableIncomes['date'].dt.strftime('%Y').astype(int)
    tableIncomes['monthNo'] = tableIncomes['date'].dt.strftime('%m').astype(int)
    tableIncomes['month'] = pd.to_datetime(tableIncomes['date']).dt.strftime('%b')


    tableIncomes = tableIncomes.groupby(['year', 'monthNo']).agg({'returns':'sum', 'month':'first'}).iloc[-time:,:].reset_index()
    yearMinMonths = tableIncomes.groupby('year').agg({'monthNo':'min'})
    tableIncomes['date'] = [f"<b>{year}</b> <i><b>{month}</b></i>" if monthNo == yearMinMonths.loc[year,'monthNo'] else f'{month}' for year, month, monthNo in zip(tableIncomes['year'], tableIncomes['month'], tableIncomes['monthNo'])]

    colorsAssigned = ['rgba(75, 222, 111, .5)' if i > 0 else 'rgba(233,79,98,.5)' for i in tableIncomes['returns']]
    borderColors = ['rgba(75, 222, 111, 1)' if i > 0 else 'rgba(233,79,98,1)' for i in tableIncomes['returns']]
    
    max_value = tableIncomes['returns'].max() + 1000
    minValueFunction = lambda x: x-5000 if x < 0 and x > -2000 else (x if x < 0 else 0)
    lambda x: x - 5000 if x < 0 else (x if x < 1000 else 0)


    min_value = minValueFunction(tableIncomes['returns'].min())

    bar = go.Bar(x=tableIncomes['returns'], y=tableIncomes['date'], marker_color=colorsAssigned, width=[.8 for _ in range(len(tableIncomes))], orientation='h',text=[f'${i/1000:,.0f}k' for i in tableIncomes['returns']],  # Agrega el texto que deseas mostrar en las barras
        textposition='auto', textfont=dict(color='black'), marker=dict(line=dict(color=borderColors, width=1.5)))
    

    plot = go.Figure([bar]).update_layout(margin=dict(l=0, r=0, t=0, b=0),template='simple_white', xaxis=dict(showticklabels=False), width=500, yaxis=dict(autorange='reversed'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', shapes=[dict(type='line',x0=0, x1=0,y0=-0.5, y1=len(tableIncomes['returns']) - 0.5,line=dict(color='black', width=2))]).update_traces(hoverinfo='y+text')
    
    return plot

def treeMapAndPopularDf(popularList):
    font_fam = ['Arial', 'Balto', 'Courier New', 'Droid Sans', 'Droid Serif', 'Droid Sans Mono', 'Gravitas One', 'Old Standard TT', 'Open Sans', 'Overpass', 'PT Sans Narrow', 'Raleway', 'Times New Roman']

    stockSymbolList = []
    marketCapList = []
    stockPctChangeList = []
    stockLongNameList = []
    stockChangeList = []
    fiftyTwoWeekPctChangeList = []
    volumeList = []

    for i in popularList:
        # Stock Symbol
        stockSymbol = i['symbol']
        # Stock long name
        stockLongName = i['shortName']
        # Stock change (daily)
        stockChange = i['regularMarketChange']
        # Stock change pct (daily)
        stockPctChange = i['regularMarketChangePercent']
        # 52 week chance percent
        fiftyTwoWeekPctChange = i['fiftyTwoWeekChangePercent']
        # Market Volume (daily)
        volume = i['regularMarketVolume']
        # Market Cap
        try:
            marketCap = i['marketCap']
            marketCapList.append(marketCap)
        except:
            print(f"There is not market cap in {stockSymbol}")
            marketCapList.append(0)
        
        stockPctChangeList.append(stockPctChange)
        stockSymbolList.append(stockSymbol)
        stockLongNameList.append(stockLongName)
        stockChangeList.append(stockChange)
        fiftyTwoWeekPctChangeList.append(fiftyTwoWeekPctChange)
        volumeList.append(volume)
        
        
    dataframe = pd.DataFrame({'Symbol':stockSymbolList,'Long Name':stockLongNameList, 'Change %':stockPctChangeList,'Change':stockChangeList, 'Volume':volumeList, 'Market Cap':marketCapList, '52 Wk Change %':fiftyTwoWeekPctChangeList}) 


    marketCapFormat = lambda x: (
        f"{x/1_000:,.2f}k" if x < 1_000_000 else
        f"{x/1_000_000:,.2f}M" if x < 1_000_000_000 else
        f"{x/1_000_000_000:,.2f}B" if x < 1_000_000_000_000 else
        f"{x/1_000_000_000_000:,.2f}T"
    )

    # Crear los labels y valores
    labels = stockSymbolList
    values = marketCapList
    marketCapFormated = [marketCapFormat(i) for i in values]
    volumeListFormatted = [marketCapFormat(i) for i in volumeList]
    # Function to determine values, for colors
    scale = lambda x: 3 if x>=3 else (-3 if x<=-3 else x)
    colors = [scale(i) for i in stockPctChangeList]

    # Function to determine format for currencie

    if len(np.unique(colors)) == 1:
        if np.unique(colors)[0] == 3:
            color = 'rgb(53, 121, 103)'
            marker = dict(
            line=dict(color='#fafafa', width=3),
            pad=dict(t=3, l=3, r=3, b=3),
            colors=[color for _ in range(len(colors))]
        )
        elif np.unique(colors)[0] == -3:
            color = 'rgb(196, 44, 45)'
            marker = dict(
            line=dict(color='#fafafa', width=3),
            pad=dict(t=3, l=3, r=3, b=3),
            colors=[color for _ in range(len(colors))]
        )
    else:
        marker = dict(
        line=dict(color='#fafafa', width=3),
        pad=dict(t=3, l=3, r=3, b=3),
        colors=colors,  # Puedes ajustar esto según lo que desees
        colorscale=['rgb(196, 44, 45)', 'rgb(222, 224, 228)', 'rgb(53, 121, 103)']
    )


    customData = np.array([marketCapFormated, stockLongNameList, stockChangeList, volumeListFormatted, fiftyTwoWeekPctChangeList]).T
    # Crear el treemap
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[''] * len(labels),  # Sin padres para un solo nivel
        values=values,
        text=[f"{i:,.2f}%" for i in stockPctChangeList],
        customdata=customData,
        textinfo='label+text',
        texttemplate='<b>%{label}</b><br>%{text}',
        marker=marker,
        hoverinfo=None,
        hoverlabel=dict(
            bgcolor="rgba(230,230,230,.8)",  
            font_size=16,               
            font_color="black",         
            bordercolor="black",      
            namelength=-1,               
            font_family=font_fam[10],
            align='auto'
            ),
        hovertemplate=(
            '<b>%{label} - %{customdata[1]}</b><br><br>'
            'Day Return R.         %{text}<br>'
            'Day Return            %{customdata[2]:$,.2f}<br>'
            'Day Volume            %{customdata[3]}<br><br>'
            'Market Cap            %{customdata[0]}<br>'
            '52 Wk Change       %{customdata[4]:,.2f}%'
            '<extra></extra>'
            ),
        root=dict(color='#fafafa'),
    ))
    # Ajustar el diseño
    fig.update_layout(paper_bgcolor='rgba(255, 255, 255, 0)', plot_bgcolor='rgba(255, 255, 255, 0)', margin=dict(l=0, r=0, t=0, b=0), uniformtext=dict(minsize=12, mode='hide'), treemapcolorway=['#fafafa'])
    
    return fig, dataframe

def importNews(stock, timelapse):
    today = dt.today()
    startDate = today - pd.Timedelta(days=timelapse)
    # Set up the client
    api_key = 'cr78eshr01qg9ve8qqjgcr78eshr01qg9ve8qqk0'
    finnhub_client = finnhub.Client(api_key=api_key)

    # Get the latest news for a company (e.g., Apple)
    news = finnhub_client.company_news(stock, _from=startDate.strftime('%Y-%m-%d'), to=today.strftime('%Y-%m-%d'))
    return news

def NewsForAllActives(allActives):
    newsDict = {}
    for i in allActives:
        newsDict[i] = importNews(i, 7)
    return newsDict