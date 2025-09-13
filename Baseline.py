#%% Libs
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from datetime import datetime as dt
from copy import copy

#%% Data
path_ = r'C:\Users\MI\Desktop\HedgeProject\Framework\database_1h\Raw_data'
vDates = np.load(path_+rf'/vDates.npy', allow_pickle=True)
btc_price = zarr.open_array(path_+rf'\crypto_db.zarr', mode = 'r')[78,0,:,:6]

#%% Support Functions
def SimpleBacktest(investment, btc_price):
    '''
    Function takes investment plan as input, and outputs savings accumulation over time
    :param investment: Vector per each tarde frequency day, that defines the investment amount
    :return: Returns the matrix of the savings plan. This is diagonal matrix and sum accros 0th axis is the investment value at point in time.
    '''

    Q = np.full((investment.shape[0]), np.nan) # Quantity bought at each frequency time point
    strategy_matrix = np.full((Q.shape[0], Q.shape[0]), np.nan) # Matrix that function returns

    for i in range(investment.shape[0]): # Loop over every investment cycle

        ### Define quantity that we buy
        Q[i] = (investment[i] / btc_price[i, 0]) # Quantity we buy ate every time point
        strategy_matrix[i, i:] = Q[i] * btc_price[i:, 0] # Fill in value of each unique investment

    return(strategy_matrix, Q)
def PlotBacktest(vDates_interval, strategy_matrix, investment, title):
    '''
    This function plots result of investing plan defined by strategy_matrix and investment.
    :param vDates_interval: Dates of investment periods
    :param strategy_matrix: Matrix of each trade point seperatly to track results of each investment point
    :param investment: Vector per each trade frequency day, that defines the investment amount
    :param title: Title for plot
    :return:
    '''
    plt.plot(vDates_interval[:, 0], np.nansum(strategy_matrix, axis=0), label=f"Savings plan -> ${round(np.nansum(strategy_matrix, axis=0)[-1], 2)}");
    plt.plot(vDates_interval[:, 0], investment.cumsum(), label=f"Investment -> ${round(investment.cumsum()[-1], 2)}");
    plt.legend(); plt.title(title); plt.show()

def getWeeklyFromHourly(vDates, btc_price, start, end):
    start, end = np.where(np.isin(vDates[:, 0], np.array([start, end])))[0]
    vDates_weekly = vDates[start:end + 1][::24 * 7]
    btc_price_weekly = btc_price[start:end + 1][::24 * 7]
    return(vDates_weekly, btc_price_weekly)

def getCapitalParityInvestment(oldInvestmnet, multiplier = 1):

    skipped_base = 0
    skipped_base_list = []
    capitalParityInvestment = []
    for i in range(len(oldInvestmnet)):
        if oldInvestmnet[i] == 250:
            capitalParityInvestment.append(oldInvestmnet[i])
            skipped_base_list.append(skipped_base)
        elif oldInvestmnet[i] < 250:
            skipped_base += (250 - oldInvestmnet[i])
            capitalParityInvestment.append(oldInvestmnet[i])
            skipped_base_list.append(skipped_base)
        elif oldInvestmnet[i] > 250:
            extra = min((oldInvestmnet[i] - 250)*multiplier, skipped_base, 250)
            skipped_base -= extra# + (oldInvestmnet[i] - 250)
            # capitalParityInvestment.append(oldInvestmnet[i] + extra)
            capitalParityInvestment.append(250 + extra)
            skipped_base_list.append(skipped_base)

    return (np.array(capitalParityInvestment), np.array(skipped_base_list))

#%% Backtest Baseline
def BaselineTradingMonthly(isPlot = False):
    df_dates = pd.DataFrame(vDates, columns=['date', 'date_id'])
    df_dates = df_dates[(df_dates['date'] >= dt(year=2018, day=1, month=1)) & (df_dates['date'] <= dt(year=2025, day=1, month=1))].copy()
    date_SOM_idxs = df_dates[(df_dates['date'].dt.day == 1) & (df_dates['date'].dt.hour == 0)].date_id.values.astype(int)

    ### Move dates and prices to monthly
    vDates_monthly = vDates[date_SOM_idxs].copy()
    btc_price_monthly = btc_price[date_SOM_idxs].copy()

    ### Linear investment
    investment = np.repeat(250, btc_price_monthly.shape[0])
    strategy_matrix, Q = SimpleBacktest(investment, btc_price_monthly)

    ### Plots
    if isPlot:
        PlotBacktest(vDates_monthly, strategy_matrix, investment, 'Investing 250$ in BTC every 1st of Month at 00:00')
        print(f'ROI from monthly investing 250$ is -> {round((np.nansum(strategy_matrix, axis=0)[-1]/investment.cumsum()[-1])*100, 2)}%')

    return(strategy_matrix, Q, investment)

#%% Indicators functions
def moving_average(price_timeseries, n):

    ma_ = np.empty(price_timeseries.shape[0])*np.nan

    ma_[n:] = np.array([np.nanmean(price_timeseries[i-n+1: i+1]) for i in range(n, price_timeseries.shape[0])])

    return ma_
def weighted_moving_average(price_timeseries, n):

    out = np.full(n, np.nan)

    weights = np.arange(1, period+1)
    denom = weights.sum()
    ma_[n:] = np.array([np.dot(price_timeseries[i-n+1:i+1], weights)/denom for i in range(n, price_timeseries.shape[0])])
def RSI(btc_price, n = 14):

    rsi = np.full((btc_price.shape[0]), 50)

    for i in range(btc_price.shape[0]):
        if i >= n:
            if np.isnan(btc_price[i-n:i+1]).sum() == 0:
                moves = np.diff(btc_price[i - n:i + 1, 0])
                upmoves = np.array([np.maximum(0, z) for z in moves])
                downmoves = np.array([np.maximum(0, z) for z in -moves])

                avg_u = upmoves.mean()
                avg_d = downmoves.mean()

                RS = avg_u/avg_d
                RSI = 100 - 100/(1+RS)

                rsi[i] = RSI

    return(rsi)

def ATH(btc_price, lag = 24*30):
    ATH_indicator = np.full((btc_price.shape[0]), np.nan)
    for i in range(ATH_indicator.shape[0]):
        if ((~np.isnan(btc_price[i-lag-1])).sum() > 0) & (i>lag):
            ATH_indicator[i] = np.nanmax(btc_price[:i - (lag)])
    return(ATH_indicator)

def EMA(btc_price, N = 200):

    alpha = 2 / (N + 1)

    ema = np.full((btc_price.shape[0]), np.nan)
    for i in range(ema.shape[0]):
        if i == 0:
            ema[0] = btc_price[0, 3]
        else:
            ema[i] = alpha * btc_price[i, 3] + (1 - alpha) * ema[i - 1]

    return(ema)

def max_drawdown(strategy_matrix):
    strategy_value = np.nansum(strategy_matrix, axis = 0)
    max_drawdown = 0
    for i in range(strategy_value.shape[0]):
        if i > 0:
            drawdown = strategy_value[i]/np.max(strategy_value[:i])-1
            if drawdown<max_drawdown:
                max_drawdown = drawdown.copy()
    return(max_drawdown)

def DD(price):
    drawdown = []
    for i in range(price.shape[0]):
        if i > 0:
            drawdown.append(price[i]/np.max(price[:i+1])-1)
        else:
            drawdown.append(0)
    return(-np.array(drawdown))

def std(price_timeseries, n):

    std_ = np.empty(price_timeseries.shape[0])*np.nan

    std_[n:] = np.array([np.nanstd(price_timeseries[i-n+1: i+1]) for i in range(n, price_timeseries.shape[0])])

    return(std_)