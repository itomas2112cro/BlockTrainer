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

#%% Simple backtest function
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

#%% Plots functions
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

#%% Start
plt.plot(vDates[:,0], btc_price[:,3]); plt.show() ## BTC PRICE

### Simple trading strategy: Invest 250$ in BTC every week from 1. Jan 2018 to 1. Jan 2025
date_start = np.where(vDates[:,0] == dt(year = 2018, day = 1, month = 1))[0][0]
date_end = np.where(vDates[:,0] == dt(year = 2025, day = 1, month = 1))[0][0]
freq = 24*7


### Move dates and prices to weekly
vDates_weekly = vDates[date_start:date_end+1][::freq].copy()
btc_price_weekly = btc_price[date_start:date_end+1][::freq].copy()

### Linear investment
investment = np.repeat(250,btc_price_weekly.shape[0])
strategy_matrix, Q = SimpleBacktest(investment, btc_price_weekly)

### Plots
plt.plot(vDates_weekly[:,0], Q); plt.title("Purchasing power of 250$ in terms of BTC"); plt.show()

### Naive savings plan
PlotBacktest(vDates_weekly, strategy_matrix, investment, 'Investing 250$ in BTC every Monday at 00:00')
print(f'ROI from weekly investing 250$ is -> {round((np.nansum(strategy_matrix, axis=0)[-1]/investment.cumsum()[-1])*100, 2)}%')

#%% same thing for investing every month
df_dates = pd.DataFrame(vDates, columns=['date', 'date_id'])
df_dates = df_dates[(df_dates['date'] >= dt(year = 2018, day = 1, month = 1) )& (df_dates['date'] <= dt(year = 2025, day = 1, month = 1) )].copy()
date_SOM_idxs = df_dates[(df_dates['date'].dt.day == 1 )& (df_dates['date'].dt.hour == 0)].date_id.values.astype(int)

### Move dates and prices to monthly
vDates_monthly = vDates[date_SOM_idxs].copy()
btc_price_monthly = btc_price[date_SOM_idxs].copy()

### Linear investment
investment = np.repeat(250, btc_price_monthly.shape[0])
strategy_matrix, Q = SimpleBacktest(investment, btc_price_monthly)


### Plots
PlotBacktest(vDates_monthly, strategy_matrix, investment, 'Investing 250$ in BTC every 1st of Month at 00:00')
print(f'ROI from monthly investing 250$ is -> {round((np.nansum(strategy_matrix, axis=0)[-1]/investment.cumsum()[-1])*100, 2)}%')


#%% investing based on indicator (weekly)

test_indicator = np.random.randint(0, 4, btc_price_weekly.shape[0])
decision_dict = {0:0, 1:125, 2:250, 3:1000}
cash_position = 0
investment = np.zeros((btc_price_weekly.shape[0]))
for i in range(btc_price_weekly.shape[0]):
    cash_position += 250
    decision_ = decision_dict[test_indicator[i]]
    if decision_ <= cash_position:
        investment[i] = decision_
        cash_position -= decision_
    else:
        investment[i] = copy(cash_position)
        cash_position = 0
investment[-1] += cash_position

strategy_matrix, Q = SimpleBacktest(investment, btc_price_weekly)

### plots
PlotBacktest(vDates_weekly, strategy_matrix, investment, 'Investing fallowing random indicator.')
print(f'ROI from investing fallowing random indicator -> {round((np.nansum(strategy_matrix, axis=0)[-1]/investment.cumsum()[-1])*100, 2)}%')



#%% Indicators

def moving_average(price_timeseries, n):

    ma_ = np.empty(price_timeseries.shape[0])*np.nan

    ma_[n:] = np.array([np.nanmean(price_timeseries[i-n+1: i+1]) for i in range(n, price_timeseries.shape[0])])

    return ma_


MA200 = moving_average(btc_price[:, 0], n=24*200)

plt.plot(vDates[:, 0], MA200); plt.title('MA200'); plt.show()

