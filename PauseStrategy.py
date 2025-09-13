#%% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime as dt
import Baseline as B

#%% Pause
vDates_daily = B.vDates[pd.Series(B.vDates[:,0]).dt.hour == 0].copy()
btc_price_daily = B.btc_price[pd.Series(B.vDates[:,0]).dt.hour == 0].copy()
btc_200ma = B.moving_average(btc_price_daily,200)
# btc_above_25x200ma = (btc_price_daily[:,0]>(2.5*btc_200ma)).copy() ## Never happens, lets kick it out for now

rsi = B.RSI(btc_price_daily,14)
rsi_above_80 = (rsi>80).copy()

# Monthly points
df_dates = pd.DataFrame(vDates_daily, columns = ['date', 'date_id'])
df_dates = df_dates[(df_dates['date'] >= dt(year=2018, day=1, month=1)) & (df_dates['date'] <= dt(year=2025, day=1, month=1))].copy()
date_SOM_idxs = df_dates[(df_dates['date'].dt.day == 1) & (df_dates['date'].dt.hour == 0)].date_id.values.astype(int)
loc_ = np.where(np.isin(vDates_daily[:,1], date_SOM_idxs))[0]

# Define investment
investment = np.repeat(250, loc_.shape[0])
vDates_monthly = vDates_daily[loc_].copy()
btc_price_monthly = btc_price_daily[loc_].copy()
indicator = rsi_above_80[loc_].copy()
investment[indicator] = 0

# Backtest RSI strategy
strategy_matrix, Q = B.SimpleBacktest(investment, btc_price_monthly)
B.PlotBacktest(vDates_monthly, strategy_matrix, investment, title="If RSI > 80 - invest 0\$, otherwise 250$")

# Compare with the benchmark
baseline_matrix, _, baseline_investment = B.BaselineTradingMonthly(isPlot=False)
B.PlotBacktest(vDates_monthly, baseline_matrix, baseline_investment, 'Baseline')

total_return_baseline = f'{(np.nansum(baseline_matrix, axis=0)[-1] / baseline_investment.sum() - 1) * 100:.2f}%'

total_return_simpleEMA = f'{(np.nansum(strategy_matrix, axis = 0)[-1]/investment.sum() - 1)*100:.2f}%' ### Also slightly outperforms

# Test if indicator has predictive power
indicator_on = []
indicator_off = []
for i in range(loc_.shape[0]):
    if indicator[i]:
        indicator_on.append((baseline_matrix[i,i:][-1]/baseline_matrix[i,i:][0])**(1/(baseline_matrix[i,i:].shape[0]))-1)
    else:
        indicator_off.append((baseline_matrix[i,i:][-1]/baseline_matrix[i,i:][0])**(1/(baseline_matrix[i,i:].shape[0]))-1)

np.mean(indicator_on)
np.mean(indicator_off)

plt.bar(np.array(["Indicator ON", "Indicator OFF"]), np.array([np.mean(indicator_on)*100, np.mean(indicator_off)*100]));
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val_,_: f"{val_}%"))
plt.title("BTC trading above and belove RSI of 80")
plt.tight_layout()
plt.show()