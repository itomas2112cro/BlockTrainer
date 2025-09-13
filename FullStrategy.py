#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime as dt
import Baseline as B

#%% Final strategy
vDates_daily = B.vDates[pd.Series(B.vDates[:,0]).dt.hour == 0].copy()
btc_price_daily = B.btc_price[pd.Series(B.vDates[:,0]).dt.hour == 0].copy()

# Filtration 1
btc_200ma = B.moving_average(btc_price_daily, 200)
btc_bellow_200ms = (btc_price_daily[:,0]<btc_200ma).copy()

#Filtration 2
btc_ath_lagged = B.ATH(B.btc_price[:,1])
btc_ath_lagged = btc_ath_lagged[vDates_daily[:,1].astype('int')].copy()
btc_ath_indicator = (btc_price_daily[:,0]>btc_ath_lagged).copy()

# Filtration 3
rsi = B.RSI(btc_price_daily,14)
rsi_above_80 = (rsi>80).copy()

# Test that events never happen at the same time
((btc_bellow_200ms*1+btc_ath_indicator*1+rsi_above_80*1)>=2).sum() ### Happens in intramonth. Lets see if it happens monthly

# Locate monthly events
df_dates = pd.DataFrame(vDates_daily, columns = ['date', 'date_id'])
df_dates = df_dates[(df_dates['date'] >= dt(year=2018, day=1, month=1)) & (df_dates['date'] <= dt(year=2025, day=1, month=1))].copy()
date_SOM_idxs = df_dates[(df_dates['date'].dt.day == 1) & (df_dates['date'].dt.hour == 0)].date_id.values.astype(int)
loc_ = np.where(np.isin(vDates_daily[:,1], date_SOM_idxs))[0]

# Test that event never happen same time
((btc_bellow_200ms[loc_]*1+btc_ath_indicator[loc_]*1+rsi_above_80[loc_]*1)>=2).sum() # 3 same occurances

((btc_ath_indicator[loc_]*1+rsi_above_80[loc_]*1)>=2).sum()
((btc_bellow_200ms[loc_]*1+rsi_above_80[loc_]*1)>=2).sum()
# 1 time when ath indicator is activated. 2 times when ma indicator is activated.
# When ath and rsi is active, we invest 0
# When MA indicator is active, we invest 500 - no matter what
# Trading strategy priority
# 1. MA
# 2. RSI
# 3. ATH

# Define investment
investment = np.repeat(250, loc_.shape[0])
vDates_monthly = vDates_daily[loc_].copy()
btc_price_monthly = btc_price_daily[loc_].copy()
indicator1 = btc_bellow_200ms[loc_].copy()
indicator2 = rsi_above_80[loc_].copy()
indicator3 = btc_ath_indicator[loc_].copy()

investment[indicator3] = 125 # Lowest priority
investment[indicator2] = 0 # Mid priority
investment[indicator1] = 500 # Highest priority

# Backtest this combined strategy
strategy_matrix, Q = B.SimpleBacktest(investment, btc_price_monthly)
B.PlotBacktest(vDates_monthly, strategy_matrix, investment, title="Combined strategy")

# Comapre it with baseline and investment amount
baseline_matrix, _, baseline_investment = B.BaselineTradingMonthly(isPlot=False)
total_return_baseline = f'{(np.nansum(baseline_matrix, axis=0)[-1] / baseline_investment.sum() - 1) * 100:.2f}%'
total_return_combined = f'{(np.nansum(strategy_matrix, axis = 0)[-1]/investment.sum() - 1)*100:.2f}%'

#%% Compare new investment and baseline
plt.plot(vDates_monthly[:,0],np.nansum(strategy_matrix, axis = 0), label = 'Combined Strategy', color = '#1f77b4');
plt.plot(vDates_monthly[:,0],np.nansum(baseline_matrix, axis = 0), label = 'Baseline', color = '#1f77b4', linestyle='--');
plt.plot(vDates_monthly[:,0], np.cumsum(investment), label = 'Investment Combined Strategy', color = '#ff7f0e')
plt.plot(vDates_monthly[:,0], np.cumsum(baseline_investment), label = 'Investment Baseline', color = '#ff7f0e', linestyle='--')
plt.legend()
plt.show()

