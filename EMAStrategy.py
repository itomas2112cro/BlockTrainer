#%% Imports
import numpy as np
import pandas as pd
import Baseline as B
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats

#%% Simple EMA strategy
vDates_daily = B.vDates[pd.Series(B.vDates[:,0]).dt.hour == 0].copy()
btc_price_daily = B.btc_price[pd.Series(B.vDates[:,0]).dt.hour == 0].copy()
btc_200ma = B.moving_average(btc_price_daily, 200)
btc_bellow_200ms = (btc_price_daily[:,0]<btc_200ma).copy()

# Monthly points
df_dates = pd.DataFrame(vDates_daily, columns=['date', 'date_id'])
df_dates = df_dates[(df_dates['date'] >= dt(year=2018, day=1, month=1)) & (df_dates['date'] <= dt(year=2025, day=1, month=1))].copy()
date_SOM_idxs = df_dates[(df_dates['date'].dt.day == 1) & (df_dates['date'].dt.hour == 0)].date_id.values.astype(int)
loc_ = np.where(np.isin(vDates_daily[:,1], date_SOM_idxs))[0]

# Define investment
investment = np.repeat(250, loc_.shape[0])
vDates_monthly = vDates_daily[loc_].copy()
btc_price_monthly = btc_price_daily[loc_].copy()
indicator = btc_bellow_200ms[loc_].copy()
investment[indicator] = 500

# Backtest this simple strategy
strategy_matrix, Q = B.SimpleBacktest(investment, btc_price_monthly)
B.PlotBacktest(vDates_monthly, strategy_matrix, investment, title='500\$ if BTC<200MA, 250\$ otherwise')

# Compare it relative to the investment amount
baseline_matrix, _, baseline_investment = B.BaselineTradingMonthly(isPlot=False)
B.PlotBacktest(vDates_monthly, baseline_matrix, baseline_investment, 'Baseline')
total_return_baseline = f'{(np.nansum(baseline_matrix, axis=0)[-1] / baseline_investment.sum() - 1) * 100:.2f}%'

total_return_simpleEMA = f'{(np.nansum(strategy_matrix, axis = 0)[-1]/investment.sum() - 1)*100:.2f}%'

# Simple if indicator has predictive power
indicator_on = []
indicator_off = []
for i in range(loc_.shape[0]):
    if indicator[i]:
        indicator_on.append((baseline_matrix[i,i:][-1]/baseline_matrix[i,i:][0])**(1/(baseline_matrix[i,i:].shape[0]))-1)
    else:
        indicator_off.append((baseline_matrix[i, i:][-1] / baseline_matrix[i, i:][0]) ** (1 / (baseline_matrix[i, i:].shape[0])) - 1)

plt.bar(np.array(['Indicator ON', 'Indicator OFF']), np.array([np.mean(indicator_on)*100, np.mean(indicator_off)*100]));
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val}%"))
plt.title("BTC trading above and bellow 200MA")
plt.tight_layout()
plt.show()

# Statistical test for significance in difference in means
t_test, p_val = stats.ttest_ind(np.array(indicator_on), np.array(indicator_off), equal_var = False)
u_stat, p_mw = stats.mannwhitneyu(np.array(indicator_on), np.array(indicator_off), alternative = "two-sided")

#%% Compare new investment and baseline
plt.plot(vDates_monthly[:,0],np.nansum(strategy_matrix, axis = 0), label = 'Simple MA', color = '#1f77b4');
plt.plot(vDates_monthly[:,0],np.nansum(baseline_matrix, axis = 0), label = 'Baseline', color = '#1f77b4', linestyle='--');
plt.plot(vDates_monthly[:,0], np.cumsum(investment), label = 'Investment Simpla MA', color = '#ff7f0e')
plt.plot(vDates_monthly[:,0], np.cumsum(baseline_investment), label = 'Investment Baseline', color = '#ff7f0e', linestyle='--')
plt.legend()
plt.show()



