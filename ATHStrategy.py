#%% imports
import numpy as np
import pandas as pd
import Baseline as B
from datetime import datetime as dt
import matplotlib.pyplot as plt
from scipy import stats


#%% ATH strategy
vDates_daily = B.vDates[pd.Series(B.vDates[:, 0]).dt.hour == 0].copy()
btc_price_daily = B.btc_price[pd.Series(B.vDates[:, 0]).dt.hour == 0].copy()

# Monthly points
df_dates = pd.DataFrame(vDates_daily, columns=['date', 'date_id'])
df_dates = df_dates[(df_dates['date'] >= dt(year=2018, day=1, month=1)) & (df_dates['date'] <= dt(year=2025, day=1, month=1))].copy()
date_SOM_idxs = df_dates[(df_dates['date'].dt.day == 1) & (df_dates['date'].dt.hour == 0)].date_id.values.astype(int)
loc_ = np.where(np.isin(vDates_daily[:,1], date_SOM_idxs))[0]


# Define Investment
investment = np.repeat(250, loc_.shape[0])
vDates_monthly = vDates_daily[loc_]
btc_price_monthly = btc_price_daily[loc_]

# Define ATH indicator
## ATH indicator is active if price at start of the month is higher than ATH price excluding last month
ATH_indicator = B.ATH(B.btc_price[:,1])


indicator = (btc_price_monthly[:,0] > ATH_indicator[vDates_daily[:,1].astype('int')][loc_]).copy()
investment[indicator] = 125


strategy_matrix, Q = B.SimpleBacktest(investment, btc_price_monthly)
B.PlotBacktest(vDates_monthly, strategy_matrix, investment, '125\$ if BTC>ATH, 250\$ otherwise')


baseline_matrix, _, baseline_investment = B.BaselineTradingMonthly(isPlot=False)
B.PlotBacktest(vDates_monthly, baseline_matrix, baseline_investment, 'Baseline')
total_return_baseline = f'{(np.nansum(baseline_matrix, axis=0)[-1]/baseline_investment.sum() -1)*100:.2f}'

total_return_ATH = f'{(np.nansum(strategy_matrix, axis=0)[-1]/investment.sum() -1 )*100:.2f}'

# Test if indicator has perdictive power
indicator_on = []
indicator_off = []
for i in range(loc_.shape[0]):
    if indicator[i]:
        indicator_on.append((baseline_matrix[i, i:][-1]/baseline_matrix[i, i:][0])**(1/baseline_matrix[i, i::].shape[0]) -1)
    else:
        indicator_off.append((baseline_matrix[i, i:][-1]/baseline_matrix[i, i:][0])**(1/baseline_matrix[i, i::].shape[0]) -1)

plt.bar(np.array(['Indicator ON', 'Indicator OFF']), np.array([np.mean(indicator_on)*100, np.mean(indicator_off)*100]))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _:f"{val:.2f}%"))
plt.title("BTC trading above and below ATH")
plt.tight_layout()
plt.show()

# Statistical test for significance in difference in menas
t_test, p_val = stats.ttest_ind(np.array(indicator_on), np.array(indicator_off), equal_var=False)
u_stat, p_mw = stats.mannwhitneyu(np.array(indicator_on), np.array(indicator_off), alternative="two-sided")

#%% Compare new investment and baselien
plt.plot(vDates_monthly[:, 0], np.nansum(strategy_matrix, axis=0), label='Simple ATH', color='#1f77b4')
plt.plot(vDates_monthly[:, 0], np.nansum(baseline_matrix, axis=0), label='Baselien', color='#1f77b4', linestyle='--')
plt.plot(vDates_monthly[:, 0], np.cumsum(investment), label='Investment Simple ATH', color='#ff7f0e')
plt.plot(vDates_monthly[:, 0], np.cumsum(baseline_investment), label='Investment Baseline', color='#ff7f0e', linestyle='--')
plt.legend(); plt.show()


#%% Function for the new ATH
btc_price = B.btc_price.copy()
vDates = B.vDates.copy()
start = dt(year = 2018, day = 1, month = 1, hour = 0)
end = dt(year = 2025, day = 1, month = 1, hour = 0)

vDates_weekly, btc_price_weekly = B.getWeeklyFromHourly(vDates,btc_price,start,end)


# ATH by new definition
ATH_old = B.ATH(btc_price[:,1], lag = 0)
plt.plot(ATH_old); plt.show()

### Create max drawdown at every point in time
### When drawdown >= 50%, we calculate the ATH before that
### When every new 50% drawdown happens, we chech if the ATH recognized is above the previous one.
max_drawdown = []
for i in range(len(btc_price)):
    if btc_price[i,1]:
        max_drawdown.append(-(btc_price[i,1]/np.nanmax(btc_price[:i+1,1])-1))
max_drawdown = np.array(max_drawdown)
plt.plot(max_drawdown)
plt.show()
