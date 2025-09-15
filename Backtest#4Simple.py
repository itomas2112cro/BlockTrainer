#%% Imports
import numpy as np
import pandas as pd
import Baseline as B
import requests
import matplotlib.pyplot as plt
from datetime import date as dt

#%% EMA feature
btc_price_weekly = np.load(r'btc_price_weekly.npy')
vDates_weekly = np.load(r'vDates_weekly.npy')

start = np.where(vDates_weekly>=dt(year = 2015, day = 5, month = 1))[0][0]
end = np.where(vDates_weekly>=dt(year = 2025, day = 1, month = 9))[0][0]

ema = B.EMA(btc_price_weekly).copy()
# ema[:200] = np.nan # As requested
ema = ema[start:end+1].copy() ## In this way we don't need to kick out 200 days
plt.plot(vDates_weekly[start:end+1],ema); plt.show()

#%% ATH feature // Firstly define ATH
# btc_price_daily = np.load(r'btc_price_array.npy')
# vDates = np.load(r'vDates.npy')
#
# max_drawdown = []
# high_price = []
# for i in range(btc_price_daily.shape[0]):
#     if i>0:
#         block = btc_price_daily[:i,3]
#         max_drawdown.append(block[-1]/np.max(block) - 1)
#         high_price.append(np.max(block))
#     else:
#         max_drawdown.append(0)
# max_drawdown = -np.array(max_drawdown)
# high_price = np.array(high_price)
#
# matrix_ATH = np.full((vDates.shape[0], vDates.shape[0]), np.nan)
# for i in np.where(max_drawdown>=0.5)[0]:
#     if high_price[i] > btc_price_daily[i,3]:
#         matrix_ATH[i, i:] = high_price[i]
#
# ATH = np.nanmax(matrix_ATH, axis = 0)
# ATH_weekly = ATH[::7][1:].copy()

# uniqs = np.unique(np.nanmax(matrix_ATH, axis = 0))
# date1 = np.where(np.nanmax(matrix_ATH, axis = 0) == uniqs[0])[0][0]
# date2 = np.where(np.nanmax(matrix_ATH, axis = 0) == uniqs[1])[0][0]
# date3 = np.where(np.nanmax(matrix_ATH, axis = 0) == uniqs[2])[0][0]
#
# vDates[date1]
# vDates[date2]
# vDates[date3]

### Hard-code ATH features
vDates = vDates = np.load(r'vDates.npy')
date1 = dt(year=2013, day = 30, month = 11)
date2 = dt(year=2017,day=17,month=12)
date3 = dt(year=2021,day=10,month=11)
ATH = np.full((vDates.shape[0]), np.nan)

cycle1 = np.where((vDates>=date1) & (vDates<date2))[0]
cycle2 = np.where((vDates>=date2) & (vDates<date3))[0]
cycle3 = np.where(vDates>=date3)
ATH[cycle1] = 1166
ATH[cycle2] = 19783
ATH[cycle3] = 67617

btc_price_daily = np.load(r'btc_price_array.npy')
plt.plot(vDates,btc_price_daily[:,3], label='BTC Price')
plt.plot(vDates,ATH, color='r', alpha=0.5, label='ATH')
plt.legend()
plt.show()

### Confirm ATH after 365 days
skip_days = 180
ATH[skip_days:] = ATH[:-skip_days]

ATH_weekly = ATH[::7][start:end+1].copy()
#%% RSI
RSI = B.RSI(btc_price_weekly).copy()
RSI = RSI[start:end+1].copy()
plt.plot(vDates_weekly[start:end+1],RSI); plt.show()


#%% Define indicators
rsi_indicator = ((RSI >= 80)).copy()
ema_indicator = (btc_price_weekly[start:end+1,3] < ema).copy()
ath_indicator = (btc_price_weekly[start:end+1,3] > ATH_weekly).copy()


#%% Check if indicators interact
((ema_indicator*1 + ath_indicator*1) > 1.5).sum()
((ema_indicator*1 + rsi_indicator*1) > 1.5).sum()
((ath_indicator*1 + rsi_indicator*1) > 1.5).sum()

#%% Priority 1. RSI, 2. EMA, 3. ATH
investment = np.repeat(250, btc_price_weekly[start:end+1].shape[0])
investment[ath_indicator] = 125
investment[ema_indicator] = 500
investment[rsi_indicator] = 0

bars = plt.bar(np.unique(investment, return_counts=True)[0].astype('str'), np.unique(investment, return_counts=True)[1]);
plt.bar_label(bars)
plt.title('Investments')
plt.show()

#%% Backtest
strategy_matrix, Q = B.SimpleBacktest(investment, btc_price_weekly[start:end+1])
B.PlotBacktest(np.expand_dims(vDates_weekly[start:end+1],1), strategy_matrix, investment,"Backtest")

#%% Baseline
investment_baseline = np.repeat(250, btc_price_weekly[start:end+1].shape[0])
strategy_matrix_baseline, _ = B.SimpleBacktest(investment_baseline, btc_price_weekly[start:end+1])
B.PlotBacktest(np.expand_dims(vDates_weekly[start:end+1],1), strategy_matrix_baseline, investment_baseline,"Baseline")

#%% Compare new investment and baseline
plt.plot(vDates_weekly[start:end+1],np.nansum(strategy_matrix, axis = 0), label = 'New Strategy', color = '#1f77b4');
plt.plot(vDates_weekly[start:end+1],np.nansum(strategy_matrix_baseline, axis = 0), label = 'Baseline', color = '#1f77b4', linestyle='--', alpha = 0.5);
plt.plot(vDates_weekly[start:end+1], np.cumsum(investment), label = 'Investment New Strategy', color = '#ff7f0e')
plt.plot(vDates_weekly[start:end+1], np.cumsum(investment_baseline), label = 'Investment Baseline', color = '#ff7f0e', linestyle='--', alpha = 0.5)
plt.legend()
plt.tight_layout()
plt.show()

#%% Max drawdown
md_enhanced = B.max_drawdown(strategy_matrix)
md_baseline = B.max_drawdown(strategy_matrix_baseline)

# plt.plot(vDates_weekly[start:end+1][:207+1], strategy_value[:207+1]); plt.tight_layout(); plt.show()

#%% Plot of the BTC agains EMA
ema = B.EMA(btc_price_weekly).copy()
ema_with_nan = ema.copy()
ema_with_nan[:200] = np.nan # As requested
plt.plot(vDates_weekly[start:start+100],ema[start:start+100], label = 'EMA');
plt.plot(vDates_weekly[start:start+100],btc_price_weekly[start:start+100,3], label = 'BTC');
plt.legend()
plt.show()

#%% Capital parity results
parity_investment, skipped_base = B.getCapitalParityInvestment(investment, multiplier=1)

bars = plt.bar((pd.Series(np.unique(parity_investment, return_counts=True)[0].astype('str'))+'$').values, np.unique(parity_investment, return_counts=True)[1]);
plt.bar_label(bars)
plt.title('Investment counts')
plt.show()


strategy_matrix_parity, Q = B.SimpleBacktest(parity_investment, btc_price_weekly[start:end+1])
B.PlotBacktest(np.expand_dims(vDates_weekly[start:end+1],1), strategy_matrix_parity, parity_investment,"Backtest")

plt.plot(vDates_weekly[start:end+1],skipped_base); plt.show()


#PLOT
fig, ax1 = plt.subplots()

# left y-axis
show_size = 52*4
ax1.plot(vDates_weekly[start:end+1][:show_size], np.cumsum(parity_investment)[:show_size], label='parity')
ax1.plot(vDates_weekly[start:end+1][:show_size], np.cumsum(investment_baseline)[:show_size], label='baseline')
ax1.set_ylabel("Cumulative Investments")

# right y-axis
ax2 = ax1.twinx()
ax2.plot(vDates_weekly[start:end+1][:show_size], np.nansum(strategy_matrix_parity, axis = 0)[:show_size], color='orange', label='BTC price')
ax2.plot(vDates_weekly[start:end+1][:show_size], np.nansum(strategy_matrix_baseline, axis = 0)[:show_size], color='orange', linestyle='--', label='BTC price')
ax2.set_ylabel("BTC Price")

# legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.legend(lines1 , labels1, loc='upper left')
fig.tight_layout()
plt.show()

#%% Compare new investment and baseline
plt.plot(vDates_weekly[start:end+1],np.nansum(strategy_matrix_parity, axis = 0), label = 'New Strategy', color = '#1f77b4');
plt.plot(vDates_weekly[start:end+1],np.nansum(strategy_matrix_baseline, axis = 0), label = 'Baseline', color = '#1f77b4', linestyle='--', alpha = 0.5);
plt.plot(vDates_weekly[start:end+1], np.cumsum(parity_investment), label = 'Investment New Strategy', color = '#ff7f0e')
plt.plot(vDates_weekly[start:end+1], np.cumsum(investment_baseline), label = 'Investment Baseline', color = '#ff7f0e', linestyle='--', alpha = 0.5)
plt.legend()
plt.show()

#%% IRR
import numpy_financial as npf
def irr_with_variable_contribs(contribs, final_value):
    """
    contribs: list of weekly contributions (positive outflow values, e.g. 250, 300, 200, ...)
    final_value: the lump sum received at the end (inflow)
    """
    # Convert contributions to negative cash flows
    cashflows = [-c for c in contribs]
    # Add final inflow at the end
    cashflows.append(final_value)

    r_week = npf.irr(cashflows)          # weekly IRR
    r_annual = (1 + r_week) ** 52 - 1    # annualized IRR
    return r_week, r_annual

def calculate_IRR(contribs, final_value):
    contribs = investment_baseline
    final_value = np.nansum(strategy_matrix_baseline, axis=0)[-1]
    _, r_a = irr_with_variable_contribs(contribs, final_value)
    return(r_a)

#%% Max drawdown
md_enhanced = B.max_drawdown(strategy_matrix_parity)
md_baseline = B.max_drawdown(strategy_matrix_baseline)

print(f"Return: {(np.nansum(strategy_matrix_parity, axis = 0)[-1]/np.cumsum(parity_investment)[-1]-1)*100:.2f}")
print(f"CAGR: {((np.nansum(strategy_matrix_parity, axis = 0)[-1]/np.cumsum(parity_investment)[-1])**(1/10)-1)*100:.2f}")
print(f"Notional: {(np.nansum(strategy_matrix_parity, axis = 0)[-1]):.2f}")
print(f"Investment: {(np.cumsum(parity_investment)[-1]):.2f}")
print(f"BTC Units: {(np.nansum(strategy_matrix_parity, axis = 0)[-1]/btc_price_weekly[-1,3]):.2f}")
print(f"Drawdown: {(md_enhanced*100):.2f}")
print(f"IRR: {calculate_IRR(parity_investment, np.nansum(strategy_matrix_parity, axis = 0)[-1])*100:.2f}")
np.round(np.array([55,31,431,1,39])/557*100,2)
#%% Get calendar quarter barier
loc_quarters = np.arange(vDates_weekly[start:end+1].shape[0])[::13]

counter = 0
for i in range(loc_quarters.shape[0] - 1):
    if parity_investment[loc_quarters[i]:loc_quarters[i+1]].sum() >= 250*13:
        counter+=1
        print(parity_investment[loc_quarters[i]:loc_quarters[i+1]].sum())
        print(i)

# baseline = 13*250
# unused = 0
# new_investment = parity_investment.copy()
# for i in range(len(loc_quarters) - 1):
#
#     barier = baseline + unused
#     total_quarterly_investment = parity_investment[loc_quarters[i]:loc_quarters[i+1]].sum()
#
#     if total_quarterly_investment < baseline:
#         unused += (baseline - total_quarterly_investment)
#     if total_quarterly_investment > baseline:
#         unused -= (total_quarterly_investment - baseline)
#
#     if total_quarterly_investment > barier:
#         print("Reinvest")
#         print(i)
#         unused = 0
#         cumsum = parity_investment[loc_quarters[i]:loc_quarters[i+1]].cumsum()
#         loc_first_above = np.where(cumsum>barier)[0][0]
#         left = cumsum[loc_first_above] - barier
#
#         interm_ = parity_investment[loc_quarters[i]:loc_quarters[i+1]].copy()
#         interm_[loc_first_above] = left
#
#         interm_[loc_first_above+1:] = 0
#
#         new_investment[loc_quarters[i]:loc_quarters[i+1]] = interm_.copy()
#
#     # print(unused)
#     i+=1
#
# np.arange(vDates_weekly.shape[0])[::13]

#%% Add two new things to report: Weekly difference VS baseline & Weeks capped & Quarterly budget usage
plt.plot(vDates_weekly[start:end+1],parity_investment - investment_baseline); plt.title("Investmnet Enhanced - Investmnet Baseline"); plt.show()

(parity_investment == 500).sum()