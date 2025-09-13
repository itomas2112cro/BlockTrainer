#%% Imports
import numpy as np
import pandas as pd
import Baseline as B
import requests
import matplotlib.pyplot as plt
from datetime import date as dt

# coingecko_api = 'CG-JusXxcdLY9yRUzU26WaL1Xdx'

#%% Functions
def convertDateToString(timest):
    month_ = timest.strftime("%m")
    day_ = timest.strftime("%d")
    return(rf"{timest.year}-{month_}-{day_}")

# #%% Get data 2015 - 2018
# start = '2013-01-01'
# end = '2025-09-01'
# all_days = pd.date_range(start=start, end=end, freq='D')
# all_splits = np.array_split(all_days,30)
#
# total_df = pd.DataFrame([])
# counter = 0
# for split in all_splits:
#
#     start_download = convertDateToString(split[0])
#     end_download = convertDateToString(split[-1])
#     url = rf"https://pro-api.coingecko.com/api/v3/coins/bitcoin/ohlc/range?vs_currency=usd&from={start_download}&to={end_download}&interval=daily"
#
#
#     headers = {
#         "accept": "application/json",
#         "x-cg-pro-api-key": "CG-JusXxcdLY9yRUzU26WaL1Xdx"
#     }
#
#     response = requests.get(url, headers=headers)
#     df = pd.DataFrame(response.json(), columns = ['date', 'open', 'high','low', 'close'])
#     df['date'] = pd.to_datetime(df.date, unit='ms')
#     total_df = pd.concat([total_df, df])
#
#     counter += 1
#     print(counter)
#
# total_df.reset_index(drop = True).rename(columns = {' low': 'low'}).to_csv(r'btc_historical_daily_data.csv')
# total_df.set_index('date').close.plot(); plt.show()

#%% Convert data to array
# df = pd.read_csv(r'btc_historical_daily_data.csv', index_col=0)
# df['date'] = pd.to_datetime(df.date)
# vDates = pd.date_range(start = df.date.iloc[0], end=df.date.iloc[-1], freq='D').values.astype('datetime64[D]')
# np_data = np.full((vDates.shape[0],4), np.nan)
# for i in range(df.shape[0]):
#     interm_ = df.iloc[i].copy()
#     loc_ = np.where(vDates == interm_.date)[0][0]
#     np_data[loc_] = interm_[['open', 'high', 'low', 'close']].values
#
# np_data = pd.DataFrame(np_data).ffill().values
# np.save(r'btc_price_array.npy', np_data)
# np.save(r'vDates.npy',vDates)
#
# plt.plot(np_data[:,0]); plt.show()
#
# #%% Get weekly data
# vDates = np.load(r'vDates.npy')
# np_data = np.load(r'btc_price_array.npy')
# vDates_weekly = vDates[::7][1:].copy()
# np_data_weekly = np.full((vDates_weekly.shape[0], 4), np.nan)
# for i in range(np_data_weekly.shape[0]):
#     block = np_data[i*7:((i+1)*7)]
#     open = block[0,0]
#     close = block[-1,3]
#     high = block[:,1].max()
#     low = block[:, 2].min()
#     np_data_weekly[i] = np.array([open, high, low, close])
#
# np.save(r'btc_price_weekly.npy', np_data_weekly)
# np.save(r'vDates_weekly.npy', vDates_weekly)

#%% EMA feature
btc_price_weekly = np.load(r'btc_price_weekly.npy')
vDates_weekly = np.load(r'vDates_weekly.npy')

start = np.where(vDates_weekly>=dt(year = 2015, day = 1, month = 1))[0][0]
end = np.where(vDates_weekly>=dt(year = 2015, day = 1, month = 1))[0][-1]

ema = B.EMA(btc_price_weekly).copy()
ema[:200] = np.nan # As requested
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
ema_indicator = (btc_price_weekly[start:end+1,3] < ema).copy()
ath_indicator = (btc_price_weekly[start:end+1,3] > ATH_weekly).copy()
rsi_indiocator = (RSI >= 80).copy()

#%% Check if indicators interact
((ema_indicator*1 + ath_indicator*1) > 1.5).sum()
((ema_indicator*1 + rsi_indiocator*1) > 1.5).sum()
((ath_indicator*1 + rsi_indiocator*1) > 1.5).sum()

#%% Priority 1. RSI, 2. EMA, 3. ATH
investment = np.repeat(250, btc_price_weekly[start:end+1].shape[0])
investment[ath_indicator] = 125
investment[ema_indicator] = 500
investment[rsi_indiocator] = 0

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

bars = plt.bar(np.unique(parity_investment, return_counts=True)[0].astype('str'), np.unique(parity_investment, return_counts=True)[1]);
plt.bar_label(bars)
plt.title('Investments')
plt.show()


strategy_matrix_parity, Q = B.SimpleBacktest(parity_investment, btc_price_weekly[start:end+1])
B.PlotBacktest(np.expand_dims(vDates_weekly[start:end+1],1), strategy_matrix_parity, parity_investment,"Backtest")

plt.plot(vDates_weekly[start:end+1],skipped_base); plt.show()


plt.plot(vDates_weekly[start:end+1], np.cumsum(parity_investment), label = 'parity');
plt.plot(vDates_weekly[start:end+1], np.cumsum(investment_baseline));
plt.legend(); plt.show()