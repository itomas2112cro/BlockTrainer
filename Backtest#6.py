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

ema_200 = B.EMA(btc_price_weekly).copy()
# ema[:200] = np.nan # As requested
ema_200 = ema_200[start:end+1].copy() ## In this way we don't need to kick out 200 days
plt.plot(vDates_weekly[start:end+1],ema_200); plt.show()