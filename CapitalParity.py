#%% Imports
import numpy as np
import pandas as pd


#%% If signal says to invest less, we accumulate the value. When signal starts saying to invest more, we invest even more but maximum 2x the investment value (or whatever is left in accumulation variable). Idea is to invest roughly the same over time - as in base example
strategy_matrix

strategy_value = np.nansum(strategy_matrix, axis = 0).copy()
oldInvestmnet = np.diag(strategy_matrix)

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
            skipped_base -= extra + (oldInvestmnet[i] - 250)
            capitalParityInvestment.append(oldInvestmnet[i] + extra)
            skipped_base_list.append(skipped_base)

    return(capitalParityInvestment, skipped_base_list)

plt.plot(vDates_weekly[start:end+1],skipped_base_list); plt.tight_layout(); plt.show()