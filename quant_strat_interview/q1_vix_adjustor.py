import pandas as pd
import numpy as np

# Apply mean reversion model to estimate vix3m and vix1y based on vix

vix = pd.read_excel('quant_strat_interview/q1_vix.xlsx', header=0, engine='openpyxl')
vix = vix.dropna()         # Drop any empty rows
vix['Date'] = pd.to_datetime(vix['Date']) # Ensure it's a date object
vix.set_index('Date', inplace=True)

vix['V0_var'] = (vix['VIX'] / 100)**2

long_term_mean = vix["VIX"].mean(axis=0)
theta = (long_term_mean / 100)**2 
kappa = 2.5                      

T_3m = 3/12
T_1y = 1

def calc_term_structure_vol(V0_var, theta, kappa, T):
    """
    Calculates the implied volatility for a given maturity T using a 
    mean-reverting variance model.
    """
    # Calculate the weight factor based on time and speed of reversion
    weight = (1 - np.exp(-kappa * T)) / (kappa * T)
    
    # Calculate implied integrated variance
    implied_var = theta + (V0_var - theta) * weight
    
    # Convert variance back to an annualized percentage volatility
    implied_vol = np.sqrt(implied_var) * 100
    return implied_vol

vix['Vol_3M'] = calc_term_structure_vol(vix['V0_var'], theta, kappa, T_3m)
vix['Vol_1Y'] = calc_term_structure_vol(vix['V0_var'], theta, kappa, T_1y)

print(vix.head())
vix.to_csv("quant_strat_interview/q1_vix_adjusted.csv")