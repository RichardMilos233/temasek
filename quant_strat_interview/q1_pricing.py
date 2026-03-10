import pandas as pd
import numpy as np
from scipy.stats import norm

vix = pd.read_csv('quant_strat_interview/q1_vix_adjusted.csv')
vix['Date'] = pd.to_datetime(vix['Date']) # Ensure it's a date object

sp = pd.read_excel('quant_strat_interview/q1_sp.xlsx', header=0, engine='openpyxl')
sp = sp.dropna()         # Drop any empty rows
sp['Date'] = pd.to_datetime(sp['Date']) # Ensure it's a date object


df = pd.merge(sp, vix, on='Date', how='inner')
df.set_index('Date', inplace=True)



def bs_put_price(S, K, T, r, q, sigma):
    """
    Calculates the Black-Scholes European put price for a single day.
    Returns intrinsic value if the option is at or past expiration.
    """
    if T <= 0:
        return max(K - S, 0)
        
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


r = 0.01               # Fixed 1% p.a. interest rate
q = 0.0                # Fixed 0 dividend yield
moneyness = 0.90       # 90% strike

# Create the parameter columns
df['Strike'] = df['SP'] * moneyness

# Calculate the daily theoretical price
df['Put_Price_90D'] = bs_put_price(
    S=df['SP'], 
    K=df['Strike'], 
    T=90/365.0, 
    r=r, 
    q=q, 
    sigma=df['Vol_3M']/100.0
)


df['Put_Price_1Y'] = bs_put_price(
    S=df['SP'], 
    K=df['Strike'], 
    T=365/365.0, 
    r=r, 
    q=q, 
    sigma=df['Vol_1Y']/100.0
)

print("\n--- Strategy Pricing Data ---")
print(df[['SP', 'VIX', 'Strike', 'Vol_3M', "Vol_1Y", 'Put_Price_90D', 'Put_Price_1Y']])
df.to_csv("quant_strat_interview/q1_pricing.csv")