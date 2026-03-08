import pandas as pd
import numpy as np
from scipy.stats import norm

vix = pd.read_excel('quant_strat_interview/q1_vix.xlsx', header=0, engine='openpyxl')
vix = vix.dropna()         # Drop any empty rows
vix['Date'] = pd.to_datetime(vix['Date']) # Ensure it's a date object

sp = pd.read_excel('quant_strat_interview/q1_sp.xlsx', header=0, engine='openpyxl')
sp = sp.dropna()         # Drop any empty rows
sp['Date'] = pd.to_datetime(sp['Date']) # Ensure it's a date object


df = pd.merge(sp, vix, on='Date', how='inner')
df.set_index('Date', inplace=True)

def vectorized_bs_put(S, K, T, r, q, sigma):
    """
    Calculates Black-Scholes European put prices for an entire pandas Series natively.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_prices


r = 0.01               # Fixed 1% p.a. interest rate
q = 0.0                # Fixed 0 dividend yield
T = 90 / 365.0         # 3-month maturity in years
moneyness = 0.90       # 90% strike

# Create the parameter columns
df['Strike'] = df['SP'] * moneyness
df['Vol_Proxy'] = df['VIX'] / 100.0  # Using VIX directly as the vol proxy

# Calculate the daily theoretical price
df['Put_Price_90D'] = vectorized_bs_put(
    S=df['SP'], 
    K=df['Strike'], 
    T=T, 
    r=r, 
    q=q, 
    sigma=df['Vol_Proxy']
)

print("\n--- Strategy Pricing Data ---")
print(df[['SP', 'VIX', 'Strike', 'Vol_Proxy', 'Put_Price_90D']])
df.to_csv("quant_strat_interview/q1_pricing.csv")