import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Load your updated data
df = pd.read_csv("quant_strat_interview/q1_pricing.csv", index_col='Date', parse_dates=True)


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

def calculate_hedge_pnl(df, vol_col, roll_period, r=0.01, q=0.0, hedge_ratio=1.0):
    """
    Calculates the P&L of a rolling put hedge with daily mark-to-market pricing,
    utilizing an external Black-Scholes pricer.
    """
    num_days = len(df)
    hedge_pnl = np.zeros(num_days)
    realized_cash = 0.0  # Tracks our bank account (premiums paid and payoffs received)
    
    i = 0
    while i < num_days:
        # 1. Lock in the option strike on the roll date
        K = df['Strike'].iloc[i]
        
        # Determine the exact expiration index
        expiration_idx = min(i + roll_period, num_days - 1)
        
        # 2. Mark the active option to market every single day until expiration
        for j in range(i, expiration_idx + 1):
            # Calculate remaining time to maturity in years
            T_years = (expiration_idx - j) / 365.0 
            
            S_j = df['SP'].iloc[j]
            sigma_j = df[vol_col].iloc[j] / 100.0
            
            # Call the isolated BS function
            current_opt_value = bs_put_price(S=S_j, K=K, T=T_years, r=r, q=q, sigma=sigma_j)
            
            # If j == i, we are buying the option today
            if j == i:
                premium_paid = current_opt_value * hedge_ratio
                realized_cash -= premium_paid
            
            # Total P&L is our realized cash + the unrealized value of the active option
            hedge_pnl[j] = realized_cash + (current_opt_value * hedge_ratio)
        
        # 3. Handle expiration payoff and roll
        if expiration_idx < num_days - 1:
            payoff = max(K - df['SP'].iloc[expiration_idx], 0) * hedge_ratio
            realized_cash += payoff
            
        i = expiration_idx
        if i == num_days - 1:
            break
            
    return hedge_pnl

# 3. Define your target ratio (e.g., 0.5 for 50% notional protection)
# You can adjust this variable to test different drawdown limits
target_ratio = 1

# Run the simulations (63 days for 3M, 252 days for 1Y)
df['Hedge_Only_90D'] = calculate_hedge_pnl(df, 'Put_Price_90D', 63, hedge_ratio=target_ratio)
df['Hedge_Only_1Y'] = calculate_hedge_pnl(df, 'Put_Price_1Y', 252, hedge_ratio=target_ratio)

# 4. Calculate total portfolio P&L
starting_spx = df['SP'].iloc[0]
df['Long_SPX_PnL'] = df['SP'] - starting_spx

df['Portfolio_90D'] = df['Long_SPX_PnL'] + df['Hedge_Only_90D']
df['Portfolio_1Y'] = df['Long_SPX_PnL'] + df['Hedge_Only_1Y']

# 5. Plotting
df = df[:-1]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top Chart: Hedge Only
ax1.plot(df.index, df['Hedge_Only_90D'], color='red', label=f'3M Hedge Only P&L (Ratio: {target_ratio})')
ax1.plot(df.index, df['Hedge_Only_1Y'], color='orange', label=f'1Y Hedge Only P&L (Ratio: {target_ratio})')
ax1.set_title('Standalone Performance of Put Option Hedges', fontsize=14)
ax1.set_ylabel('P&L (Index Points)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower left')

# Bottom Chart: Total Portfolio
ax2.plot(df.index, df['Long_SPX_PnL'], color='blue', alpha=0.4, label='Unhedged (Long SPX)', linewidth=2)
ax2.plot(df.index, df['Portfolio_90D'], color='green', linewidth=1.5, label=f'Hedged 3M (Ratio: {target_ratio})')
ax2.plot(df.index, df['Portfolio_1Y'], color='purple', linewidth=1.5, label=f'Hedged 1Y (Ratio: {target_ratio})')
ax2.set_title('Portfolio Performance: Unhedged vs. Hedged', fontsize=14)
ax2.set_ylabel('Cumulative P&L (Index Points)')
ax2.set_xlabel('Date')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"quant_strat_interview/q1_pnl_hedge_ratio_{target_ratio}.png")