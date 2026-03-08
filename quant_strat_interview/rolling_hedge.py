import pandas as pd
import numpy as np

# 1. Load your data
df = pd.read_csv("quant_strat_interview/q1_pricing.csv", index_col='Date', parse_dates=True)

# 2. Setup Backtest Parameters
roll_period = 63  # Approx. number of trading days in 3 months
num_days = len(df)

# We will track our cumulative hedge P&L daily so we can plot it later
hedge_pnl = np.zeros(num_days)
current_cash = 0.0

# 3. Run the Rolling Hedge Loop
i = 0
while i < num_days:
    # --- ENTER THE TRADE ---
    premium_paid = df['Put_Price_90D'].iloc[i]
    strike_price = df['Strike'].iloc[i]
    
    # Deduct premium from our cash balance
    current_cash -= premium_paid
    
    # Find the expiration day (or the last day of our dataset if we run out of data)
    expiration_idx = min(i + roll_period, num_days - 1)
    
    # Fill the daily P&L for the holding period (step-function: cash stays flat until expiration)
    hedge_pnl[i:expiration_idx] = current_cash
    
    # --- EXIT THE TRADE (EXPIRATION) ---
    if expiration_idx < num_days - 1: # Normal expiration
        settlement_spx = df['SP'].iloc[expiration_idx]
        
        # Calculate payoff: max(Strike - SPX, 0)
        payoff = max(strike_price - settlement_spx, 0)
        
        # Add payoff to our cash balance
        current_cash += payoff
        hedge_pnl[expiration_idx] = current_cash
        
    # Move our index forward to the expiration day to roll the contract
    i = expiration_idx
    
    # Break if we've reached the end of the dataset
    if i == num_days - 1:
        break

# 4. Add the results back to the DataFrame
df['Hedge_Only_PnL'] = hedge_pnl

# Calculate the Long S&P 500 P&L (Relative to the starting price)
# We track point-based P&L so it matches the option P&L scale
starting_spx = df['SP'].iloc[0]
df['Long_SPX_PnL'] = df['SP'] - starting_spx

# Calculate the combined Portfolio P&L
df['Index_plus_Hedge_PnL'] = df['Long_SPX_PnL'] + df['Hedge_Only_PnL']

print("Backtest complete! Dataframe updated with P&L columns.")
print(df.head())
print(df[:-1].shape)
df = df[:-1]

import matplotlib.pyplot as plt

# Ensure your index is set to datetime for proper x-axis formatting
# (This should already be the case if you used parse_dates=True earlier)

# Create a figure with 2 subplots (stacked vertically)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Top Subplot: The Hedge Only P&L ---
ax1.plot(df.index, df['Hedge_Only_PnL'], color='red', label='Hedge Only P&L (Rolling 90% Puts)')
ax1.set_title('Standalone Performance of the Put Option Hedge', fontsize=14)
ax1.set_ylabel('P&L (Index Points)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left')

# --- Bottom Subplot: Hedged vs. Unhedged Portfolio ---
ax2.plot(df.index, df['Long_SPX_PnL'], color='blue', alpha=0.6, label='Unhedged (Long SPX)')
ax2.plot(df.index, df['Index_plus_Hedge_PnL'], color='green', linewidth=1.5, label='Hedged (SPX + Puts)')
ax2.set_title('Portfolio Performance: Hedged vs. Unhedged', fontsize=14)
ax2.set_ylabel('Cumulative P&L (Index Points)')
ax2.set_xlabel('Date')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper left')

# Adjust layout and display
plt.tight_layout()
# plt.show()
plt.savefig("quant_strat_interview/q1_pnl.png")