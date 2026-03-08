import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your updated data
df = pd.read_csv("quant_strat_interview/q1_pricing.csv", index_col='Date', parse_dates=True)

# 2. Define the reusable backtesting function with hedge_ratio
def calculate_hedge_pnl(df, price_col, roll_period, hedge_ratio=1.0):
    num_days = len(df)
    hedge_pnl = np.zeros(num_days)
    current_cash = 0.0
    
    i = 0
    while i < num_days:
        # Scale the cost of the option by the hedge ratio
        premium_paid = df[price_col].iloc[i] * hedge_ratio
        strike_price = df['Strike'].iloc[i] 
        
        current_cash -= premium_paid
        expiration_idx = min(i + roll_period, num_days - 1)
        
        # Fill the daily P&L flat until expiration
        hedge_pnl[i:expiration_idx] = current_cash
        
        # Calculate exit
        if expiration_idx < num_days - 1:
            settlement_spx = df['SP'].iloc[expiration_idx]
            
            # Scale the payoff by the same hedge ratio
            payoff = max(strike_price - settlement_spx, 0) * hedge_ratio
            
            current_cash += payoff
            hedge_pnl[expiration_idx] = current_cash
            
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