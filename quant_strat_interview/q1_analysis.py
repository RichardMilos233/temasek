import matplotlib.pyplot as plt
import pandas as pd
from q1_rolling_hedge import calculate_rebalanced_portfolio



df = pd.read_csv("quant_strat_interview/q1_pricing.csv", index_col='Date', parse_dates=True)
# Define target ratio (e.g., 1.0 means 1 put per 1 share of SPX)
target_ratio = 0.5

# Initial capital. Setting it to the starting SPX price makes it easy to compare 
# against the raw index, effectively starting both at the same baseline value.
initial_capital = df['SP'].iloc[0]

# Calculate Unhedged Portfolio Value (Buy and Hold)
shares_unhedged = initial_capital / df['SP'].iloc[0]
df['Unhedged_Value'] = shares_unhedged * df['SP']

# Run the Rebalanced Portfolio simulations
df['Hedged_3M_Value'] = calculate_rebalanced_portfolio(
    df, 'Vol_3M', 63, initial_capital, hedge_ratio=target_ratio
)
df['Hedged_1Y_Value'] = calculate_rebalanced_portfolio(
    df, 'Vol_1Y', 252, initial_capital, hedge_ratio=target_ratio
)
# --- PLOTTING ---
df_plot = df[:-1] # Drop last row if incomplete
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(df_plot.index, df_plot['Unhedged_Value'], color='blue', alpha=0.4, label='Unhedged (100% SPX)', linewidth=2)
ax.plot(df_plot.index, df_plot['Hedged_3M_Value'], color='green', linewidth=1.5, label=f'Hedged 3M (Ratio: {target_ratio})')
ax.plot(df_plot.index, df_plot['Hedged_1Y_Value'], color='purple', linewidth=1.5, label=f'Hedged 1Y (Ratio: {target_ratio})')

ax.set_title('Rebalanced Portfolio Value: Unhedged vs. Put Hedge', fontsize=14)
ax.set_ylabel('Total Portfolio Value ($)')
ax.set_xlabel('Date')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"quant_strat_interview/q1_portfolio_value_ratio_{target_ratio}.png")


# --- MAX DRAWDOWN ANALYSIS ---
print("\n" + "="*40)
print("QUANTITATIVE RISK ANALYSIS: MAX DRAWDOWN")
print("="*40)

def get_max_drawdown(value_series):
    """Calculates the Maximum Drawdown as a percentage."""
    rolling_max = value_series.cummax()
    drawdown = (value_series - rolling_max) / rolling_max
    return drawdown.min()

crisis_periods = {
    "2008 Global Financial Crisis (Oct 2007 - Mar 2009)": ('2007-10-01', '2009-03-31'),
    "2020 COVID Crash (Feb 2020 - Apr 2020)": ('2020-02-01', '2020-04-30')
}

for name, (start_date, end_date) in crisis_periods.items():
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_period = df.loc[mask]
    
    if not df_period.empty:
        print(f"\n{name}:")
        print(f"  Unhedged Max Drawdown: {get_max_drawdown(df_period['Unhedged_Value']):.2%}")
        print(f"  Hedged 3M Max Drawdown : {get_max_drawdown(df_period['Hedged_3M_Value']):.2%}")
        print(f"  Hedged 1Y Max Drawdown : {get_max_drawdown(df_period['Hedged_1Y_Value']):.2%}")

print("\nOverall Timeline (Full Backtest):")
print(f"  Unhedged Max Drawdown: {get_max_drawdown(df['Unhedged_Value']):.2%}")
print(f"  Hedged 3M Max Drawdown : {get_max_drawdown(df['Hedged_3M_Value']):.2%}")
print(f"  Hedged 1Y Max Drawdown : {get_max_drawdown(df['Hedged_1Y_Value']):.2%}")