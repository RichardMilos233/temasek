import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Generate a Realistic Mock ESG Dataset
# ---------------------------------------------------------
# Create 3 years of business day dates
dates = pd.date_range(start='2023-01-01', end='2026-01-01', freq='B')
tickers = ['TECH', 'BANK', 'ENGY']

data = []
np.random.seed(42) # For reproducibility

for ticker in tickers:
    # Simulate ESG scores using a random walk so it looks like real historical data
    # Starting at a base score of 50
    daily_changes = np.random.normal(loc=0.005, scale=0.2, size=len(dates))
    esg_scores = np.clip(np.cumsum(daily_changes) + 50, 0, 100) 
    
    df_ticker = pd.DataFrame({
        'date': dates, 
        'ticker': ticker, 
        'esg_score': esg_scores
    })
    data.append(df_ticker)

df = pd.concat(data).reset_index(drop=True)

# ---------------------------------------------------------
# 2. Run the Momentum & Z-Score Pipeline
# ---------------------------------------------------------
# Sort to prevent look-ahead bias
df = df.sort_values(by=['ticker', 'date'])

# Calculate 1-year (252 trading days) momentum
df['esg_momentum'] = df.groupby('ticker')['esg_score'].diff(periods=252)

# Drop the first year of data which will be NaN due to the lookback period
df = df.dropna(subset=['esg_momentum'])

# Define and apply the cross-sectional Z-score function
def compute_zscore(group):
    std = group.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=group.index)
    return (group - group.mean()) / std

df['signal_zscore'] = df.groupby('date')['esg_momentum'].transform(compute_zscore)

# ---------------------------------------------------------
# 3. View the Results for the Final Day
# ---------------------------------------------------------
latest_date = df['date'].max()
final_day_signals = df[df['date'] == latest_date].copy()

# Round for cleaner display
final_day_signals['esg_score'] = final_day_signals['esg_score'].round(2)
final_day_signals['esg_momentum'] = final_day_signals['esg_momentum'].round(2)
final_day_signals['signal_zscore'] = final_day_signals['signal_zscore'].round(4)

print(final_day_signals.to_string(index=False))