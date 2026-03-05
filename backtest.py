import pandas as pd
import numpy as np

def run_esg_backtest(returns_df: pd.DataFrame, esg_scores_df: pd.DataFrame, lag_periods: int = 1, tc_bps: float = 0.001) -> pd.Series:
    """
    Simulates a long-short ESG strategy.
    
    Parameters:
    - returns_df: DataFrame of daily asset returns (Index: Dates, Columns: Tickers)
    - esg_scores_df: DataFrame of raw ESG scores (Index: Dates, Columns: Tickers)
    - lag_periods: Number of periods to lag the ESG signal to prevent look-ahead bias
    - tc_bps: Transaction cost in basis points (e.g., 0.001 = 10 bps)
    """
    
    # ---------------------------------------------------------
    # 1. Signal Generation & Bias Prevention
    # ---------------------------------------------------------
    # Shift the ESG scores forward to ensure we only trade on publicly available data.
    # If the data is monthly, lag_periods=1 means a 1-month lag.
    lagged_esg = esg_scores_df.shift(lag_periods)
    
    # Cross-sectional ranking: For each date, rank stocks from 0 (worst) to 1 (best)
    # axis=1 ensures we rank across the tickers on each specific day
    esg_ranks = lagged_esg.rank(axis=1, pct=True, ascending=True)
    
    # ---------------------------------------------------------
    # 2. Portfolio Construction (Target Weights)
    # ---------------------------------------------------------
    # Define our quantiles: Long the top 20%, Short the bottom 20%
    long_condition = esg_ranks >= 0.80
    short_condition = esg_ranks <= 0.20
    
    # Initialize an empty weights dataframe with the same shape as our returns
    weights = pd.DataFrame(0, index=esg_ranks.index, columns=esg_ranks.columns)
    
    # Calculate the number of assets in the long and short legs for equal weighting
    # (This dynamically handles missing data/delisted stocks on any given day)
    num_longs = long_condition.sum(axis=1)
    num_shorts = short_condition.sum(axis=1)
    
    # Apply weights: 1/N for longs, -1/N for shorts. 
    # Use np.where to broadcast the conditions across the dataframe
    weights = weights.mask(long_condition, 1.0 / num_longs, axis=0)
    weights = weights.mask(short_condition, -1.0 / num_shorts, axis=0)
    
    # Replace any potential NaNs or infinite values with 0
    weights = weights.fillna(0).replace([np.inf, -np.inf], 0)
    
    # ---------------------------------------------------------
    # 3. Return Calculation & Execution Simulation
    # ---------------------------------------------------------
    # Shift weights by 1 period because weights determined at the end of day t-1 
    # generate returns during day t.
    actual_weights = weights.shift(1)
    
    # Calculate gross daily returns (element-wise multiplication, then sum across tickers)
    gross_returns = (actual_weights * returns_df).sum(axis=1)
    
    # Calculate daily turnover to estimate transaction costs
    # Turnover is the absolute change in weights from t-1 to t
    turnover = actual_weights.diff().abs().sum(axis=1)
    transaction_costs = turnover * tc_bps
    
    # Calculate net returns
    net_returns = gross_returns - transaction_costs
    
    return net_returns

# Example usage to calculate cumulative return:
# strategy_returns = run_esg_backtest(returns, esg_scores)
# cumulative_returns = (1 + strategy_returns).cumprod() - 1