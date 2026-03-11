import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def implement_hedging_strategy(df, lookback=180, rebalance_freq=21, initial_capital=1_000_000):
    """
    df: DataFrame containing columns 'Stock 1', 'FTSE 100', 'GBPUSD'
    lookback: Window for calculating rolling beta (e.g., 180 days)
    rebalance_freq: How often to recalculate beta and rebalance (e.g., 21 days for monthly)
    initial_capital: Starting capital used to size the stock and hedge notionals
    """
    prices = df.set_index('Date').rename(columns={'Stock 1': 'Stock', 'FTSE 100': 'Index', 'GBPUSD': 'FX'})

    # Keep original prices and add daily pct-change return columns for modeling.
    data = prices.copy()
    data['Ret_Stock'] = data['Stock'].pct_change()
    data['Ret_Index'] = data['Index'].pct_change()
    data['Ret_FX'] = data['FX'].pct_change()
    data = data.dropna(subset=['Ret_Stock', 'Ret_Index', 'Ret_FX']).copy()
    print(data[['Ret_Stock', 'Ret_Index', 'Ret_FX']].head())
    
    # 1. Periodic Beta Calculation
    # Calculate beta only on rebalance dates to avoid noisy daily turnover
    rebalance_dates = data.index[lookback::rebalance_freq]
    
    beta_index = pd.Series(index=data.index, dtype=float)
    beta_fx = pd.Series(index=data.index, dtype=float)
    
    for date in rebalance_dates:
        end_loc = data.index.get_loc(date)
        start_loc = end_loc - lookback
        
        y = data['Ret_Stock'].iloc[start_loc:end_loc]
        X = data[['Ret_Index', 'Ret_FX']].iloc[start_loc:end_loc]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        beta_index.loc[date] = model.params['Index']
        beta_fx.loc[date] = model.params['FX']
        
    # Forward-fill betas for the days between rebalances
    data['Beta_Index'] = beta_index.ffill()
    data['Beta_FX'] = beta_fx.ffill()
    
    # Drop initial lookback period
    data = data.dropna(subset=['Beta_Index', 'Beta_FX']).copy()
    
    # 2. Compounding Stock Value & Dynamic Hedging
    # Use the original stock price path scaled to the chosen initial capital.
    data['Stock_Value'] = initial_capital * data['Stock'] / data['Stock'].iloc[0]
    
    # Hedge exposure for day t is based on the stock value at the end of day t-1
    prev_stock_value = data['Stock_Value'].shift(1).fillna(initial_capital)
    
    # Calculate daily Dollar PnL based on the configured initial capital.
    data['Stock_PnL'] = prev_stock_value * data['Ret_Stock']
    data['Index_Hedge_PnL'] = -data['Beta_Index'] * prev_stock_value * data['Ret_Index']
    data['FX_Hedge_PnL'] = -data['Beta_FX'] * prev_stock_value * data['Ret_FX']
    
    data['Total_Hedged_PnL'] = data['Stock_PnL'] + data['Index_Hedge_PnL'] + data['FX_Hedge_PnL']
    
    return data

def track_performance(returns_df, lookback, initial_capital=1_000_000):
    """
    Calculates accurate performance metrics using the dollar PnL method.
    """
    # 1. Calculate Portfolio Values
    returns_df['Cum_Stock'] = returns_df['Stock_Value']
    
    returns_df['Total_Index_Hedged_PnL'] = returns_df['Stock_PnL'] + returns_df['Index_Hedge_PnL']
    returns_df['Cum_Index_Hedged'] = initial_capital + returns_df['Total_Index_Hedged_PnL'].cumsum()
    
    returns_df['Cum_Fully_Hedged'] = initial_capital + returns_df['Total_Hedged_PnL'].cumsum()
    
    # 2. Calculate Daily Percentage Returns for Vol/Sharpe
    returns_df['Return_Stock'] = returns_df['Ret_Stock']
    returns_df['Return_Index_Hedged'] = returns_df['Total_Index_Hedged_PnL'] / returns_df['Cum_Index_Hedged'].shift(1).fillna(initial_capital)
    returns_df['Return_Fully_Hedged'] = returns_df['Total_Hedged_PnL'] / returns_df['Cum_Fully_Hedged'].shift(1).fillna(initial_capital)
    
    # 3. Performance Metrics
    metrics = {}
    cols_to_evaluate = {
        'Unhedged Stock': 'Return_Stock',
        'Index Hedged': 'Return_Index_Hedged', 
        'Fully Hedged': 'Return_Fully_Hedged'
    }
    
    for label, col in cols_to_evaluate.items():
        ann_return = returns_df[col].mean() * 252
        ann_vol = returns_df[col].std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        metrics[label] = {'Ann. Return': ann_return, 'Ann. Volatility': ann_vol, 'Sharpe Ratio': sharpe}
        
    print(pd.DataFrame(metrics).T)

    # 4. Plotting
    def get_drawdown(cum_rets):
        return (cum_rets / cum_rets.cummax()) - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(returns_df['Cum_Stock'], label='Unhedged Stock', color='gray', alpha=0.5)
    ax1.plot(returns_df['Cum_Index_Hedged'], label='Index Hedged', color='blue', linewidth=1.5)
    ax1.plot(returns_df['Cum_Fully_Hedged'], label='Fully Hedged', color='green', linewidth=2)
    ax1.set_title('Strategy PnL: Dynamic Rebalancing', fontsize=14)
    ax1.set_ylabel('Cumulative Value (Base 1.0)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.plot(returns_df.index, get_drawdown(returns_df['Cum_Stock']), color='gray', alpha=0.5)
    ax2.plot(returns_df.index, get_drawdown(returns_df['Cum_Index_Hedged']), color='blue', alpha=1)
    ax2.plot(returns_df.index, get_drawdown(returns_df['Cum_Fully_Hedged']), color='green', alpha=1)
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"quant_strat_interview/q2_corrected_lookback_{lookback}.png")

    return returns_df

df = pd.read_csv("quant_strat_interview/q2.csv", header=0, parse_dates=["Date"])
hedged_df = implement_hedging_strategy(df, lookback=180, rebalance_freq=21)
# track_performance(hedged_df, lookback=180)