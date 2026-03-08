import numpy as np
import pandas as pd

def track_performance(returns_df):
    """
    returns_df: Output from the implement_hedging_strategy function
    """
    # 1. Calculate Cumulative Returns (Starting from 1.0)
    # Scenario A: Just the Stock (Unhedged)
    returns_df['Cum_Stock'] = (1 + returns_df['Stock']).cumprod()
    
    # Scenario B: Stock + Index Hedge
    returns_df['Return_Index_Hedged'] = returns_df['Stock'] + returns_df['Index_Hedge_PnL']
    returns_df['Cum_Index_Hedged'] = (1 + returns_df['Return_Index_Hedged']).cumprod()
    
    # Scenario C: Stock + Index Hedge + FX Hedge
    returns_df['Cum_Fully_Hedged'] = (1 + returns_df['Total_Hedged_Return']).cumprod()
    
    # 2. Performance Metrics Calculation
    metrics = {}
    for col in ['Stock', 'Return_Index_Hedged', 'Total_Hedged_Return']:
        # Annualized Return (assuming 252 trading days)
        ann_return = returns_df[col].mean() * 252
        # Annualized Volatility
        ann_vol = returns_df[col].std() * np.sqrt(252)
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        sharpe = ann_return / ann_vol
        
        # Max Drawdown
        cum_ret = (1 + returns_df[col]).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        metrics[col] = {
            'Ann. Return': f"{ann_return:.2%}",
            'Ann. Volatility': f"{ann_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.2%}"
        }
    
    return returns_df, pd.DataFrame(metrics).T

# Execute tracking
hedged_returns = pd.read_csv("quant_strat_interview/q2_hedged.csv")
final_df, performance_summary = track_performance(hedged_returns)
print(performance_summary)

import matplotlib.pyplot as plt

def plot_hedging_performance(returns_df):
    # 1. Calculate Cumulative Growth
    cum_stock = (1 + returns_df['Stock']).cumprod()
    cum_index_hedged = (1 + (returns_df['Stock'] + returns_df['Index_Hedge_PnL'])).cumprod()
    cum_fully_hedged = (1 + returns_df['Total_Hedged_Return']).cumprod()

    # 2. Calculate Drawdowns for the sub-plot
    def get_drawdown(cum_rets):
        return (cum_rets / cum_rets.cummax()) - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- Top Plot: Cumulative PnL ---
    ax1.plot(cum_stock, label='Unhedged (Stock 1)', color='gray', alpha=0.5)
    ax1.plot(cum_index_hedged, label='Index Hedged (Beta Adjusted)', color='blue', linewidth=1.5)
    ax1.plot(cum_fully_hedged, label='Fully Hedged (Index + FX)', color='green', linewidth=2)
    
    ax1.set_title('Strategy PnL Comparison: Hedged vs. Unhedged', fontsize=14)
    ax1.set_ylabel('Cumulative Return (Base 1.0)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Bottom Plot: Drawdowns ---
    ax2.fill_between(returns_df.index, get_drawdown(cum_stock), color='gray', alpha=0.2, label='Stock DD')
    ax2.plot(get_drawdown(cum_fully_hedged), color='green', label='Fully Hedged DD')
    
    ax2.set_title('Drawdown Analysis (Risk Reduction)', fontsize=12)
    ax2.set_ylabel('Drawdown %')
    ax2.set_ylim(-0.5, 0.05) # Focus on the drops
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

plot_hedging_performance(hedged_returns)