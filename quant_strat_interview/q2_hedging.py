import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def implement_hedging_strategy(df, lookback=60):
    """
    df: DataFrame containing columns 'Stock 1', 'FTSE 100', 'GBPUSD'
    lookback: Window for calculating rolling beta 
    """
    # 1. Ensure Date is index and calculate daily returns
    returns = df.set_index('Date').pct_change().dropna()
    
    # Rename for easier internal handling
    returns = returns.rename(columns={
        'Stock 1': 'Stock',
        'FTSE 100': 'Index',
        'GBPUSD': 'FX'
    })
    
    hedge_ratios_index = []
    hedge_ratios_fx = []
    
    # 2. Rolling Beta Calculation (Multiple Linear Regression)
    for i in range(len(returns)):
        if i < lookback:
            hedge_ratios_index.append(np.nan)
            hedge_ratios_fx.append(np.nan)
            continue
            
        # Get historical window
        y = returns['Stock'].iloc[i-lookback:i]
        X = returns[['Index', 'FX']].iloc[i-lookback:i]
        X = sm.add_constant(X) # Include intercept for OLS
        
        model = sm.OLS(y, X).fit()
        
        # Extract partial betas for Index and FX
        hedge_ratios_index.append(model.params['Index'])
        hedge_ratios_fx.append(model.params['FX'])
        
    returns['Beta_Index'] = hedge_ratios_index
    returns['Beta_FX'] = hedge_ratios_fx
    
    # 3. Calculate Performance
    # Performance of the long stock position
    returns['Stock_PnL'] = returns['Stock']
    
    # Performance of the Short Index Hedge
    returns['Index_Hedge_PnL'] = -returns['Beta_Index'] * returns['Index']
    
    # Performance of the Dynamic FX Hedge
    returns['FX_Hedge_PnL'] = -returns['Beta_FX'] * returns['FX'] 
    
    # 4. Total Combined Daily Return
    returns['Total_Hedged_Return'] = (
        returns['Stock_PnL'] + 
        returns['Index_Hedge_PnL'] + 
        returns['FX_Hedge_PnL']
    )
    
    # Drop the NaN rows caused by the lookback period
    return returns.dropna()

df = pd.read_csv("quant_strat_interview/q2.csv", header=0, parse_dates=["Date"])

lookback = 10
hedged_df = implement_hedging_strategy(df, lookback)

def track_performance(returns_df, lookback):
    """
    returns_df: Output from the implement_hedging_strategy function
    """
    # 1. Calculate the Unhedged USD Return for apples-to-apples baseline
    returns_df['Stock_USD_Return'] = (1 + returns_df['Stock']) * (1 + returns_df['FX']) - 1
    
    # Calculate Cumulative Returns (Starting from 1.0)
    # Scenario A: Unhedged Stock in USD
    returns_df['Cum_Stock'] = (1 + returns_df['Stock_USD_Return']).cumprod()
    
    # Scenario B: Stock + Index Hedge
    returns_df['Return_Index_Hedged'] = returns_df['Stock'] + returns_df['Index_Hedge_PnL']
    returns_df['Cum_Index_Hedged'] = (1 + returns_df['Return_Index_Hedged']).cumprod()
    
    # Scenario C: Stock + Index Hedge + FX Hedge
    returns_df['Cum_Fully_Hedged'] = (1 + returns_df['Total_Hedged_Return']).cumprod()
    
    # 2. Performance Metrics Calculation
    metrics = {}
    for col in ['Stock_USD_Return', 'Return_Index_Hedged', 'Total_Hedged_Return']:
        # Annualized Return (assuming 252 trading days)
        ann_return = returns_df[col].mean() * 252
        # Annualized Volatility
        ann_vol = returns_df[col].std() * np.sqrt(252)
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        metrics[col] = {'Ann. Return': ann_return, 'Ann. Volatility': ann_vol, 'Sharpe Ratio': sharpe}
        
    print(pd.DataFrame(metrics).T)

    # 3. Calculate Drawdowns for the sub-plot
    def get_drawdown(cum_rets):
        return (cum_rets / cum_rets.cummax()) - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- Top Plot: Cumulative PnL ---
    ax1.plot(returns_df['Cum_Stock'], label='Unhedged (USD Stock Return)', color='gray', alpha=0.5)
    ax1.plot(returns_df['Cum_Index_Hedged'], label='Index Hedged (Beta Adjusted)', color='blue', linewidth=1.5)
    ax1.plot(returns_df['Cum_Fully_Hedged'], label='Fully Hedged (Index + FX)', color='green', linewidth=2)
    
    ax1.set_title('Strategy PnL Comparison: Hedged vs. Unhedged', fontsize=14)
    ax1.set_ylabel('Cumulative Return (Base 1.0)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Bottom Plot: Drawdowns ---
    ax2.plot(returns_df.index, get_drawdown(returns_df['Cum_Stock']), color='gray', alpha=0.5, label='Unhedged Drawdown')
    ax2.plot(returns_df.index, get_drawdown(returns_df['Cum_Index_Hedged']), color='blue', alpha=1, label='Index Hedged Drawdown')
    ax2.plot(returns_df.index, get_drawdown(returns_df['Cum_Fully_Hedged']), color='green', alpha=1, label='Fully Hedged Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"quant_strat_interview/q2_pnl_lookback_{lookback}.png")

    return returns_df

track_performance(hedged_df, lookback)