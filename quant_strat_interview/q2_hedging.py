import pandas as pd
import numpy as np
import statsmodels.api as sm

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

df = pd.read_csv("quant_strat_interview/q2.csv", header=0)
hedged_df = implement_hedging_strategy(df)
print(hedged_df.head())
hedged_df.to_csv("quant_strat_interview/q2_hedged.csv")