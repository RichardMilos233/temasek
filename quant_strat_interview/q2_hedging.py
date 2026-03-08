import pandas as pd
import numpy as np
import statsmodels.api as sm

def implement_hedging_strategy(df, lookback=15):
    """
    df: DataFrame containing columns 'Stock 1', 'FTSE 100', 'GBPUSD'
    lookback: Window for calculating rolling beta 
    """
    # 1. Ensure Date is index and calculate daily returns
    # Note: Using pct_change() to get daily growth rates
    returns = df.set_index('Date').pct_change().dropna()
    
    # Rename for easier internal handling
    returns = returns.rename(columns={
        'Stock 1': 'Stock',
        'FTSE 100': 'Index',
        'GBPUSD': 'FX'
    })
    
    hedge_ratios = []
    
    # 2. Rolling Beta Calculation (Index Hedge Sizing)
    for i in range(len(returns)):
        if i < lookback:
            hedge_ratios.append(np.nan)
            continue
            
        # Get historical window
        y = returns['Stock'].iloc[i-lookback:i]
        X = returns['Index'].iloc[i-lookback:i]
        X = sm.add_constant(X) # Include intercept for OLS
        
        model = sm.OLS(y, X).fit()
        # The coefficient (Beta) tells us how many units of FTSE to short 
        # for every 1 unit of Stock held.
        beta = model.params['Index']
        hedge_ratios.append(beta)
        
    returns['Beta_Index'] = hedge_ratios
    
    # 3. Calculate Performance
    # Performance of the long stock position
    returns['Stock_PnL'] = returns['Stock']
    
    # Performance of the Short Index Hedge
    # (Inverse of Index return * Beta)
    returns['Index_Hedge_PnL'] = -returns['Beta_Index'] * returns['Index']
    
    # Performance of the Short FX Hedge
    # Hedging GBPUSD (Assuming your base currency is USD)
    # This offsets the impact of GBP weakening against the USD
    returns['FX_Hedge_PnL'] = -1.0 * returns['FX'] 
    
    # 4. Total Combined Daily Return
    returns['Total_Hedged_Return'] = (
        returns['Stock_PnL'] + 
        returns['Index_Hedge_PnL'] + 
        returns['FX_Hedge_PnL']
    )
    
    return returns.dropna()

df = pd.read_csv("quant_strat_interview/q2.csv", header=0)
hedged_df = implement_hedging_strategy(df)
print(hedged_df.head())
hedged_df.to_csv("quant_strat_interview/q2_hedged.csv")