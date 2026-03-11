import numpy as np
from q1_pricing import bs_put_price

# at the end of the roll period, we need to rebalance our portfolio, 
# if the stock price goes up, we need to put more, which is exactly why rebalance is needed
# rebalance according to hedge ratio

def calculate_rebalanced_portfolio(df, vol_col, roll_period, initial_capital, r=0.01, q=0.0, hedge_ratio=1.0):
    """
    Calculates the total portfolio value over time, rebalancing at each roll period 
    to maintain the target ratio of put options to underlying stock.
    """
    num_days = len(df)
    portfolio_value = np.zeros(num_days)
    
    # Track the number of shares and puts currently held
    N_s = 0.0
    N_p = 0.0
    
    # Initialize capital at t=0
    V_current = initial_capital
    
    i = 0
    while i < num_days:
        K_i = df['Strike'].iloc[i]
        S_i = df['SP'].iloc[i]
        sigma_i = df[vol_col].iloc[i] / 100.0
        
        expiration_idx = min(i + roll_period, num_days - 1)
        T_years_initial = (expiration_idx - i) / 252.0
        
        # 1. Price the put option today to figure out the premium
        P_i = bs_put_price(S=S_i, K=K_i, T=T_years_initial, r=r, q=q, sigma=sigma_i)
        
        # 2. Rebalance: Calculate how many shares and puts we can buy with V_current
        # Formula: N_s = V_current / (S_current + hedge_ratio * P_current)
        N_s = V_current / (S_i + hedge_ratio * P_i)
        N_p = hedge_ratio * N_s
        
        # 3. Mark to market daily until expiration
        for j in range(i, expiration_idx + 1):
            S_j = df['SP'].iloc[j]
            # ith day 3m volatility to estimate jth day expiration_idx - j day volatility
            sigma_j = df[vol_col].iloc[j] / 100.0
            T_years_j = (expiration_idx - j) / 252.0
            
            # Current value of the active option
            P_j = bs_put_price(S=S_j, K=K_i, T=T_years_j, r=r, q=q, sigma=sigma_j)
            
            # Total portfolio value on day j
            V_j = (N_s * S_j) + (N_p * P_j)
            portfolio_value[j] = V_j
            
        # 4. The current capital for the next roll is the portfolio value at expiration
        V_current = portfolio_value[expiration_idx]
            
        i = expiration_idx
        if i == num_days - 1:
            break
            
    return portfolio_value