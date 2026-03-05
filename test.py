import numpy as np
import pandas as pd
import cvxpy as cp

def run_environment_check():
    print("Initializing Quantamental Optimizer Check...")
    
    # 1. Generate Synthetic Data (10 Stocks)
    np.random.seed(42)
    n_assets = 10
    tickers = [f"STOCK_{i}" for i in range(1, n_assets + 1)]
    
    # Synthetic ESG Alpha Signal (Expected Returns)
    # Ranging roughly between -5% to +15%
    mu = np.random.randn(n_assets) * 0.05 + 0.05 
    
    # Synthetic Covariance Matrix (Risk Model)
    # Using a random matrix to generate a positive semi-definite covariance matrix
    temp_matrix = np.random.randn(n_assets, n_assets)
    Sigma = temp_matrix.T @ temp_matrix / 100 

    # 2. Setup the Convex Optimization Problem using CVXPY
    w = cp.Variable(n_assets) # The weights we want to solve for
    gamma = cp.Parameter(nonneg=True) # Risk aversion penalty
    gamma.value = 2.0 

    # Define the objective function: Maximize (Return - Risk Penalty)
    expected_return = mu.T @ w
    portfolio_variance = cp.quad_form(w, Sigma)
    objective = cp.Maximize(expected_return - (gamma / 2) * portfolio_variance)

    # Define the constraints: Fully invested (sum to 1) and Long-only (weights >= 0)
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    # 3. Solve the Problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        print("\nOptimization Successful!")
        print(f"Solver Status: {prob.status}")
        print(f"Optimal Objective Value: {prob.value:.4f}\n")
        
        # Format the output weights nicely using Pandas
        weights_df = pd.DataFrame({
            'Ticker': tickers,
            'Alpha Signal': np.round(mu, 4),
            'Optimal Weight': np.round(w.value, 4)
        }).sort_values(by='Optimal Weight', ascending=False)
        
        print("Final Portfolio Construction:")
        print(weights_df.to_string(index=False))
        
    except Exception as e:
        print(f"\nOptimization Failed. Error: {e}")

if __name__ == "__main__":
    run_environment_check()