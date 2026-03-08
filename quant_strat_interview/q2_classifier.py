import pandas as pd
import statsmodels.api as sm
import itertools

# 1. Load your data
df = pd.read_csv("quant_strat_interview/q2.csv", header=0)


# 2. Convert Prices to Returns
# We use pct_change() to get daily returns. 
# We drop the first row because it will be NaN.
returns_df = df.set_index('Date').pct_change().dropna()

# 3. Define your candidates based on your column names
indices = ['SPX Index', 'Eurostoxx 50', 'FTSE 100']
currencies = ['GBPUSD', 'GBPEUR', 'EURUSD']

best_r2 = -1
results_summary = []

# 4. Loop through combinations
for idx, curr in itertools.product(indices, currencies):
    # Dependent variable: Stock 1
    y = returns_df['Stock 1']
    
    # Independent variables: Index + Currency + Constant
    X = returns_df[[idx, curr]]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    print(f"index: {idx}, currency: {curr}")
    print(f"R^2: {model.rsquared}")
    
    results_summary.append({
        'Index': idx,
        'Currency': curr,
        'R2': model.rsquared,
        'Beta_Index': model.params[idx],
        'Beta_Curr': model.params[curr],
        'P_Index': model.pvalues[idx],
        'P_Curr': model.pvalues[curr]
    })

# 5. Find the winner
results_df = pd.DataFrame(results_summary)
best = results_df.sort_values(by='R2', ascending=False).iloc[0]

print("--- Best Hedge Combination ---")
print(f"Index: {best['Index']}")
print(f"Currency: {best['Currency']}")
print(f"R-Squared: {best['R2']:.4f}")
print(f"Hedge Ratio (Index): {best['Beta_Index']:.4f}")
print(f"Hedge Ratio (Currency): {best['Beta_Curr']:.4f}")