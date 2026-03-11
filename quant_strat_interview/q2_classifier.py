import pandas as pd
import statsmodels.api as sm
import itertools

# we first conducted multi-linear regression on (index, currency pair) pair
# to find out which pair has the most explanatory power (R^2)

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
    print(f"index: {idx}, p_index: {model.pvalues[idx]}")
    print(f"currency: {curr}, p_curr: {model.pvalues[curr]}")
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

# Obviously FTSE has the most explanatory power, but which currency pair is unclear
# GBPEUR is statistically insignificant, filtered out

# since GBPUSD and EURUSD are both statistically significant, they are highly collinear
# put both of them into another MLR, 
# the one with stronger explanatory power will remain statistically significant, 
# while the other one will not
y = returns_df['Stock 1']
idx = 'FTSE 100'
curr1 = 'GBPUSD'
curr2 = 'EURUSD'
X = returns_df[[idx, curr1, curr2]]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(f"index: {idx}, p_index: {model.pvalues[idx]}")
print(f"currency: {curr1}, p_curr: {model.pvalues[curr1]}")
print(f"currency: {curr2}, p_curr: {model.pvalues[curr2]}")
print(f"R^2: {model.rsquared}")