import pandas as pd
import numpy as np

# The VIX, officially known as the Chicago Board Options Exchange (CBOE) Volatility Index, 
# is a real-time market index that represents the stock market's expectation of 
# volatility over the next 30 days. 
# Widely recognized as the premier gauge of U.S. equity market volatility, 
# it is commonly referred to as Wall Street's "fear gauge" or "fear index."

# in short, market drops -> high vix -> fear, vice versa
#  vix is negatively correlated to S&P500

VIX_INPUT_PATH = 'quant_strat_interview/q1_vix.xlsx'
VIX_OUTPUT_PATH = 'quant_strat_interview/q1_vix_adjusted.csv'
T_3M = 3 / 12
T_1Y = 1
KAPPA = 2.5

def calc_term_structure_vol(V0_var, theta, kappa, T):
    """
    Calculates the implied volatility for a given maturity T using a 
    mean-reverting variance model.
    """
    # Calculate the weight factor based on time and speed of reversion
    weight = (1 - np.exp(-kappa * T)) / (kappa * T)
    
    # Calculate implied integrated variance
    implied_var = theta + (V0_var - theta) * weight
    
    # Convert variance back to an annualized percentage volatility
    implied_vol = np.sqrt(implied_var) * 100
    return implied_vol


def load_vix_data(path=VIX_INPUT_PATH):
    vix = pd.read_excel(path, header=0, engine='openpyxl')
    vix = vix.dropna()
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix.set_index('Date', inplace=True)
    return vix


def build_adjusted_vix_dataframe(vix_df):
    vix = vix_df.copy()
    vix['V0_var'] = (vix['VIX'] / 100) ** 2

    long_term_mean = vix['VIX'].mean(axis=0)
    theta = (long_term_mean / 100) ** 2

    vix['Vol_3M'] = calc_term_structure_vol(vix['V0_var'], theta, KAPPA, T_3M)
    vix['Vol_1Y'] = calc_term_structure_vol(vix['V0_var'], theta, KAPPA, T_1Y)
    return vix


def generate_adjusted_vix(input_path=VIX_INPUT_PATH, output_path=VIX_OUTPUT_PATH):
    vix = load_vix_data(input_path)
    vix = build_adjusted_vix_dataframe(vix)
    print(vix.head())
    vix.to_csv(output_path)
    return vix


if __name__ == '__main__':
    generate_adjusted_vix()