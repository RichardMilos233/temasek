import pandas as pd
import numpy as np
from scipy.stats import norm

VIX_ADJUSTED_PATH = 'quant_strat_interview/q1_vix_adjusted.csv'
SP_INPUT_PATH = 'quant_strat_interview/q1_sp.xlsx'
PRICING_OUTPUT_PATH = 'quant_strat_interview/q1_pricing.csv'
RISK_FREE_RATE = 0.01
DIVIDEND_YIELD = 0.0
MONEYNESS = 0.90



def bs_put_price(S, K, T, r, q, sigma):
    """
    Calculates the Black-Scholes European put price for a single day.
    Returns intrinsic value if the option is at or past expiration.
    """
    if T <= 0:
        return max(K - S, 0)
        
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


def load_inputs(vix_path=VIX_ADJUSTED_PATH, sp_path=SP_INPUT_PATH):
    vix = pd.read_csv(vix_path)
    vix['Date'] = pd.to_datetime(vix['Date'])

    sp = pd.read_excel(sp_path, header=0, engine='openpyxl')
    sp = sp.dropna()
    sp['Date'] = pd.to_datetime(sp['Date'])
    return sp, vix


def build_pricing_dataframe(sp_df, vix_df, moneyness=MONEYNESS):
    df = pd.merge(sp_df, vix_df, on='Date', how='inner')
    df.set_index('Date', inplace=True)

    df['Strike'] = df['SP'] * moneyness

    df['Put_Price_90D'] = bs_put_price(
        S=df['SP'],
        K=df['Strike'],
        T=90 / 365.0,
        r=RISK_FREE_RATE,
        q=DIVIDEND_YIELD,
        sigma=df['Vol_3M'] / 100.0,
    )

    df['Put_Price_1Y'] = bs_put_price(
        S=df['SP'],
        K=df['Strike'],
        T=365 / 365.0,
        r=RISK_FREE_RATE,
        q=DIVIDEND_YIELD,
        sigma=df['Vol_1Y'] / 100.0,
    )
    return df


def generate_pricing_data(
    vix_path=VIX_ADJUSTED_PATH,
    sp_path=SP_INPUT_PATH,
    output_path=PRICING_OUTPUT_PATH,
    moneyness=MONEYNESS,
):
    sp, vix = load_inputs(vix_path=vix_path, sp_path=sp_path)
    df = build_pricing_dataframe(sp, vix, moneyness=moneyness)
    print("\n--- Strategy Pricing Data ---")
    print(df[['SP', 'VIX', 'Strike', 'Vol_3M', 'Vol_1Y', 'Put_Price_90D', 'Put_Price_1Y']])
    df.to_csv(output_path)
    return df


if __name__ == '__main__':
    generate_pricing_data()