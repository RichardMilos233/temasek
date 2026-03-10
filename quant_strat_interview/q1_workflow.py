import argparse

from q1_vix_adjustor import generate_adjusted_vix
from q1_pricing import generate_pricing_data
from q1_analysis import run_analysis


def run_q1_workflow(hedge_ratio=1.0, strike_pct=0.90):
    # Step 1: VIX adjustment
    generate_adjusted_vix()

    # Step 2: Option pricing inputs and values
    generate_pricing_data(moneyness=strike_pct)

    # Step 3 and Step 4: Rolling hedge backtest and analysis/plotting
    run_analysis(target_ratio=hedge_ratio)


def parse_args():
    parser = argparse.ArgumentParser(description='Run Q1 workflow with configurable hedge ratio and strike percentage.')
    parser.add_argument('--hedge-ratio', type=float, default=1.0, help='Put hedge ratio used in rolling hedge analysis.')
    parser.add_argument('--strike-pct', type=float, default=0.7, help='Strike as a percentage of spot (e.g. 0.90 for 90%%).')
    return parser.parse_args()


if __name__ == '__main__':
    # args = parse_args()
    for hedge_ratio in [0.5, 0.7, 1.0, 1.5, 2.0]:
        for strike_pct in [0.7, 0.8, 0.9, 1.0]:
            run_q1_workflow(hedge_ratio=hedge_ratio, strike_pct=strike_pct)
