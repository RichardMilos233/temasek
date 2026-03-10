from q1_vix_adjustor import generate_adjusted_vix
from q1_pricing import generate_pricing_data
from q1_analysis import run_analysis


def run_q1_workflow():
    # Step 1: VIX adjustment
    generate_adjusted_vix()

    # Step 2: Option pricing inputs and values
    generate_pricing_data()

    # Step 3 and Step 4: Rolling hedge backtest and analysis/plotting
    run_analysis()


if __name__ == '__main__':
    run_q1_workflow()
