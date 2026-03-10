import matplotlib.pyplot as plt
import pandas as pd
from q1_rolling_hedge import calculate_rebalanced_portfolio

PRICING_INPUT_PATH = 'quant_strat_interview/q1_pricing.csv'
TARGET_RATIO = 2


def load_pricing_data(path=PRICING_INPUT_PATH):
    return pd.read_csv(path, index_col='Date', parse_dates=True)


def add_hedge_backtest_columns(df, target_ratio=TARGET_RATIO):
    result = df.copy()

    # Initial capital is matched to the first SP level for easy comparison to buy-and-hold.
    initial_capital = result['SP'].iloc[0]

    # Buy-and-hold benchmark.
    shares_unhedged = initial_capital / result['SP'].iloc[0]
    result['Unhedged_Value'] = shares_unhedged * result['SP']

    # Rebalanced hedged portfolios.
    result['Hedged_3M_Value'] = calculate_rebalanced_portfolio(
        result, 'Vol_3M', 63, initial_capital, hedge_ratio=target_ratio
    )
    result['Hedged_1Y_Value'] = calculate_rebalanced_portfolio(
        result, 'Vol_1Y', 252, initial_capital, hedge_ratio=target_ratio
    )
    return result


def plot_portfolio_values(df, target_ratio=TARGET_RATIO):
    strike_pct = df['Strike'].iloc[0] / df['SP'].iloc[0]
    strike_pct_label = f'{strike_pct * 100:.1f}%'

    df_plot = df[:-1]  # Drop last row if incomplete
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_plot.index, df_plot['Unhedged_Value'], color='blue', alpha=0.4, label='Unhedged (100% SPX)', linewidth=2)
    ax.plot(df_plot.index, df_plot['Hedged_3M_Value'], color='green', linewidth=1.5, label=f'Hedged 3M (Ratio: {target_ratio})')
    ax.plot(df_plot.index, df_plot['Hedged_1Y_Value'], color='purple', linewidth=1.5, label=f'Hedged 1Y (Ratio: {target_ratio})')

    ax.set_title(
        f'Rebalanced Portfolio Value: Unhedged vs. Put Hedge (Strike: {strike_pct_label})',
        fontsize=14,
    )
    ax.set_ylabel('Total Portfolio Value ($)')
    ax.set_xlabel('Date')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    plt.tight_layout()
    strike_pct_for_file = f'{strike_pct:.2f}'.replace('.', 'p')
    plt.savefig(
        f'quant_strat_interview/q1_portfolio_value_ratio_{target_ratio}_strike_{strike_pct_for_file}.png'
    )

def get_max_drawdown(value_series):
    """Calculates the Maximum Drawdown as a percentage."""
    rolling_max = value_series.cummax()
    drawdown = (value_series - rolling_max) / rolling_max
    return drawdown.min()

def print_drawdown_analysis(df):
    print("\n" + "=" * 40)
    print("QUANTITATIVE RISK ANALYSIS: MAX DRAWDOWN")
    print("=" * 40)

    crisis_periods = {
        '2008 Global Financial Crisis (Oct 2007 - Mar 2009)': ('2007-10-01', '2009-03-31'),
        '2020 COVID Crash (Feb 2020 - Apr 2020)': ('2020-02-01', '2020-04-30'),
    }

    for name, (start_date, end_date) in crisis_periods.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_period = df.loc[mask]

        if not df_period.empty:
            print(f"\n{name}:")
            print(f"  Unhedged Max Drawdown: {get_max_drawdown(df_period['Unhedged_Value']):.2%}")
            print(f"  Hedged 3M Max Drawdown : {get_max_drawdown(df_period['Hedged_3M_Value']):.2%}")
            print(f"  Hedged 1Y Max Drawdown : {get_max_drawdown(df_period['Hedged_1Y_Value']):.2%}")

    print("\nOverall Timeline (Full Backtest):")
    print(f"  Unhedged Max Drawdown: {get_max_drawdown(df['Unhedged_Value']):.2%}")
    print(f"  Hedged 3M Max Drawdown : {get_max_drawdown(df['Hedged_3M_Value']):.2%}")
    print(f"  Hedged 1Y Max Drawdown : {get_max_drawdown(df['Hedged_1Y_Value']):.2%}")


def run_analysis(pricing_path=PRICING_INPUT_PATH, target_ratio=TARGET_RATIO):
    df = load_pricing_data(pricing_path)
    df = add_hedge_backtest_columns(df, target_ratio=target_ratio)
    plot_portfolio_values(df, target_ratio=target_ratio)
    print_drawdown_analysis(df)
    return df


if __name__ == '__main__':
    run_analysis()