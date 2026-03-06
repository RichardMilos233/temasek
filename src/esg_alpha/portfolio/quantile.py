from __future__ import annotations

import pandas as pd

from esg_alpha.portfolio.base import PortfolioConstructor


class QuantileLongShortConstructor(PortfolioConstructor):
    def __init__(
        self,
        long_quantile: float = 0.8,
        short_quantile: float = 0.2,
        gross_leverage: float = 1.0,
    ) -> None:
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.gross_leverage = gross_leverage

    def construct(
        self,
        expected_returns: pd.DataFrame,
        realized_returns: pd.DataFrame,
        sectors: pd.Series,
        benchmark_weights: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=expected_returns.index, columns=expected_returns.columns)

        for date in expected_returns.index:
            mu = expected_returns.loc[date].dropna()
            if mu.empty:
                continue

            long_cutoff = mu.quantile(self.long_quantile)
            short_cutoff = mu.quantile(self.short_quantile)

            long_names = mu[mu >= long_cutoff].index
            short_names = mu[mu <= short_cutoff].index

            n_long = len(long_names)
            n_short = len(short_names)

            if n_long > 0:
                weights.loc[date, long_names] = (self.gross_leverage / 2.0) / n_long
            if n_short > 0:
                weights.loc[date, short_names] = -(self.gross_leverage / 2.0) / n_short

        return weights.fillna(0.0)
