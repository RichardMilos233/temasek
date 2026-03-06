from __future__ import annotations

import numpy as np
import pandas as pd

from esg_alpha.models import ProcessedDataBundle
from esg_alpha.signal.base import SignalOrthogonalizer
from esg_alpha.utils import cross_sectional_zscore


class NoOpOrthogonalizer(SignalOrthogonalizer):
    def orthogonalize(self, signal: pd.DataFrame, data: ProcessedDataBundle) -> pd.DataFrame:
        return signal.fillna(0.0)


class CrossSectionalOLSOrthogonalizer(SignalOrthogonalizer):
    """Regress signal on style factors and sector dummies, keep residual."""

    def __init__(self, include_sector_dummies: bool = True, zscore_output: bool = True) -> None:
        self.include_sector_dummies = include_sector_dummies
        self.zscore_output = zscore_output

    def orthogonalize(self, signal: pd.DataFrame, data: ProcessedDataBundle) -> pd.DataFrame:
        dates = signal.index
        tickers = signal.columns
        output = pd.DataFrame(0.0, index=dates, columns=tickers)
        sector_map = data.sectors.reindex(tickers).fillna("UNKNOWN")

        for date in dates:
            residual = signal.loc[date].astype(float).copy()

            # Sequentially project out each style factor to avoid unstable linear algebra backends.
            for _, exposures in data.style_factors.items():
                x = exposures.reindex(index=dates, columns=tickers).loc[date].astype(float)

                valid = residual.notna() & np.isfinite(residual)
                valid &= x.notna() & np.isfinite(x)
                if valid.sum() < 3:
                    continue

                x_centered = x.loc[valid] - x.loc[valid].mean()
                residual_centered = residual.loc[valid] - residual.loc[valid].mean()

                denom = float((x_centered**2).sum())
                if denom <= 1e-12:
                    continue

                beta = float((residual_centered * x_centered).sum() / denom)
                residual.loc[valid] = residual.loc[valid] - beta * x_centered

            if self.include_sector_dummies:
                for sector in sector_map.unique():
                    members = sector_map[sector_map == sector].index
                    if len(members) <= 1:
                        continue
                    sector_mean = residual.loc[members].mean()
                    residual.loc[members] = residual.loc[members] - sector_mean

            output.loc[date] = residual.fillna(0.0)

        output = output.fillna(0.0)
        if self.zscore_output:
            output = cross_sectional_zscore(output)
        return output
