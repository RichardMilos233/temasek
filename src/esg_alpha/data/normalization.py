from __future__ import annotations

import numpy as np
import pandas as pd

from esg_alpha.data.base import SectorNormalizer


class NoOpSectorNormalizer(SectorNormalizer):
    def normalize(self, esg_features: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
        return esg_features.copy()


class SectorZScoreNormalizer(SectorNormalizer):
    """Within each date, z-score every ESG metric by sector."""

    def normalize(self, esg_features: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
        sector_map = sectors.to_dict()
        normalized_blocks: list[pd.DataFrame] = []

        for _, block in esg_features.groupby(level="date"):
            tickers = block.index.get_level_values("ticker")
            sector_labels = pd.Series(
                [sector_map.get(ticker, "UNKNOWN") for ticker in tickers],
                index=block.index,
            )

            normalized = pd.DataFrame(index=block.index, columns=block.columns, dtype=float)
            for column in block.columns:
                values = block[column].astype(float)
                group_mean = values.groupby(sector_labels).transform("mean")
                group_std = values.groupby(sector_labels).transform("std").replace(0, np.nan)
                normalized[column] = ((values - group_mean) / group_std).fillna(0.0)

            normalized_blocks.append(normalized)

        if not normalized_blocks:
            return esg_features.fillna(0.0)

        return pd.concat(normalized_blocks).sort_index().fillna(0.0)
