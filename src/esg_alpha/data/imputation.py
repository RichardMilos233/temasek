from __future__ import annotations

import pandas as pd

from esg_alpha.data.base import ESGImputer


class ForwardFillMedianImputer(ESGImputer):
    """Forward-fill within ticker, then fill leftover gaps with same-date median."""

    def impute(self, esg_features: pd.DataFrame) -> pd.DataFrame:
        ffilled = esg_features.groupby(level="ticker", group_keys=False).ffill()
        median_filled = ffilled.groupby(level="date", group_keys=False).apply(
            lambda frame: frame.fillna(frame.median())
        )
        return median_filled.fillna(0.0)


class CrossSectionalKNNImputer(ESGImputer):
    """Apply KNN imputation per date across tickers."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def impute(self, esg_features: pd.DataFrame) -> pd.DataFrame:
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        blocks: list[pd.DataFrame] = []

        for _, block in esg_features.groupby(level="date"):
            if block.isna().all().all():
                blocks.append(block.fillna(0.0))
                continue

            transformed = imputer.fit_transform(block)
            filled_block = pd.DataFrame(transformed, index=block.index, columns=block.columns)
            blocks.append(filled_block)

        if not blocks:
            return esg_features.fillna(0.0)

        return pd.concat(blocks).sort_index().fillna(0.0)
