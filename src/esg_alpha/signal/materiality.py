from __future__ import annotations

import pandas as pd

from esg_alpha.models import ProcessedDataBundle
from esg_alpha.signal.base import SignalBuilder
from esg_alpha.utils import cross_sectional_zscore

DEFAULT_MATERIALITY_MAP = {
    "Technology": {
        "data_privacy": 0.4,
        "governance_quality": 0.25,
        "board_diversity": 0.15,
        "carbon_intensity": 0.1,
        "water_use": 0.1,
    },
    "Financials": {
        "data_privacy": 0.35,
        "governance_quality": 0.35,
        "board_diversity": 0.2,
        "carbon_intensity": 0.05,
        "water_use": 0.05,
    },
    "Energy": {
        "carbon_intensity": 0.4,
        "water_use": 0.25,
        "governance_quality": 0.2,
        "board_diversity": 0.1,
        "data_privacy": 0.05,
    },
    "Healthcare": {
        "governance_quality": 0.35,
        "board_diversity": 0.25,
        "data_privacy": 0.2,
        "water_use": 0.1,
        "carbon_intensity": 0.1,
    },
    "Industrials": {
        "carbon_intensity": 0.3,
        "water_use": 0.25,
        "governance_quality": 0.2,
        "board_diversity": 0.15,
        "data_privacy": 0.1,
    },
    "Consumer": {
        "governance_quality": 0.3,
        "board_diversity": 0.2,
        "water_use": 0.2,
        "carbon_intensity": 0.15,
        "data_privacy": 0.15,
    },
}


class MaterialityMomentumSignalBuilder(SignalBuilder):
    """Use SASB-style sector materiality weights over metric momentum."""

    def __init__(
        self,
        lookback: int = 21,
        materiality_map: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.lookback = lookback
        self.materiality_map = materiality_map or DEFAULT_MATERIALITY_MAP

    def build(self, data: ProcessedDataBundle) -> pd.DataFrame:
        tickers = list(data.returns.columns)
        dates = data.returns.index

        if data.esg_features.empty:
            return pd.DataFrame(0.0, index=dates, columns=tickers)

        metrics = list(data.esg_features.columns)
        metric_momentum: dict[str, pd.DataFrame] = {}

        for metric in metrics:
            metric_history = data.esg_features[metric].unstack("ticker")
            metric_history = metric_history.reindex(index=dates, columns=tickers)
            metric_momentum[metric] = metric_history.diff(self.lookback)

        signal = pd.DataFrame(0.0, index=dates, columns=tickers)

        for ticker in tickers:
            sector = str(data.sectors.get(ticker, "UNKNOWN"))
            weights = self.materiality_map.get(sector, {})
            if not weights:
                equal_weight = 1.0 / max(len(metrics), 1)
                weights = {metric: equal_weight for metric in metrics}

            norm = sum(abs(weight) for weight in weights.values()) or 1.0
            for metric, weight in weights.items():
                if metric not in metric_momentum:
                    continue
                signal[ticker] += (weight / norm) * metric_momentum[metric][ticker].fillna(0.0)

        return cross_sectional_zscore(signal.fillna(0.0))
