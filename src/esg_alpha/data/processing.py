from __future__ import annotations

import pandas as pd

from esg_alpha.data.base import ESGImputer, SectorNormalizer
from esg_alpha.data.pit import build_point_in_time_feature_panel
from esg_alpha.models import ProcessedDataBundle, RawDataBundle


class DataProcessor:
    def __init__(self, imputer: ESGImputer, normalizer: SectorNormalizer) -> None:
        self.imputer = imputer
        self.normalizer = normalizer

    def process(self, raw: RawDataBundle) -> ProcessedDataBundle:
        returns = raw.returns.sort_index()
        dates = returns.index
        tickers = list(returns.columns)

        metrics = sorted(raw.esg_records["metric"].astype(str).unique().tolist())
        point_in_time_features = build_point_in_time_feature_panel(
            raw.esg_records,
            dates=dates,
            tickers=tickers,
            metrics=metrics,
        )

        imputed = self.imputer.impute(point_in_time_features)
        normalized = self.normalizer.normalize(imputed, raw.sectors)

        style_factors: dict[str, pd.DataFrame] = {}
        for factor_name, exposures in raw.style_factors.items():
            aligned = exposures.reindex(index=dates, columns=tickers)
            style_factors[factor_name] = aligned.ffill().bfill().fillna(0.0)

        factor_returns = raw.factor_returns.reindex(index=dates).ffill().bfill().fillna(0.0)

        if raw.benchmark_returns is None:
            benchmark_returns = returns.mean(axis=1).rename("benchmark_return")
        else:
            benchmark_returns = raw.benchmark_returns.reindex(index=dates).ffill().bfill().fillna(0.0)
            benchmark_returns.name = "benchmark_return"

        benchmark_weights = None
        if raw.benchmark_weights is not None:
            benchmark_weights = raw.benchmark_weights.reindex(index=dates, columns=tickers).fillna(0.0)

        news = None
        if raw.news is not None:
            news = raw.news.copy()
            news["date"] = pd.to_datetime(news["date"])

        return ProcessedDataBundle(
            returns=returns,
            esg_features=normalized,
            sectors=raw.sectors.reindex(tickers),
            style_factors=style_factors,
            factor_returns=factor_returns,
            benchmark_returns=benchmark_returns,
            benchmark_weights=benchmark_weights,
            news=news,
        )
