from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from esg_alpha.data.base import DataIngestor
from esg_alpha.models import RawDataBundle

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "UNH",
    "JNJ",
    "PFE",
    "MRK",
    "WMT",
    "COST",
    "TGT",
    "NKE",
    "CAT",
    "DE",
    "BA",
    "LMT",
    "IBM",
    "ORCL",
    "SBUX",
    "MCD",
]

DEFAULT_METRICS = [
    "carbon_intensity",
    "water_use",
    "board_diversity",
    "data_privacy",
    "governance_quality",
]


@dataclass
class MockPointInTimeIngestor(DataIngestor):
    start_date: str = "2019-01-01"
    end_date: str = "2025-12-31"
    seed: int = 7
    tickers: list[str] | None = None
    metrics: list[str] | None = None
    restatement_probability: float = 0.12

    def ingest(self) -> RawDataBundle:
        rng = np.random.default_rng(self.seed)
        tickers = self.tickers or DEFAULT_TICKERS
        metrics = self.metrics or DEFAULT_METRICS
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        sectors = self._generate_sector_map(tickers, rng)
        style_factors = self._generate_style_factor_exposures(dates, tickers, rng)
        factor_returns = self._generate_factor_returns(dates, list(style_factors.keys()), rng)
        returns = self._generate_asset_returns(style_factors, factor_returns, rng)

        esg_records = self._generate_esg_records(dates, tickers, metrics, sectors, rng)
        news = self._generate_mock_news(esg_records, rng)

        benchmark_returns = returns.mean(axis=1).rename("benchmark_return")
        benchmark_weights = pd.DataFrame(1.0 / len(tickers), index=dates, columns=tickers)

        return RawDataBundle(
            returns=returns,
            esg_records=esg_records,
            sectors=sectors,
            style_factors=style_factors,
            factor_returns=factor_returns,
            benchmark_returns=benchmark_returns,
            benchmark_weights=benchmark_weights,
            news=news,
        )

    @staticmethod
    def _generate_sector_map(tickers: list[str], rng: np.random.Generator) -> pd.Series:
        sector_buckets = [
            "Technology",
            "Financials",
            "Energy",
            "Healthcare",
            "Industrials",
            "Consumer",
        ]
        assignments = [sector_buckets[idx % len(sector_buckets)] for idx in range(len(tickers))]
        rng.shuffle(assignments)
        return pd.Series(assignments, index=tickers, name="sector")

    @staticmethod
    def _generate_style_factor_exposures(
        dates: pd.DatetimeIndex,
        tickers: list[str],
        rng: np.random.Generator,
    ) -> dict[str, pd.DataFrame]:
        factor_names = ["size", "value", "momentum", "quality", "low_vol"]
        exposures: dict[str, pd.DataFrame] = {}

        for factor_name in factor_names:
            base = rng.normal(0.0, 1.0, size=len(tickers))
            innovations = rng.normal(0.0, 0.05, size=(len(dates), len(tickers)))
            path = base + np.cumsum(innovations, axis=0)
            exposures[factor_name] = pd.DataFrame(path, index=dates, columns=tickers)

        return exposures

    @staticmethod
    def _generate_factor_returns(
        dates: pd.DatetimeIndex,
        factor_names: list[str],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        factor_vol = {
            "size": 0.003,
            "value": 0.003,
            "momentum": 0.004,
            "quality": 0.0025,
            "low_vol": 0.002,
        }

        data = {}
        for factor_name in factor_names:
            vol = factor_vol.get(factor_name, 0.003)
            data[factor_name] = rng.normal(0.00005, vol, size=len(dates))

        return pd.DataFrame(data, index=dates)

    @staticmethod
    def _generate_asset_returns(
        style_factors: dict[str, pd.DataFrame],
        factor_returns: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        dates = factor_returns.index
        tickers = style_factors[next(iter(style_factors))].columns
        returns = np.zeros((len(dates), len(tickers)))

        for factor_name, exposure_df in style_factors.items():
            factor_component = exposure_df.values * factor_returns[factor_name].values.reshape(-1, 1)
            returns += 0.35 * factor_component

        returns += rng.normal(0.0, 0.008, size=returns.shape)
        returns = np.clip(returns, -0.2, 0.2)

        return pd.DataFrame(returns, index=dates, columns=tickers)

    def _generate_esg_records(
        self,
        dates: pd.DatetimeIndex,
        tickers: list[str],
        metrics: list[str],
        sectors: pd.Series,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        observation_dates = dates[::21]
        state: dict[tuple[str, str], float] = {
            (ticker, metric): float(rng.uniform(35.0, 65.0))
            for ticker in tickers
            for metric in metrics
        }

        rows: list[dict[str, object]] = []

        for period_end in observation_dates:
            for ticker in tickers:
                sector = sectors.loc[ticker]
                for metric in metrics:
                    drift = self._sector_metric_drift(sector, metric)
                    shock = rng.normal(drift, 2.5)
                    state[(ticker, metric)] = float(np.clip(state[(ticker, metric)] + shock, 0.0, 100.0))

                    lag_days = int(rng.integers(5, 35))
                    asof_date = period_end + BDay(lag_days)

                    rows.append(
                        {
                            "source_period_end": period_end,
                            "asof_date": asof_date,
                            "ticker": ticker,
                            "metric": metric,
                            "value": state[(ticker, metric)],
                            "is_restatement": False,
                        }
                    )

                    if rng.random() < self.restatement_probability:
                        restated_asof = asof_date + BDay(int(rng.integers(20, 80)))
                        restated_value = float(
                            np.clip(state[(ticker, metric)] + rng.normal(0.0, 1.75), 0.0, 100.0)
                        )
                        rows.append(
                            {
                                "source_period_end": period_end,
                                "asof_date": restated_asof,
                                "ticker": ticker,
                                "metric": metric,
                                "value": restated_value,
                                "is_restatement": True,
                            }
                        )

        records = pd.DataFrame(rows)
        return records.sort_values("asof_date").reset_index(drop=True)

    @staticmethod
    def _sector_metric_drift(sector: str, metric: str) -> float:
        if metric == "carbon_intensity":
            return -0.18 if sector == "Energy" else -0.05
        if metric == "water_use":
            return -0.12 if sector in {"Energy", "Industrials"} else -0.03
        if metric == "board_diversity":
            return 0.08
        if metric == "data_privacy":
            return 0.12 if sector in {"Technology", "Financials"} else 0.02
        if metric == "governance_quality":
            return 0.06
        return 0.0

    @staticmethod
    def _generate_mock_news(esg_records: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if esg_records.empty:
            return pd.DataFrame(columns=["date", "ticker", "text"])

        templates = {
            "positive": [
                "{ticker} announced an improvement in {metric} practices.",
                "{ticker} received favorable ESG attention tied to {metric}.",
            ],
            "neutral": [
                "{ticker} updated stakeholders on its {metric} disclosures.",
                "{ticker} published routine reporting related to {metric}.",
            ],
            "negative": [
                "{ticker} faces scrutiny over {metric} controls.",
                "{ticker} reported a setback linked to {metric} performance.",
            ],
        }

        sampled = esg_records.groupby(["asof_date", "ticker"], as_index=False).first()
        rows: list[dict[str, object]] = []

        for row in sampled.itertuples(index=False):
            sentiment = str(rng.choice(["positive", "neutral", "negative"], p=[0.35, 0.4, 0.25]))
            template = str(rng.choice(templates[sentiment]))
            metric = str(row.metric).replace("_", " ")
            rows.append(
                {
                    "date": pd.Timestamp(row.asof_date),
                    "ticker": str(row.ticker),
                    "text": template.format(ticker=row.ticker, metric=metric),
                }
            )

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
