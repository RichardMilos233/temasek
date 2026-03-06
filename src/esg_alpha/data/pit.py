from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"asof_date", "ticker", "metric", "value"}


def build_point_in_time_feature_panel(
    records: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    tickers: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Build a point-in-time panel by taking the last known value as of each trade date."""
    missing_columns = REQUIRED_COLUMNS.difference(records.columns)
    if missing_columns:
        raise ValueError(f"ESG records missing required columns: {sorted(missing_columns)}")

    clean_records = records.copy()
    clean_records["asof_date"] = pd.to_datetime(clean_records["asof_date"])
    clean_records = clean_records.sort_values("asof_date")

    dates_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    full_index = pd.MultiIndex.from_product([dates_index, tickers], names=["date", "ticker"])

    snapshots: list[pd.DataFrame] = []
    for asof in dates_index:
        available = clean_records.loc[clean_records["asof_date"] <= asof]
        if available.empty:
            snapshot = pd.DataFrame(index=pd.Index(tickers, name="ticker"), columns=metrics, dtype=float)
        else:
            latest = available.groupby(["ticker", "metric"], as_index=False).tail(1)
            snapshot = latest.pivot(index="ticker", columns="metric", values="value")
            snapshot = snapshot.reindex(index=tickers, columns=metrics)

        snapshot.index = pd.MultiIndex.from_product([[asof], snapshot.index], names=["date", "ticker"])
        snapshots.append(snapshot)

    panel = pd.concat(snapshots).sort_index()
    panel = panel.reindex(full_index)
    panel.columns = panel.columns.astype(str)
    return panel
