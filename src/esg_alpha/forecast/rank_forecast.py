from __future__ import annotations

import numpy as np
import pandas as pd

from esg_alpha.forecast.base import AlphaForecaster


class RankToReturnForecaster(AlphaForecaster):
    """Map cross-sectional ranks to expected returns in [-scale, scale]."""

    def __init__(self, scale: float = 0.02) -> None:
        self.scale = scale

    def forecast(self, signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        ranks = signal.rank(axis=1, pct=True)
        centered = (ranks - 0.5) * 2.0
        return centered.fillna(0.0) * self.scale


class ICScaledRankForecaster(AlphaForecaster):
    """Scale rank forecast by rolling information coefficient strength."""

    def __init__(self, scale: float = 0.02, ic_window: int = 63, min_periods: int = 21) -> None:
        self.scale = scale
        self.ic_window = ic_window
        self.min_periods = min_periods

    def forecast(self, signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        base_forecast = RankToReturnForecaster(scale=self.scale).forecast(signal, returns)
        future_returns = returns.shift(-1).reindex(index=signal.index, columns=signal.columns)

        ic_values: list[float] = []
        for date in signal.index:
            sig = signal.loc[date]
            fwd = future_returns.loc[date]
            valid = sig.notna() & fwd.notna()

            if valid.sum() < 3:
                ic_values.append(np.nan)
                continue

            rank_sig = sig[valid].rank(pct=True)
            rank_ret = fwd[valid].rank(pct=True)
            ic_values.append(self._pearson_corr(rank_sig, rank_ret))

        ic_series = pd.Series(ic_values, index=signal.index)
        rolling_ic = ic_series.rolling(self.ic_window, min_periods=self.min_periods).mean().fillna(0.0)

        # Keep scaling stable by clipping to a plausible IC range.
        multiplier = 1.0 + rolling_ic.clip(lower=-0.05, upper=0.05) / 0.05
        scaled = base_forecast.mul(multiplier, axis=0)
        return scaled.fillna(0.0)

    @staticmethod
    def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
        a_values = a.to_numpy(dtype=float)
        b_values = b.to_numpy(dtype=float)

        a_centered = a_values - a_values.mean()
        b_centered = b_values - b_values.mean()

        denom = float(np.sqrt((a_centered**2).sum() * (b_centered**2).sum()))
        if denom <= 1e-12:
            return np.nan
        return float((a_centered * b_centered).sum() / denom)
