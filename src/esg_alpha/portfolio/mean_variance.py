from __future__ import annotations

import numpy as np
import pandas as pd

from esg_alpha.portfolio.base import PortfolioConstructor
from esg_alpha.portfolio.quantile import QuantileLongShortConstructor


class ConstrainedMeanVarianceConstructor(PortfolioConstructor):
    """Approximate constrained mean-variance optimizer using projected gradient."""

    def __init__(
        self,
        risk_aversion: float = 5.0,
        gross_leverage: float = 1.5,
        max_abs_weight: float = 0.08,
        turnover_limit: float | None = 0.35,
        cov_lookback: int = 60,
        ridge: float = 1e-4,
        sector_neutral: bool = True,
        n_iterations: int = 120,
        step_size: float = 0.05,
        fallback_constructor: PortfolioConstructor | None = None,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.gross_leverage = gross_leverage
        self.max_abs_weight = max_abs_weight
        self.turnover_limit = turnover_limit
        self.cov_lookback = cov_lookback
        self.ridge = ridge
        self.sector_neutral = sector_neutral
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.fallback_constructor = fallback_constructor or QuantileLongShortConstructor(
            gross_leverage=min(gross_leverage, 1.0)
        )

    def construct(
        self,
        expected_returns: pd.DataFrame,
        realized_returns: pd.DataFrame,
        sectors: pd.Series,
        benchmark_weights: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        tickers = list(expected_returns.columns)
        dates = expected_returns.index
        sector_map = sectors.reindex(tickers).fillna("UNKNOWN")

        fallback_weights = self.fallback_constructor.construct(
            expected_returns,
            realized_returns,
            sectors,
            benchmark_weights,
        )

        weights = pd.DataFrame(0.0, index=dates, columns=tickers)
        previous_weights = pd.Series(0.0, index=tickers)

        for date in dates:
            mu = expected_returns.loc[date].reindex(tickers).fillna(0.0)
            history = realized_returns.loc[:date, tickers].tail(self.cov_lookback)

            if history.shape[0] < max(10, len(tickers) // 3):
                candidate = fallback_weights.loc[date].reindex(tickers).fillna(0.0)
                weights.loc[date] = candidate
                previous_weights = candidate
                continue

            variance = history.var(axis=0).reindex(tickers).fillna(0.0).to_numpy(dtype=float)
            variance = variance + self.ridge

            candidate_array = self._solve_projected_gradient(
                mu=mu.to_numpy(dtype=float),
                variance=variance,
                previous=previous_weights.to_numpy(dtype=float),
                sector_map=sector_map,
            )

            if candidate_array is None:
                candidate = fallback_weights.loc[date].reindex(tickers).fillna(0.0)
                weights.loc[date] = candidate
                previous_weights = candidate
                continue

            candidate = pd.Series(candidate_array, index=tickers).fillna(0.0)
            gross = candidate.abs().sum()
            if gross > self.gross_leverage and gross > 0:
                candidate *= self.gross_leverage / gross

            weights.loc[date] = candidate
            previous_weights = candidate

        return weights.fillna(0.0)

    def _solve_projected_gradient(
        self,
        mu: np.ndarray,
        variance: np.ndarray,
        previous: np.ndarray,
        sector_map: pd.Series,
    ) -> np.ndarray | None:
        if len(mu) == 0 or len(variance) != len(mu):
            return None

        weights = previous.copy()
        for _ in range(self.n_iterations):
            gradient = mu - 2.0 * self.risk_aversion * variance * weights
            weights = weights + self.step_size * gradient
            weights = self._project_constraints(weights, previous, sector_map)

        return weights

    def _project_constraints(
        self,
        weights: np.ndarray,
        previous: np.ndarray,
        sector_map: pd.Series,
    ) -> np.ndarray:
        projected = np.clip(weights, -self.max_abs_weight, self.max_abs_weight)

        # Enforce dollar neutrality.
        projected = projected - projected.mean()

        if self.sector_neutral:
            for sector in sector_map.unique():
                mask = sector_map == sector
                if int(mask.sum()) <= 1:
                    continue
                sector_mean = projected[mask].mean()
                projected[mask] = projected[mask] - sector_mean

        gross = np.abs(projected).sum()
        if gross > self.gross_leverage and gross > 0:
            projected = projected * (self.gross_leverage / gross)

        if self.turnover_limit is not None:
            delta = projected - previous
            turnover = np.abs(delta).sum()
            if turnover > self.turnover_limit and turnover > 0:
                projected = previous + delta * (self.turnover_limit / turnover)

        projected = np.clip(projected, -self.max_abs_weight, self.max_abs_weight)

        gross = np.abs(projected).sum()
        if gross > self.gross_leverage and gross > 0:
            projected = projected * (self.gross_leverage / gross)

        projected = projected - projected.mean()
        return projected
