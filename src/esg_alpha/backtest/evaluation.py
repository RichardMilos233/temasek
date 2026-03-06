from __future__ import annotations

import numpy as np
import pandas as pd

from esg_alpha.models import BacktestResult, EvaluationResult
from esg_alpha.utils import safe_divide


class PerformanceEvaluator:
    def __init__(self, annualization: int = 252) -> None:
        self.annualization = annualization

    def evaluate(
        self,
        backtest: BacktestResult,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        metrics = self._compute_metrics(backtest)
        ic_series = self._compute_ic_series(signal, returns)

        mean_ic = float(ic_series.mean()) if not ic_series.empty else 0.0
        std_ic = float(ic_series.std()) if not ic_series.empty else 0.0
        metrics["mean_ic"] = mean_ic
        metrics["ic_information_ratio"] = safe_divide(mean_ic, std_ic) * np.sqrt(self.annualization)

        factor_betas, attribution = self._factor_attribution(backtest.excess_returns, factor_returns)

        return EvaluationResult(
            metrics=metrics,
            ic_series=ic_series,
            factor_betas=factor_betas,
            attribution_contribution=attribution,
        )

    def _compute_metrics(self, backtest: BacktestResult) -> dict[str, float]:
        net = backtest.net_returns.fillna(0.0)
        excess = backtest.excess_returns.fillna(0.0)

        ann_return = float(net.mean() * self.annualization)
        ann_vol = float(net.std() * np.sqrt(self.annualization))
        sharpe = safe_divide(ann_return, ann_vol)

        ann_excess = float(excess.mean() * self.annualization)
        ann_tracking_error = float(excess.std() * np.sqrt(self.annualization))
        information_ratio = safe_divide(ann_excess, ann_tracking_error)

        cumulative = (1.0 + net).cumprod()
        max_drawdown = float(((cumulative / cumulative.cummax()) - 1.0).min()) if not cumulative.empty else 0.0

        return {
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe": sharpe,
            "annualized_excess_return": ann_excess,
            "tracking_error": ann_tracking_error,
            "information_ratio": information_ratio,
            "max_drawdown": max_drawdown,
            "average_turnover": float(backtest.turnover.mean()),
            "cost_drag_annualized": float(backtest.transaction_costs.mean() * self.annualization),
            "hit_rate": float((net > 0).mean()) if not net.empty else 0.0,
            "cumulative_return": float(cumulative.iloc[-1] - 1.0) if not cumulative.empty else 0.0,
        }

    @staticmethod
    def _compute_ic_series(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
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
            ic_values.append(PerformanceEvaluator._pearson_corr(rank_sig, rank_ret))

        return pd.Series(ic_values, index=signal.index, name="ic")

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

    def _factor_attribution(
        self,
        excess_returns: pd.Series,
        factor_returns: pd.DataFrame | None,
    ) -> tuple[pd.Series, pd.Series]:
        if factor_returns is None or factor_returns.empty:
            return pd.Series(dtype=float, name="beta"), pd.Series(dtype=float, name="attribution")

        aligned_factors = factor_returns.reindex(excess_returns.index).dropna(how="all")
        aligned_excess = excess_returns.reindex(aligned_factors.index).fillna(0.0)
        aligned_factors = aligned_factors.fillna(0.0)

        if aligned_factors.empty:
            return pd.Series(dtype=float, name="beta"), pd.Series(dtype=float, name="attribution")

        factor_beta_values: dict[str, float] = {}
        residual = aligned_excess.copy()

        # Sequential projection is robust and avoids dependencies on unstable LAPACK backends.
        for factor_name in aligned_factors.columns:
            x = aligned_factors[factor_name].astype(float)
            valid = residual.notna() & x.notna()
            if valid.sum() < 3:
                factor_beta_values[factor_name] = 0.0
                continue

            x_centered = x.loc[valid] - x.loc[valid].mean()
            residual_centered = residual.loc[valid] - residual.loc[valid].mean()

            denom = float((x_centered**2).sum())
            if denom <= 1e-12:
                beta = 0.0
            else:
                beta = float((residual_centered * x_centered).sum() / denom)

            factor_beta_values[factor_name] = beta
            residual.loc[valid] = residual.loc[valid] - beta * x_centered

        factor_beta = pd.Series(factor_beta_values, name="beta")
        annualized_factor_premia = aligned_factors.mean() * self.annualization
        factor_contribution = factor_beta * annualized_factor_premia
        alpha_annualized = float(aligned_excess.mean() * self.annualization - factor_contribution.sum())

        attribution = pd.concat(
            [
                pd.Series({"alpha": alpha_annualized}),
                factor_contribution.rename("factor_contribution"),
            ]
        )

        return factor_beta, attribution
