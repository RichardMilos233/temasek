from __future__ import annotations

import pandas as pd

from esg_alpha.models import BacktestResult


class BacktestEngine:
    def __init__(
        self,
        execution_lag: int = 1,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.execution_lag = execution_lag
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps

    def run(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
    ) -> BacktestResult:
        aligned_weights = target_weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
        executed_weights = aligned_weights.shift(self.execution_lag).fillna(0.0)

        gross_returns = (executed_weights * returns).sum(axis=1)

        turnover = executed_weights.diff().abs().sum(axis=1)
        if not turnover.empty:
            turnover.iloc[0] = executed_weights.iloc[0].abs().sum()

        total_cost_rate = (self.transaction_cost_bps + self.slippage_bps) / 10000.0
        transaction_costs = turnover * total_cost_rate
        net_returns = gross_returns - transaction_costs

        if benchmark_returns is None:
            benchmark_aligned = returns.mean(axis=1)
        else:
            benchmark_aligned = benchmark_returns.reindex(returns.index).fillna(0.0)

        excess_returns = net_returns - benchmark_aligned

        return BacktestResult(
            target_weights=aligned_weights,
            executed_weights=executed_weights,
            gross_returns=gross_returns,
            transaction_costs=transaction_costs,
            net_returns=net_returns,
            excess_returns=excess_returns,
            turnover=turnover,
        )
