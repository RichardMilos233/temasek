from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class RawDataBundle:
    returns: pd.DataFrame
    esg_records: pd.DataFrame
    sectors: pd.Series
    style_factors: Dict[str, pd.DataFrame]
    factor_returns: pd.DataFrame
    benchmark_returns: Optional[pd.Series] = None
    benchmark_weights: Optional[pd.DataFrame] = None
    news: Optional[pd.DataFrame] = None


@dataclass
class ProcessedDataBundle:
    returns: pd.DataFrame
    esg_features: pd.DataFrame
    sectors: pd.Series
    style_factors: Dict[str, pd.DataFrame]
    factor_returns: pd.DataFrame
    benchmark_returns: pd.Series
    benchmark_weights: Optional[pd.DataFrame] = None
    news: Optional[pd.DataFrame] = None


@dataclass
class BacktestResult:
    target_weights: pd.DataFrame
    executed_weights: pd.DataFrame
    gross_returns: pd.Series
    transaction_costs: pd.Series
    net_returns: pd.Series
    excess_returns: pd.Series
    turnover: pd.Series


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    ic_series: pd.Series
    factor_betas: pd.Series
    attribution_contribution: pd.Series


@dataclass
class PipelineResult:
    processed_data: ProcessedDataBundle
    raw_signal: pd.DataFrame
    orthogonal_signal: pd.DataFrame
    expected_returns: pd.DataFrame
    target_weights: pd.DataFrame
    backtest: BacktestResult
    evaluation: EvaluationResult
