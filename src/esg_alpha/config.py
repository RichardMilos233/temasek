from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataStageConfig:
    ingestor: str = "mock_pit"
    imputer: str = "forward_fill_median"
    normalizer: str = "sector_zscore"
    ingestor_params: Dict[str, Any] = field(default_factory=dict)
    imputer_params: Dict[str, Any] = field(default_factory=dict)
    normalizer_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalStageConfig:
    builder: str = "materiality_momentum"
    orthogonalizer: str = "ols"
    builder_params: Dict[str, Any] = field(default_factory=lambda: {"lookback": 21})
    orthogonalizer_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastStageConfig:
    method: str = "rank_to_return"
    params: Dict[str, Any] = field(default_factory=lambda: {"scale": 0.02})


@dataclass
class PortfolioStageConfig:
    constructor: str = "quantile_long_short"
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "long_quantile": 0.8,
            "short_quantile": 0.2,
            "gross_leverage": 1.0,
        }
    )


@dataclass
class BacktestStageConfig:
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "execution_lag": 1,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
        }
    )


@dataclass
class PipelineConfig:
    data: DataStageConfig = field(default_factory=DataStageConfig)
    signal: SignalStageConfig = field(default_factory=SignalStageConfig)
    forecast: ForecastStageConfig = field(default_factory=ForecastStageConfig)
    portfolio: PortfolioStageConfig = field(default_factory=PortfolioStageConfig)
    backtest: BacktestStageConfig = field(default_factory=BacktestStageConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        data_raw = raw.get("data", {})
        signal_raw = raw.get("signal", {})

        # Backward compatibility: allow legacy single "params" object.
        legacy_data_params = data_raw.get("params", {})
        legacy_signal_params = signal_raw.get("params", {})

        return cls(
            data=DataStageConfig(
                ingestor=data_raw.get("ingestor", "mock_pit"),
                imputer=data_raw.get("imputer", "forward_fill_median"),
                normalizer=data_raw.get("normalizer", "sector_zscore"),
                ingestor_params=data_raw.get("ingestor_params", legacy_data_params),
                imputer_params=data_raw.get("imputer_params", {}),
                normalizer_params=data_raw.get("normalizer_params", {}),
            ),
            signal=SignalStageConfig(
                builder=signal_raw.get("builder", "materiality_momentum"),
                orthogonalizer=signal_raw.get("orthogonalizer", "ols"),
                builder_params=signal_raw.get("builder_params", legacy_signal_params),
                orthogonalizer_params=signal_raw.get("orthogonalizer_params", {}),
            ),
            forecast=ForecastStageConfig(**raw.get("forecast", {})),
            portfolio=PortfolioStageConfig(**raw.get("portfolio", {})),
            backtest=BacktestStageConfig(**raw.get("backtest", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": {
                "ingestor": self.data.ingestor,
                "imputer": self.data.imputer,
                "normalizer": self.data.normalizer,
                "ingestor_params": self.data.ingestor_params,
                "imputer_params": self.data.imputer_params,
                "normalizer_params": self.data.normalizer_params,
            },
            "signal": {
                "builder": self.signal.builder,
                "orthogonalizer": self.signal.orthogonalizer,
                "builder_params": self.signal.builder_params,
                "orthogonalizer_params": self.signal.orthogonalizer_params,
            },
            "forecast": {
                "method": self.forecast.method,
                "params": self.forecast.params,
            },
            "portfolio": {
                "constructor": self.portfolio.constructor,
                "params": self.portfolio.params,
            },
            "backtest": {"params": self.backtest.params},
        }
