from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esg_alpha.config import PipelineConfig
from esg_alpha.factory import build_pipeline


def test_pipeline_smoke_run() -> None:
    config = PipelineConfig.from_dict(
        {
            "data": {
                "ingestor": "mock_pit",
                "imputer": "forward_fill_median",
                "normalizer": "sector_zscore",
                "ingestor_params": {
                    "start_date": "2022-01-01",
                    "end_date": "2023-12-31",
                    "seed": 11,
                },
            },
            "signal": {
                "builder": "materiality_momentum",
                "orthogonalizer": "ols",
                "builder_params": {"lookback": 21},
            },
            "forecast": {"method": "rank_to_return", "params": {"scale": 0.01}},
            "portfolio": {
                "constructor": "quantile_long_short",
                "params": {
                    "long_quantile": 0.8,
                    "short_quantile": 0.2,
                    "gross_leverage": 1.0,
                },
            },
            "backtest": {
                "params": {
                    "execution_lag": 1,
                    "transaction_cost_bps": 5.0,
                    "slippage_bps": 2.0,
                }
            },
        }
    )

    pipeline = build_pipeline(config)
    result = pipeline.run()

    assert not result.backtest.net_returns.empty
    assert "information_ratio" in result.evaluation.metrics
    assert result.target_weights.shape == result.processed_data.returns.shape
