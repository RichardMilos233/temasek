from __future__ import annotations

from copy import deepcopy
from typing import Any

from esg_alpha.backtest import BacktestEngine, PerformanceEvaluator
from esg_alpha.config import PipelineConfig
from esg_alpha.data import (
    CrossSectionalKNNImputer,
    DataProcessor,
    ForwardFillMedianImputer,
    MockPointInTimeIngestor,
    NoOpSectorNormalizer,
    SectorZScoreNormalizer,
)
from esg_alpha.forecast import ICScaledRankForecaster, RankToReturnForecaster
from esg_alpha.pipeline import ESGAlphaPipeline
from esg_alpha.portfolio import ConstrainedMeanVarianceConstructor, QuantileLongShortConstructor
from esg_alpha.signal import (
    CrossSectionalOLSOrthogonalizer,
    FinBERTNewsSignalBuilder,
    LexiconNLPESGSignalBuilder,
    MaterialityMomentumSignalBuilder,
    NoOpOrthogonalizer,
)


def _build_data_ingestor(name: str, params: dict[str, Any]):
    if name == "mock_pit":
        return MockPointInTimeIngestor(**params)
    raise ValueError(f"Unsupported data ingestor: {name}")


def _build_imputer(name: str, params: dict[str, Any]):
    if name == "forward_fill_median":
        return ForwardFillMedianImputer()
    if name == "knn":
        return CrossSectionalKNNImputer(**params)
    raise ValueError(f"Unsupported imputer: {name}")


def _build_normalizer(name: str, params: dict[str, Any]):
    if name == "sector_zscore":
        return SectorZScoreNormalizer()
    if name == "none":
        return NoOpSectorNormalizer()
    raise ValueError(f"Unsupported normalizer: {name}")


def _build_signal_builder(name: str, params: dict[str, Any]):
    if name == "materiality_momentum":
        return MaterialityMomentumSignalBuilder(**params)
    if name == "lexicon_nlp":
        return LexiconNLPESGSignalBuilder(**params)
    if name == "finbert_nlp":
        return FinBERTNewsSignalBuilder(**params)
    raise ValueError(f"Unsupported signal builder: {name}")


def _build_orthogonalizer(name: str, params: dict[str, Any]):
    if name == "ols":
        return CrossSectionalOLSOrthogonalizer(**params)
    if name == "none":
        return NoOpOrthogonalizer()
    raise ValueError(f"Unsupported orthogonalizer: {name}")


def _build_forecaster(name: str, params: dict[str, Any]):
    if name == "rank_to_return":
        return RankToReturnForecaster(**params)
    if name == "ic_scaled_rank":
        return ICScaledRankForecaster(**params)
    raise ValueError(f"Unsupported forecaster: {name}")


def _build_portfolio_constructor(name: str, params: dict[str, Any]):
    params = deepcopy(params)

    if name == "quantile_long_short":
        return QuantileLongShortConstructor(**params)

    if name == "constrained_mean_variance":
        fallback_name = params.pop("fallback_constructor", "quantile_long_short")
        if fallback_name == "constrained_mean_variance":
            fallback_name = "quantile_long_short"

        fallback_constructor = _build_portfolio_constructor(fallback_name, {})
        return ConstrainedMeanVarianceConstructor(
            fallback_constructor=fallback_constructor,
            **params,
        )

    raise ValueError(f"Unsupported portfolio constructor: {name}")


def build_pipeline(config: PipelineConfig) -> ESGAlphaPipeline:
    data_ingestor = _build_data_ingestor(config.data.ingestor, config.data.ingestor_params)
    imputer = _build_imputer(config.data.imputer, config.data.imputer_params)
    normalizer = _build_normalizer(config.data.normalizer, config.data.normalizer_params)
    data_processor = DataProcessor(imputer=imputer, normalizer=normalizer)

    signal_builder = _build_signal_builder(config.signal.builder, config.signal.builder_params)
    orthogonalizer = _build_orthogonalizer(
        config.signal.orthogonalizer,
        config.signal.orthogonalizer_params,
    )
    forecaster = _build_forecaster(config.forecast.method, config.forecast.params)

    portfolio_constructor = _build_portfolio_constructor(
        config.portfolio.constructor,
        config.portfolio.params,
    )

    backtester = BacktestEngine(**config.backtest.params)
    evaluator = PerformanceEvaluator()

    return ESGAlphaPipeline(
        data_ingestor=data_ingestor,
        data_processor=data_processor,
        signal_builder=signal_builder,
        orthogonalizer=orthogonalizer,
        forecaster=forecaster,
        portfolio_constructor=portfolio_constructor,
        backtester=backtester,
        evaluator=evaluator,
    )
