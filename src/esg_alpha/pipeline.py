from __future__ import annotations

from esg_alpha.backtest import BacktestEngine, PerformanceEvaluator
from esg_alpha.data import DataIngestor, DataProcessor
from esg_alpha.forecast import AlphaForecaster
from esg_alpha.models import PipelineResult
from esg_alpha.portfolio import PortfolioConstructor
from esg_alpha.signal import SignalBuilder, SignalOrthogonalizer


class ESGAlphaPipeline:
    def __init__(
        self,
        data_ingestor: DataIngestor,
        data_processor: DataProcessor,
        signal_builder: SignalBuilder,
        orthogonalizer: SignalOrthogonalizer,
        forecaster: AlphaForecaster,
        portfolio_constructor: PortfolioConstructor,
        backtester: BacktestEngine,
        evaluator: PerformanceEvaluator,
    ) -> None:
        self.data_ingestor = data_ingestor
        self.data_processor = data_processor
        self.signal_builder = signal_builder
        self.orthogonalizer = orthogonalizer
        self.forecaster = forecaster
        self.portfolio_constructor = portfolio_constructor
        self.backtester = backtester
        self.evaluator = evaluator

    def run(self) -> PipelineResult:
        raw_data = self.data_ingestor.ingest()
        processed_data = self.data_processor.process(raw_data)

        raw_signal = self.signal_builder.build(processed_data)
        orthogonal_signal = self.orthogonalizer.orthogonalize(raw_signal, processed_data)
        expected_returns = self.forecaster.forecast(orthogonal_signal, processed_data.returns)

        target_weights = self.portfolio_constructor.construct(
            expected_returns=expected_returns,
            realized_returns=processed_data.returns,
            sectors=processed_data.sectors,
            benchmark_weights=processed_data.benchmark_weights,
        )

        backtest_result = self.backtester.run(
            target_weights=target_weights,
            returns=processed_data.returns,
            benchmark_returns=processed_data.benchmark_returns,
        )

        evaluation = self.evaluator.evaluate(
            backtest=backtest_result,
            signal=orthogonal_signal,
            returns=processed_data.returns,
            factor_returns=processed_data.factor_returns,
        )

        return PipelineResult(
            processed_data=processed_data,
            raw_signal=raw_signal,
            orthogonal_signal=orthogonal_signal,
            expected_returns=expected_returns,
            target_weights=target_weights,
            backtest=backtest_result,
            evaluation=evaluation,
        )
