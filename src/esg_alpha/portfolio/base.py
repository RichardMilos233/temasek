from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PortfolioConstructor(ABC):
    @abstractmethod
    def construct(
        self,
        expected_returns: pd.DataFrame,
        realized_returns: pd.DataFrame,
        sectors: pd.Series,
        benchmark_weights: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build target portfolio weights for each date."""
