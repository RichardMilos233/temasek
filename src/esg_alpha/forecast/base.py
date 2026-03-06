from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class AlphaForecaster(ABC):
    @abstractmethod
    def forecast(self, signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Convert normalized signal into expected-return forecasts."""
