from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from esg_alpha.models import ProcessedDataBundle


class SignalBuilder(ABC):
    @abstractmethod
    def build(self, data: ProcessedDataBundle) -> pd.DataFrame:
        """Build a date x ticker raw alpha signal."""


class SignalOrthogonalizer(ABC):
    @abstractmethod
    def orthogonalize(self, signal: pd.DataFrame, data: ProcessedDataBundle) -> pd.DataFrame:
        """Neutralize signal against chosen risk factors."""
