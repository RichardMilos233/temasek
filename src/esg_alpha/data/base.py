from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from esg_alpha.models import RawDataBundle


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self) -> RawDataBundle:
        """Load returns, ESG records, and supporting risk model data."""


class ESGImputer(ABC):
    @abstractmethod
    def impute(self, esg_features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing ESG feature values in a point-in-time panel."""


class SectorNormalizer(ABC):
    @abstractmethod
    def normalize(self, esg_features: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
        """Normalize ESG features within sectors per date."""
