from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional z-score normalization on each date (row)."""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    z = frame.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0.0)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return 0.0
    return float(numerator / denominator)
