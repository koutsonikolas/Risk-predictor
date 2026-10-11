"""Normalization helpers for deterministic scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_series(
    values: pd.Series,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    scale_max: float = 100.0,
) -> pd.Series:
    """Normalize a numeric series to a 0-100 scale with quantile clipping."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)

    lower = float(numeric.quantile(lower_quantile))
    upper = float(numeric.quantile(upper_quantile))
    if lower == upper:
        return pd.Series(np.where(numeric.notna(), scale_max / 2.0, 0.0), index=values.index, dtype=float)

    clipped = numeric.clip(lower=lower, upper=upper)
    normalized = (clipped - lower) / (upper - lower)
    return normalized.fillna(0.0) * scale_max


def mean_of_available(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Compute row-wise means while ignoring missing values."""
    if not columns:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return frame[columns].mean(axis=1, skipna=True).fillna(0.0)
