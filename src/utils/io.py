"""IO helpers for parquet and vector data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from src.utils.exceptions import MissingInputError
from src.utils.paths import ensure_parent


def require_file(path: Path, help_text: str) -> Path:
    """Ensure a required file exists and is not empty."""
    if not path.exists() or path.stat().st_size == 0:
        raise MissingInputError(f"{help_text} Expected file: {path}")
    return path


def read_vector(path: Path) -> gpd.GeoDataFrame:
    """Read a vector file, including GeoParquet."""
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def write_geoparquet(frame: gpd.GeoDataFrame, path: Path) -> Path:
    """Write a GeoDataFrame to parquet."""
    ensure_parent(path)
    frame.to_parquet(path, index=False)
    return path


def read_geoparquet(path: Path, help_text: str) -> gpd.GeoDataFrame:
    """Read a geoparquet file with a helpful missing-file error."""
    require_file(path, help_text)
    return gpd.read_parquet(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write a tabular parquet file."""
    ensure_parent(path)
    frame.to_parquet(path, index=False)
    return path


def read_parquet(path: Path, help_text: str) -> pd.DataFrame:
    """Read a parquet file with a helpful missing-file error."""
    require_file(path, help_text)
    return pd.read_parquet(path)


def first_existing_path(candidates: Iterable[Path]) -> Path | None:
    """Return the first existing path in an iterable."""
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None
