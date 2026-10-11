"""Shared helpers for yearly Earth Engine ingestion modules."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ee

from src.utils.config import ProjectConfig, load_config
from src.utils.exceptions import MissingInputError
from src.utils.io import read_geoparquet, write_parquet
from src.utils.logging_utils import get_logger
from src.utils.paths import get_paths


LOGGER = get_logger(__name__)


def load_master_grid(config: ProjectConfig | None = None):
    """Load the persisted H3 master grid."""
    config = config or load_config()
    paths = get_paths()
    grid_path = paths.grid_path(config.hex_resolution)
    return read_geoparquet(
        grid_path,
        "Master H3 grid is missing. Run `python -m src.cli build-grid --config configs/phoenix.yaml` first.",
    )


def grid_to_ee_feature_collection(ee: Any, grid) -> Any:
    """Convert the local H3 grid to an Earth Engine FeatureCollection."""
    features = []
    for row in grid.itertuples(index=False):
        features.append(
            ee.Feature(
                ee.Geometry(row.geometry.__geo_interface__),
                {"hex_id": row.hex_id},
            )
        )
    return ee.FeatureCollection(features)


def feature_collection_to_frame(feature_collection: Any) -> pd.DataFrame:
    """Convert an EE FeatureCollection to a pandas DataFrame."""
    try:
        import geemap

        frame = geemap.ee_to_df(feature_collection)
        if "system:index" in frame.columns:
            frame = frame.drop(columns=["system:index"])
        return frame
    except Exception:
        payload = feature_collection.getInfo()
        records = [feature["properties"] for feature in payload["features"]]
        return pd.DataFrame(records)


def reduce_image_by_hex(
    image: Any,
    grid_fc: Any,
    scale: int,
    crs: str | None = None,
    tile_scale: int = 4,
) -> pd.DataFrame:
    """Run reduceRegions on a hex grid and return a DataFrame."""
    return reduce_image_with_reducer(
        image=image,
        grid_fc=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        crs=crs,
        tile_scale=tile_scale,
    )


def reduce_image_with_reducer(
    image: Any,
    grid_fc: Any,
    reducer: Any,
    scale: int,
    crs: str | None = None,
    tile_scale: int = 4,
) -> pd.DataFrame:
    """Reduce an image over the hex grid using a caller-provided reducer."""
    reduced = image.reduceRegions(
        collection=grid_fc,
        reducer=reducer,
        scale=scale,
        crs=crs,
        tileScale=tile_scale,
    )
    return feature_collection_to_frame(reduced)


def build_hex_year_template(grid, years: list[int], columns: list[str]) -> pd.DataFrame:
    """Return a full hex-year template with missing feature columns added."""
    base = pd.MultiIndex.from_product(
        [grid["hex_id"].tolist(), years],
        names=["hex_id", "year"],
    ).to_frame(index=False)
    for column in columns:
        if column not in base.columns:
            base[column] = pd.NA
    return base


def finalize_hex_year_output(
    frame: pd.DataFrame,
    grid,
    years: list[int],
    columns: list[str],
    output_path,
) -> pd.DataFrame:
    """Reindex a yearly feature set to the full hex-year template and write parquet."""
    template = build_hex_year_template(grid, years, columns)
    merged = template.merge(frame, on=["hex_id", "year"], how="left", suffixes=("", "_value"))

    for column in columns:
        value_column = f"{column}_value"
        if value_column in merged.columns:
            merged[column] = merged[value_column]
            merged = merged.drop(columns=[value_column])

    merged = merged[columns].sort_values(["year", "hex_id"]).reset_index(drop=True)
    write_parquet(merged, output_path)
    LOGGER.info("Wrote %s", output_path)
    return merged


def ensure_yearly_results(results: list[pd.DataFrame], dataset_name: str) -> pd.DataFrame:
    """Concatenate yearly DataFrames with a clear error if none were produced."""
    if not results:
        raise MissingInputError(f"{dataset_name} did not produce any yearly results.")
    return pd.concat(results, ignore_index=True)
