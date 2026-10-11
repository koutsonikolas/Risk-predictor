"""Aggregate local building footprints into the master H3 grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.aoi.loader import load_aoi
from src.ingestion.common import load_master_grid
from src.utils.config import ProjectConfig, load_config
from src.utils.exceptions import DataValidationError, MissingInputError
from src.utils.h3_utils import polygon_area_m2
from src.utils.io import first_existing_path, read_vector, write_parquet
from src.utils.logging_utils import get_logger
from src.utils.paths import get_paths


LOGGER = get_logger(__name__)

OUTPUT_COLUMNS = [
    "hex_id",
    "building_count",
    "building_area_sum_m2",
    "building_density",
    "mean_building_area_m2",
]


def _resolve_buildings_path(config: ProjectConfig) -> Path:
    configured = config.buildings_path
    candidates = [
        configured,
        configured.with_suffix(".parquet"),
        configured.with_suffix(".gpkg"),
        configured.with_suffix(".shp"),
        get_paths().raw_buildings_dir / "buildings.geojson",
        get_paths().raw_buildings_dir / "buildings.parquet",
        get_paths().raw_buildings_dir / "buildings.gpkg",
        get_paths().raw_buildings_dir / "buildings.shp",
    ]
    selected = first_existing_path(candidates)
    if selected is None:
        raise MissingInputError(
            "A local building-footprint dataset is required. Add one of "
            "`data/raw/buildings/buildings.geojson`, `buildings.parquet`, `buildings.gpkg`, "
            f"or `buildings.shp`. Configured path: {configured}"
        )
    return selected


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Read local building footprints, aggregate to H3, and write parquet output."""
    config = config or load_config()
    paths = get_paths()
    grid = load_master_grid(config)
    aoi = load_aoi(config)
    buildings_path = _resolve_buildings_path(config)

    buildings = read_vector(buildings_path)
    if buildings.empty:
        raise DataValidationError(f"Building-footprint file is empty: {buildings_path}")
    if buildings.crs is None:
        LOGGER.warning("Building-footprint file has no CRS metadata. Assuming %s.", config.analysis_crs)
        buildings = buildings.set_crs(config.analysis_crs)

    buildings = buildings.to_crs(config.analysis_crs)
    if not buildings.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise DataValidationError("Building footprints must be polygon geometries.")

    clipped = gpd.clip(buildings, aoi)
    if clipped.empty:
        result = grid[["hex_id", "area_m2"]].copy()
        result["building_count"] = 0
        result["building_area_sum_m2"] = 0.0
        result["building_density"] = 0.0
        result["mean_building_area_m2"] = 0.0
        result = result[OUTPUT_COLUMNS]
        write_parquet(result, paths.buildings_features)
        return result

    clipped = clipped.copy()
    clipped["building_area_sum_m2"] = clipped.geometry.apply(polygon_area_m2)
    clipped["centroid"] = clipped.geometry.representative_point()
    centroids = gpd.GeoDataFrame(
        clipped.drop(columns="geometry"),
        geometry="centroid",
        crs=config.analysis_crs,
    )
    joined = gpd.sjoin(centroids, grid[["hex_id", "geometry", "area_m2"]], how="left", predicate="within")
    joined = joined.dropna(subset=["hex_id"]).copy()

    aggregated = (
        joined.groupby("hex_id")
        .agg(
            building_count=("hex_id", "size"),
            building_area_sum_m2=("building_area_sum_m2", "sum"),
            mean_building_area_m2=("building_area_sum_m2", "mean"),
        )
        .reset_index()
    )
    result = grid[["hex_id", "area_m2"]].merge(aggregated, on="hex_id", how="left")
    result["building_count"] = result["building_count"].fillna(0).astype(int)
    result["building_area_sum_m2"] = result["building_area_sum_m2"].fillna(0.0)
    result["mean_building_area_m2"] = result["mean_building_area_m2"].fillna(0.0)
    result["building_density"] = result["building_count"] / (result["area_m2"] / 1_000_000.0)
    result = result[OUTPUT_COLUMNS].sort_values("hex_id").reset_index(drop=True)
    write_parquet(result, paths.buildings_features)
    LOGGER.info("Wrote %s", paths.buildings_features)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate local building footprints to the H3 grid.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
