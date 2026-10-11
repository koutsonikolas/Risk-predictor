"""AOI loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely import make_valid

from src.utils.config import ProjectConfig, load_config
from src.utils.exceptions import DataValidationError, MissingInputError
from src.utils.io import require_file
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def _aoi_help_text(path: Path) -> str:
    return (
        "Greater Phoenix AOI is required before the pipeline can build the master H3 grid. "
        "Add a valid GeoJSON polygon file at "
        f"{path}. See data/raw/aoi/README.md for the expected workflow."
    )


def load_aoi(config: ProjectConfig | None = None) -> gpd.GeoDataFrame:
    """Load, validate, and normalize the Greater Phoenix AOI."""
    config = config or load_config()
    require_file(config.aoi_path, _aoi_help_text(config.aoi_path))

    gdf = gpd.read_file(config.aoi_path)
    if gdf.empty:
        raise MissingInputError(_aoi_help_text(config.aoi_path))

    if gdf.crs is None:
        LOGGER.warning("AOI file has no CRS metadata. Assuming %s.", config.analysis_crs)
        gdf = gdf.set_crs(config.analysis_crs)

    gdf = gdf.to_crs(config.analysis_crs)
    gdf = gdf[gdf.geometry.notna()].copy()
    if gdf.empty:
        raise DataValidationError("AOI file does not contain any valid geometry rows.")

    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise DataValidationError("AOI geometry is empty after validation.")

    allowed = {"Polygon", "MultiPolygon"}
    invalid_types = sorted(set(gdf.geometry.geom_type) - allowed)
    if invalid_types:
        raise DataValidationError(
            f"AOI geometry must be Polygon or MultiPolygon, found: {', '.join(invalid_types)}"
        )

    dissolved = gdf[["geometry"]].dissolve().reset_index(drop=True)
    dissolved["aoi_name"] = config.aoi_name
    return dissolved[["aoi_name", "geometry"]]


def load_aoi_geometry(config: ProjectConfig | None = None):
    """Return the dissolved AOI geometry as a shapely object."""
    return load_aoi(config).geometry.iloc[0]
