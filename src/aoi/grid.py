"""Build the master H3 grid for Greater Phoenix."""

from __future__ import annotations

import argparse

import geopandas as gpd

from src.aoi.loader import load_aoi, load_aoi_geometry
from src.utils.config import ProjectConfig, load_config
from src.utils.h3_utils import cell_area_m2, cell_to_centroid, cell_to_polygon, polygon_to_cells
from src.utils.io import write_geoparquet
from src.utils.logging_utils import get_logger
from src.utils.paths import get_paths


LOGGER = get_logger(__name__)


def build_master_grid(config: ProjectConfig | None = None) -> gpd.GeoDataFrame:
    """Generate and persist the H3 master grid for the configured AOI."""
    config = config or load_config()
    paths = get_paths()
    output_path = paths.grid_path(config.hex_resolution)

    aoi = load_aoi(config)
    geometry = load_aoi_geometry(config)
    hex_ids = polygon_to_cells(geometry, config.hex_resolution)
    if not hex_ids:
        raise ValueError(
            f"No H3 cells were generated for {config.aoi_name} at resolution {config.hex_resolution}."
        )

    records: list[dict[str, float | str | object]] = []
    for cell_id in hex_ids:
        centroid_lon, centroid_lat = cell_to_centroid(cell_id)
        records.append(
            {
                "hex_id": cell_id,
                "geometry": cell_to_polygon(cell_id),
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
                "area_m2": cell_area_m2(cell_id),
            }
        )

    grid = gpd.GeoDataFrame(records, geometry="geometry", crs=config.analysis_crs)
    grid = grid[grid.intersects(aoi.geometry.iloc[0])].reset_index(drop=True)
    write_geoparquet(grid, output_path)
    LOGGER.info("Wrote %s H3 cells to %s", len(grid), output_path)
    return grid


def main() -> None:
    """CLI entrypoint for grid generation."""
    parser = argparse.ArgumentParser(description="Build the Greater Phoenix H3 master grid.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    build_master_grid(load_config(args.config))


if __name__ == "__main__":
    main()
