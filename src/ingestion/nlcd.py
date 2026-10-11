"""NLCD yearly hex summaries."""

from __future__ import annotations

import argparse

import ee
import pandas as pd

from src.ingestion.common import (
    finalize_hex_year_output,
    grid_to_ee_feature_collection,
    load_master_grid,
)
from src.ingestion.ee_utils import aoi_to_ee_geometry, initialize_earth_engine
from src.utils.config import ProjectConfig, load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import get_paths

LOGGER = get_logger(__name__)

NLCD_AVAILABLE_YEARS = [2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021]

OUTPUT_COLUMNS = [
    "hex_id",
    "year",
    "pct_developed",
    "pct_shrub_scrub",
    "pct_grassland",
    "pct_barren",
    "pct_tree_canopy",
    "pct_impervious",
    "delta_pct_developed",
    "delta_pct_barren",
    "delta_pct_tree_canopy",
]


def _closest_nlcd_year(year: int) -> int:
    return max(candidate for candidate in NLCD_AVAILABLE_YEARS if candidate <= year)


def _get_nlcd_image(source_year: int) -> ee.Image:
    if source_year <= 2019:
        collection = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")
    elif source_year == 2021:
        collection = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
    else:
        raise ValueError(f"Unsupported NLCD source year: {source_year}")

    return ee.Image(
        collection.filter(ee.Filter.eq("system:index", str(source_year))).first()
    )


def _nlcd_summary_image(image: ee.Image) -> ee.Image:
    landcover = image.select("landcover")

    developed = (
        landcover.remap([21, 22, 23, 24], [1, 1, 1, 1], 0)
        .rename("pct_developed")
        .multiply(100)
    )
    shrub = landcover.eq(52).rename("pct_shrub_scrub").multiply(100)
    grass = landcover.eq(71).rename("pct_grassland").multiply(100)
    barren = landcover.eq(31).rename("pct_barren").multiply(100)
    trees = (
        landcover.remap([41, 42, 43], [1, 1, 1], 0)
        .rename("pct_tree_canopy")
        .multiply(100)
    )
    impervious = image.select("impervious").rename("pct_impervious")

    return ee.Image.cat([developed, shrub, grass, barren, trees, impervious])


def _summarize_year(grid_fc, year: int, scale: int) -> pd.DataFrame:
    source_year = _closest_nlcd_year(year)
    image = _get_nlcd_image(source_year)
    summary = _nlcd_summary_image(image)

    reduced = summary.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )

    frame = pd.DataFrame(
        [feature["properties"] for feature in reduced.getInfo()["features"]]
    )

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "hex_id",
                "year",
                "pct_developed",
                "pct_shrub_scrub",
                "pct_grassland",
                "pct_barren",
                "pct_tree_canopy",
                "pct_impervious",
            ]
        )

    frame["year"] = year

    return frame[
        [
            "hex_id",
            "year",
            "pct_developed",
            "pct_shrub_scrub",
            "pct_grassland",
            "pct_barren",
            "pct_tree_canopy",
            "pct_impervious",
        ]
    ]


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Run NLCD yearly feature extraction and write parquet output."""
    config = config or load_config()
    initialize_earth_engine(config)

    # Keep AOI loading here so AOI validation still happens consistently.
    _ = aoi_to_ee_geometry(config)

    grid = load_master_grid(config)
    grid_fc = grid_to_ee_feature_collection(ee, grid)
    paths = get_paths()

    yearly_frames = [
        _summarize_year(grid_fc, year, config.earth_engine.nlcd_scale)
        for year in config.analysis_years
    ]

    combined = pd.concat(yearly_frames, ignore_index=True)
    combined = combined.sort_values(["hex_id", "year"]).reset_index(drop=True)

    combined["delta_pct_developed"] = (
        combined.groupby("hex_id")["pct_developed"].diff().fillna(0.0)
    )
    combined["delta_pct_barren"] = (
        combined.groupby("hex_id")["pct_barren"].diff().fillna(0.0)
    )
    combined["delta_pct_tree_canopy"] = (
        combined.groupby("hex_id")["pct_tree_canopy"].diff().fillna(0.0)
    )

    return finalize_hex_year_output(
        combined[OUTPUT_COLUMNS],
        grid=grid,
        years=config.analysis_years,
        columns=OUTPUT_COLUMNS,
        output_path=paths.nlcd_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest NLCD yearly hex summaries."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    run(load_config(args.config))


if __name__ == "__main__":
    main()