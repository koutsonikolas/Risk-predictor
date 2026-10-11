"""Dynamic World yearly hex summaries."""

from __future__ import annotations

import argparse

import ee
import pandas as pd

from src.ingestion.common import finalize_hex_year_output, grid_to_ee_feature_collection, load_master_grid
from src.ingestion.ee_utils import aoi_to_ee_geometry, initialize_earth_engine, year_date_filter
from src.utils.config import ProjectConfig, load_config
from src.utils.paths import get_paths


OUTPUT_COLUMNS = [
    "hex_id",
    "year",
    "dw_built_prob_mean",
    "dw_bare_prob_mean",
    "dw_trees_prob_mean",
    "dw_grass_prob_mean",
    "dw_shrub_prob_mean",
    "dw_built_share",
    "dw_bare_share",
    "dw_vegetation_share",
    "dw_transition_veg_to_built",
    "dw_transition_veg_to_bare",
]


def _dw_collection(aoi, year: int) -> ee.ImageCollection:
    return ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi).filter(year_date_filter(ee, year))


def _empty_year(grid, year: int) -> pd.DataFrame:
    frame = pd.DataFrame({"hex_id": grid["hex_id"], "year": year})
    for column in OUTPUT_COLUMNS[2:]:
        frame[column] = pd.NA
    return frame


def _summarize_year(aoi, grid, grid_fc, year: int, scale: int) -> pd.DataFrame:
    if year < 2015:
        return _empty_year(grid, year)

    collection = _dw_collection(aoi, year)
    if collection.size().getInfo() == 0:
        return _empty_year(grid, year)

    annual_prob = collection.select(
        ["built", "bare", "trees", "grass", "shrub_and_scrub"]
    ).mean().rename(
        [
            "dw_built_prob_mean",
            "dw_bare_prob_mean",
            "dw_trees_prob_mean",
            "dw_grass_prob_mean",
            "dw_shrub_prob_mean",
        ]
    )
    annual_label = collection.select("label").mode()
    built_share = annual_label.eq(6).rename("dw_built_share")
    bare_share = annual_label.eq(7).rename("dw_bare_share")
    vegetation_share = annual_label.remap([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 0).rename("dw_vegetation_share")

    if year == 2015:
        transition_built = ee.Image.constant(0).rename("dw_transition_veg_to_built").clip(aoi)
        transition_bare = ee.Image.constant(0).rename("dw_transition_veg_to_bare").clip(aoi)
    else:
        previous = _dw_collection(aoi, year - 1)
        if previous.size().getInfo() == 0:
            transition_built = ee.Image.constant(0).rename("dw_transition_veg_to_built").clip(aoi)
            transition_bare = ee.Image.constant(0).rename("dw_transition_veg_to_bare").clip(aoi)
        else:
            previous_label = previous.select("label").mode()
            previous_vegetation = previous_label.remap([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 0)
            transition_built = previous_vegetation.And(annual_label.eq(6)).rename("dw_transition_veg_to_built")
            transition_bare = previous_vegetation.And(annual_label.eq(7)).rename("dw_transition_veg_to_bare")

    summary = ee.Image.cat(
        [annual_prob, built_share, bare_share, vegetation_share, transition_built, transition_bare]
    )
    reduced = summary.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )
    frame = pd.DataFrame([feature["properties"] for feature in reduced.getInfo()["features"]])
    frame["year"] = year
    return frame[OUTPUT_COLUMNS]


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Run Dynamic World yearly feature extraction and write parquet output."""
    config = config or load_config()
    initialize_earth_engine(config)
    aoi = aoi_to_ee_geometry(config)
    grid = load_master_grid(config)
    grid_fc = grid_to_ee_feature_collection(ee, grid)
    paths = get_paths()

    frames = [
        _summarize_year(aoi, grid, grid_fc, year, config.earth_engine.dynamic_world_scale)
        for year in config.analysis_years
    ]
    combined = pd.concat(frames, ignore_index=True)
    return finalize_hex_year_output(
        combined,
        grid=grid,
        years=config.analysis_years,
        columns=OUTPUT_COLUMNS,
        output_path=paths.dynamic_world_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Dynamic World yearly hex summaries.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
