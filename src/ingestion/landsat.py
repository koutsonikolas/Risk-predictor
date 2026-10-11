"""Landsat Collection 2 Level-2 yearly summer summaries."""

from __future__ import annotations

import argparse

import ee
import pandas as pd

from src.ingestion.common import finalize_hex_year_output, grid_to_ee_feature_collection, load_master_grid
from src.ingestion.ee_utils import aoi_to_ee_geometry, initialize_earth_engine, seasonal_date_filter
from src.utils.config import ProjectConfig, load_config
from src.utils.logging_utils import get_logger
from src.utils.paths import get_paths


LOGGER = get_logger(__name__)

OUTPUT_COLUMNS = [
    "hex_id",
    "year",
    "ndvi_mean",
    "ndvi_std",
    "ndbi_mean",
    "bsi_mean",
    "lst_mean_summer",
    "lst_p90_summer",
    "valid_pixel_fraction",
]


def _mask_and_derive(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
        .And(image.select("QA_RADSAT").eq(0))
    )

    optical = image.select(["SR_B2", "SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
    blue = optical.select("SR_B2")
    red = optical.select("SR_B4")
    nir = optical.select("SR_B5")
    swir = optical.select("SR_B6")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    ndbi = swir.subtract(nir).divide(swir.add(nir)).rename("ndbi")
    bsi = (
        swir.add(red)
        .subtract(nir.add(blue))
        .divide(swir.add(red).add(nir).add(blue))
        .rename("bsi")
    )
    lst = image.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst")
    valid = ee.Image.constant(1).rename("valid")

    return (
        ee.Image.cat([ndvi, ndbi, bsi, lst, valid])
        .updateMask(mask)
        .copyProperties(image, ["system:time_start"])
    )


def _landsat_collection(aoi, year: int, months: list[int]) -> ee.ImageCollection:
    collections = [
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"),
    ]
    merged = collections[0]
    for collection in collections[1:]:
        merged = merged.merge(collection)
    return (
        merged.filterBounds(aoi)
        .filter(seasonal_date_filter(ee, year, months))
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .map(_mask_and_derive)
    )


def _summarize_year(aoi, grid_fc, year: int, months: list[int], scale: int) -> pd.DataFrame:
    collection = _landsat_collection(aoi, year, months)
    if collection.size().getInfo() == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    mean_image = collection.select(["ndvi", "ndbi", "bsi", "lst", "valid"]).mean()
    ndvi_std = collection.select("ndvi").reduce(ee.Reducer.stdDev()).rename("ndvi_std")
    lst_p90 = collection.select("lst").reduce(ee.Reducer.percentile([90])).rename("lst_p90_summer")

    summary = ee.Image.cat(
        [
            mean_image.select("ndvi").rename("ndvi_mean"),
            ndvi_std,
            mean_image.select("ndbi").rename("ndbi_mean"),
            mean_image.select("bsi").rename("bsi_mean"),
            mean_image.select("lst").rename("lst_mean_summer"),
            lst_p90,
            mean_image.select("valid").rename("valid_pixel_fraction"),
        ]
    )

    reduced = summary.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )
    frame = pd.DataFrame([feature["properties"] for feature in reduced.getInfo()["features"]])
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame["year"] = year
    return frame[OUTPUT_COLUMNS]


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Run Landsat summer feature extraction and write parquet output."""
    config = config or load_config()
    initialize_earth_engine(config)
    aoi = aoi_to_ee_geometry(config)
    grid = load_master_grid(config)
    grid_fc = grid_to_ee_feature_collection(ee, grid)
    paths = get_paths()

    yearly_frames = [
        _summarize_year(
            aoi=aoi,
            grid_fc=grid_fc,
            year=year,
            months=config.season_months,
            scale=config.earth_engine.landsat_scale,
        )
        for year in config.analysis_years
    ]
    combined = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return finalize_hex_year_output(
        combined,
        grid=grid,
        years=config.analysis_years,
        columns=OUTPUT_COLUMNS,
        output_path=paths.landsat_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Landsat yearly summer hex summaries.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
