"""ERA5-Land yearly hex summaries."""

from __future__ import annotations

import argparse

import ee
import numpy as np
import pandas as pd

from src.ingestion.common import finalize_hex_year_output, grid_to_ee_feature_collection, load_master_grid
from src.ingestion.ee_utils import aoi_to_ee_geometry, initialize_earth_engine, seasonal_date_filter, year_date_filter
from src.utils.config import ProjectConfig, load_config
from src.utils.paths import get_paths


HOT_DAY_THRESHOLD_C = 35.0
EXTREME_HOT_DAY_THRESHOLD_C = 40.0

OUTPUT_COLUMNS = [
    "hex_id",
    "year",
    "air_temp_mean_annual",
    "air_temp_mean_summer",
    "air_temp_anomaly_vs_baseline",
    "hot_days_count",
    "extreme_hot_days_count",
    "max_consecutive_hot_days",
    "temp_trend_3y",
]


def _kelvin_to_celsius(image: ee.Image, band: str) -> ee.Image:
    return image.select(band).subtract(273.15).rename(band)


def _run_length_image(binary_collection: ee.ImageCollection) -> ee.Image:
    initial = ee.List([ee.Image.constant(0).rename("current"), ee.Image.constant(0).rename("max")])

    def accumulate(image, state):
        state = ee.List(state)
        current = ee.Image(state.get(0))
        maximum = ee.Image(state.get(1))
        binary = ee.Image(image).rename("hot")
        current_run = current.add(1).multiply(binary).rename("current")
        max_run = maximum.max(current_run).rename("max")
        return ee.List([current_run, max_run])

    state = ee.List(binary_collection.sort("system:time_start").iterate(accumulate, initial))
    return ee.Image(state.get(1)).rename("max_consecutive_hot_days")


def _summarize_year(aoi, grid_fc, year: int, months: list[int], scale: int) -> pd.DataFrame:
    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(aoi)
        .filter(year_date_filter(ee, year))
    )
    if collection.size().getInfo() == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    summer = collection.filter(seasonal_date_filter(ee, year, months))
    annual_mean = collection.select("temperature_2m").mean().subtract(273.15).rename("air_temp_mean_annual")
    summer_mean = summer.select("temperature_2m").mean().subtract(273.15).rename("air_temp_mean_summer")
    hot_days = (
        collection.map(lambda image: _kelvin_to_celsius(image, "temperature_2m_max").gt(HOT_DAY_THRESHOLD_C))
        .sum()
        .rename("hot_days_count")
    )
    extreme_hot_days = (
        collection.map(
            lambda image: _kelvin_to_celsius(image, "temperature_2m_max").gt(EXTREME_HOT_DAY_THRESHOLD_C)
        )
        .sum()
        .rename("extreme_hot_days_count")
    )
    max_consecutive = _run_length_image(
        collection.map(lambda image: _kelvin_to_celsius(image, "temperature_2m_max").gt(HOT_DAY_THRESHOLD_C))
    )
    summary = ee.Image.cat([annual_mean, summer_mean, hot_days, extreme_hot_days, max_consecutive])
    reduced = summary.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=4,
    )
    frame = pd.DataFrame([feature["properties"] for feature in reduced.getInfo()["features"]])
    frame["year"] = year
    return frame[
        [
            "hex_id",
            "year",
            "air_temp_mean_annual",
            "air_temp_mean_summer",
            "hot_days_count",
            "extreme_hot_days_count",
            "max_consecutive_hot_days",
        ]
    ]


def _rolling_slope(values: pd.Series) -> pd.Series:
    slopes = []
    for idx in range(len(values)):
        if idx < 2:
            slopes.append(0.0)
            continue
        y = values.iloc[idx - 2 : idx + 1].to_numpy(dtype=float)
        x = np.array([0.0, 1.0, 2.0])
        slope = np.polyfit(x, y, 1)[0] if not np.isnan(y).any() else 0.0
        slopes.append(float(slope))
    return pd.Series(slopes, index=values.index)


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Run ERA5-Land yearly feature extraction and write parquet output."""
    config = config or load_config()
    initialize_earth_engine(config)
    aoi = aoi_to_ee_geometry(config)
    grid = load_master_grid(config)
    grid_fc = grid_to_ee_feature_collection(ee, grid)
    paths = get_paths()

    frames = [
        _summarize_year(aoi, grid_fc, year, config.season_months, config.earth_engine.era5_scale)
        for year in config.analysis_years
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        combined = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        combined = combined.sort_values(["hex_id", "year"]).reset_index(drop=True)
        baseline = combined[combined["year"].isin(config.baseline_years)].groupby("hex_id")[
            "air_temp_mean_annual"
        ].mean()
        combined["air_temp_anomaly_vs_baseline"] = (
            combined["air_temp_mean_annual"] - combined["hex_id"].map(baseline)
        )
        combined["temp_trend_3y"] = (
            combined.groupby("hex_id")["air_temp_mean_summer"].transform(_rolling_slope).fillna(0.0)
        )

    if "air_temp_anomaly_vs_baseline" not in combined.columns:
        combined["air_temp_anomaly_vs_baseline"] = pd.NA
    if "temp_trend_3y" not in combined.columns:
        combined["temp_trend_3y"] = pd.NA

    return finalize_hex_year_output(
        combined[OUTPUT_COLUMNS],
        grid=grid,
        years=config.analysis_years,
        columns=OUTPUT_COLUMNS,
        output_path=paths.era5_features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ERA5-Land yearly hex summaries.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
