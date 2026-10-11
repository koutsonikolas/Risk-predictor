"""Build the unified yearly feature store."""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config import ProjectConfig, load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.normalization import mean_of_available, normalize_series
from src.utils.paths import get_paths


def _index_to_pct(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").clip(-1.0, 1.0)
    return ((numeric + 1.0) / 2.0) * 100.0


def _diff_by_group(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return frame.groupby("hex_id")[column].diff(periods=periods)


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Merge yearly features and derive change metrics for scoring and modeling."""
    config = config or load_config()
    paths = get_paths()

    landsat = read_parquet(
        paths.landsat_features,
        "Landsat yearly features are missing. Run `python -m src.cli ingest-landsat` first.",
    )
    nlcd = read_parquet(
        paths.nlcd_features,
        "NLCD yearly features are missing. Run `python -m src.cli ingest-nlcd` first.",
    )
    dynamic_world = read_parquet(
        paths.dynamic_world_features,
        "Dynamic World yearly features are missing. Run `python -m src.cli ingest-dynamic-world` first.",
    )
    era5 = read_parquet(
        paths.era5_features,
        "ERA5 yearly features are missing. Run `python -m src.cli ingest-era5` first.",
    )
    buildings = read_parquet(
        paths.buildings_features,
        "Building-footprint features are missing. Run `python -m src.cli ingest-buildings` first.",
    )

    feature_store = landsat.merge(nlcd, on=["hex_id", "year"], how="outer")
    feature_store = feature_store.merge(dynamic_world, on=["hex_id", "year"], how="outer")
    feature_store = feature_store.merge(era5, on=["hex_id", "year"], how="outer")
    feature_store = feature_store.merge(buildings, on="hex_id", how="left")
    feature_store = feature_store.sort_values(["hex_id", "year"]).reset_index(drop=True)

    feature_store["ndvi_pct_proxy"] = _index_to_pct(feature_store["ndvi_mean"])
    feature_store["ndbi_pct_proxy"] = _index_to_pct(feature_store["ndbi_mean"])
    feature_store["bsi_pct_proxy"] = _index_to_pct(feature_store["bsi_mean"])
    feature_store["dw_vegetation_pct_proxy"] = pd.to_numeric(
        feature_store["dw_vegetation_share"],
        errors="coerce",
    ) * 100.0
    feature_store["dw_built_pct_proxy"] = pd.to_numeric(feature_store["dw_built_share"], errors="coerce") * 100.0
    feature_store["dw_bare_pct_proxy"] = pd.to_numeric(feature_store["dw_bare_share"], errors="coerce") * 100.0

    feature_store["vegetation_index"] = mean_of_available(
        feature_store,
        [
            "pct_tree_canopy",
            "pct_shrub_scrub",
            "pct_grassland",
            "dw_vegetation_pct_proxy",
            "ndvi_pct_proxy",
        ],
    )
    feature_store["built_index"] = mean_of_available(
        feature_store,
        [
            "pct_developed",
            "pct_impervious",
            "dw_built_pct_proxy",
            "ndbi_pct_proxy",
        ],
    )
    feature_store["barren_index"] = mean_of_available(
        feature_store,
        ["pct_barren", "dw_bare_pct_proxy", "bsi_pct_proxy"],
    )

    feature_store["veg_loss_1y"] = (-_diff_by_group(feature_store, "vegetation_index", 1)).clip(lower=0).fillna(0.0)
    feature_store["veg_loss_3y"] = (-_diff_by_group(feature_store, "vegetation_index", 3)).clip(lower=0).fillna(0.0)
    feature_store["built_gain_1y"] = _diff_by_group(feature_store, "built_index", 1).clip(lower=0).fillna(0.0)
    feature_store["built_gain_3y"] = _diff_by_group(feature_store, "built_index", 3).clip(lower=0).fillna(0.0)
    feature_store["surface_heat_change_1y"] = _diff_by_group(feature_store, "lst_mean_summer", 1).fillna(0.0)
    feature_store["surface_heat_change_3y"] = _diff_by_group(feature_store, "lst_mean_summer", 3).fillna(0.0)
    feature_store["air_heat_change_1y"] = _diff_by_group(feature_store, "air_temp_mean_summer", 1).fillna(0.0)
    feature_store["air_heat_change_3y"] = _diff_by_group(feature_store, "air_temp_mean_summer", 3).fillna(0.0)

    feature_store["land_stress_score"] = mean_of_available(
        pd.DataFrame(
            {
                "barren_index": normalize_series(feature_store["barren_index"]),
                "surface_heat": normalize_series(feature_store["lst_mean_summer"]),
                "air_anomaly": normalize_series(feature_store["air_temp_anomaly_vs_baseline"]),
                "hot_days": normalize_series(feature_store["hot_days_count"]),
            }
        ),
        ["barren_index", "surface_heat", "air_anomaly", "hot_days"],
    )
    feature_store["asset_exposure_score"] = mean_of_available(
        pd.DataFrame(
            {
                "building_count": normalize_series(feature_store["building_count"]),
                "building_density": normalize_series(feature_store["building_density"]),
                "building_area_sum_m2": normalize_series(feature_store["building_area_sum_m2"]),
                "mean_building_area_m2": normalize_series(feature_store["mean_building_area_m2"]),
            }
        ),
        ["building_count", "building_density", "building_area_sum_m2", "mean_building_area_m2"],
    )
    feature_store = feature_store.drop(
        columns=[
            "ndvi_pct_proxy",
            "ndbi_pct_proxy",
            "bsi_pct_proxy",
            "dw_vegetation_pct_proxy",
            "dw_built_pct_proxy",
            "dw_bare_pct_proxy",
        ]
    )
    write_parquet(feature_store, paths.feature_store)
    return feature_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified yearly feature store.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
