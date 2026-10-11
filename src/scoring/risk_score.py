"""Deterministic risk scoring for the EarthRisk AI hex grid."""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config import ProjectConfig, load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.normalization import mean_of_available, normalize_series
from src.utils.paths import get_paths


OUTPUT_COLUMNS = [
    "hex_id",
    "year",
    "heat_score",
    "vegetation_loss_score",
    "urban_expansion_score",
    "land_stress_score",
    "asset_exposure_score",
    "risk_score",
]


def _nan_series(index: pd.Index) -> pd.Series:
    """Return a float series of NaNs aligned to the input index."""
    return pd.Series(float("nan"), index=index, dtype="float64")


def _robust_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize after clipping extreme tails.

    This keeps a few outliers from dominating the 0-100 scaling.
    """
    if series.isna().all():
        return _nan_series(series.index)

    non_null = series.dropna()
    lower = non_null.quantile(0.05)
    upper = non_null.quantile(0.95)

    if pd.isna(lower) or pd.isna(upper):
        return normalize_series(series)

    clipped = series.clip(lower=lower, upper=upper)
    return normalize_series(clipped)


def _normalized_feature(features: pd.DataFrame, column: str, *, inverse: bool = False) -> pd.Series:
    """Safely normalize a feature column, returning NaNs if the column is missing."""
    if column not in features.columns:
        result = _nan_series(features.index)
    else:
        result = _robust_normalize(features[column])

    return 100.0 - result if inverse else result


def _bounded_feature(
    features: pd.DataFrame,
    column: str,
    *,
    lower: float = 0.0,
    upper: float = 100.0,
) -> pd.Series:
    """Clip an already-scored feature to a safe 0-100 range."""
    if column not in features.columns:
        return pd.Series(0.0, index=features.index, dtype="float64")
    return features[column].clip(lower=lower, upper=upper).fillna(0.0)


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Compute deterministic component scores and final risk score."""
    config = config or load_config()
    paths = get_paths()
    features = read_parquet(
        paths.feature_store,
        "Feature store is missing. Run `python -m src.cli build-feature-store` first.",
    ).copy()

    heat_components = pd.DataFrame(
        {
            "lst_mean_summer": _normalized_feature(features, "lst_mean_summer"),
            "air_temp_anomaly_vs_baseline": _normalized_feature(features, "air_temp_anomaly_vs_baseline"),
            "hot_days_count": _normalized_feature(features, "hot_days_count"),
            "surface_heat_change_1y": _normalized_feature(features, "surface_heat_change_1y"),
            "air_heat_change_1y": _normalized_feature(features, "air_heat_change_1y"),
        },
        index=features.index,
    )
    features["heat_score"] = mean_of_available(heat_components, list(heat_components.columns)).fillna(0.0)

    vegetation_components = pd.DataFrame(
        {
            "veg_loss_1y": _normalized_feature(features, "veg_loss_1y"),
            "veg_loss_3y": _normalized_feature(features, "veg_loss_3y"),
            "tree_canopy_inverse": _normalized_feature(features, "pct_tree_canopy", inverse=True),
            "dw_vegetation_inverse": _normalized_feature(features, "dw_vegetation_share", inverse=True),
        },
        index=features.index,
    )
    features["vegetation_loss_score"] = mean_of_available(
        vegetation_components,
        list(vegetation_components.columns),
    ).fillna(0.0)

    urban_components = pd.DataFrame(
        {
            "built_gain_1y": _normalized_feature(features, "built_gain_1y"),
            "built_gain_3y": _normalized_feature(features, "built_gain_3y"),
            "pct_impervious": _normalized_feature(features, "pct_impervious"),
            "dw_transition_veg_to_built": _normalized_feature(features, "dw_transition_veg_to_built"),
        },
        index=features.index,
    )
    features["urban_expansion_score"] = mean_of_available(
        urban_components,
        list(urban_components.columns),
    ).fillna(0.0)

    land_stress_components = pd.DataFrame(
        {
            "base_land_stress": _bounded_feature(features, "land_stress_score"),
            "veg_to_bare_transition": _normalized_feature(features, "dw_transition_veg_to_bare"),
        },
        index=features.index,
    )
    features["land_stress_score"] = mean_of_available(
        land_stress_components,
        list(land_stress_components.columns),
    ).fillna(0.0)

    features["asset_exposure_score"] = _bounded_feature(features, "asset_exposure_score")

    features["risk_score"] = (
        0.35 * features["heat_score"]
        + 0.25 * features["vegetation_loss_score"]
        + 0.20 * features["urban_expansion_score"]
        + 0.10 * features["land_stress_score"]
        + 0.10 * features["asset_exposure_score"]
    )

    scores = features[OUTPUT_COLUMNS].copy()
    write_parquet(scores, paths.risk_scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deterministic yearly risk scores.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()