"""Filesystem paths used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for local project assets."""

    project_root: Path = PROJECT_ROOT
    configs_dir: Path = CONFIGS_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    interim_dir: Path = INTERIM_DIR
    processed_dir: Path = PROCESSED_DIR
    outputs_dir: Path = OUTPUTS_DIR
    raw_aoi_dir: Path = RAW_DIR / "aoi"
    raw_buildings_dir: Path = RAW_DIR / "buildings"
    processed_grids_dir: Path = PROCESSED_DIR / "grids"
    processed_features_dir: Path = PROCESSED_DIR / "features"
    processed_scoring_dir: Path = PROCESSED_DIR / "scoring"
    processed_modeling_dir: Path = PROCESSED_DIR / "modeling"
    processed_alerts_dir: Path = PROCESSED_DIR / "alerts"
    model_dir: Path = OUTPUTS_DIR / "models"

    def grid_path(self, resolution: int) -> Path:
        return self.processed_grids_dir / f"phoenix_h3_res{resolution}.parquet"

    @property
    def landsat_features(self) -> Path:
        return self.processed_features_dir / "landsat_hex_year.parquet"

    @property
    def nlcd_features(self) -> Path:
        return self.processed_features_dir / "nlcd_hex_year.parquet"

    @property
    def dynamic_world_features(self) -> Path:
        return self.processed_features_dir / "dynamic_world_hex_year.parquet"

    @property
    def era5_features(self) -> Path:
        return self.processed_features_dir / "era5_hex_year.parquet"

    @property
    def buildings_features(self) -> Path:
        return self.processed_features_dir / "buildings_hex_static.parquet"

    @property
    def feature_store(self) -> Path:
        return self.processed_features_dir / "feature_store_hex_year.parquet"

    @property
    def risk_scores(self) -> Path:
        return self.processed_scoring_dir / "risk_scores_hex_year.parquet"

    @property
    def training_table(self) -> Path:
        return self.processed_modeling_dir / "training_table.parquet"

    @property
    def predictions_current(self) -> Path:
        return self.processed_modeling_dir / "predictions_current.parquet"

    @property
    def alerts_current(self) -> Path:
        return self.processed_alerts_dir / "alerts_current.parquet"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "risk_increase_xgb.pkl"

    @property
    def feature_importance(self) -> Path:
        return self.model_dir / "feature_importance.csv"


def get_paths() -> ProjectPaths:
    """Return a shared path registry."""
    return ProjectPaths()


def ensure_directory(path: Path) -> Path:
    """Create a directory when it does not yet exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    """Create a file's parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
