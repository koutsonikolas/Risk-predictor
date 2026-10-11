"""Command-line entrypoints for EarthRisk AI."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys


COMMAND_TARGETS = {
    "build-grid": "src.aoi.grid:build_master_grid",
    "ingest-landsat": "src.ingestion.landsat:run",
    "ingest-nlcd": "src.ingestion.nlcd:run",
    "ingest-dynamic-world": "src.ingestion.dynamic_world:run",
    "ingest-era5": "src.ingestion.era5:run",
    "ingest-buildings": "src.preprocessing.buildings:run",
    "build-feature-store": "src.features.build_feature_store:run",
    "score-risk": "src.scoring.risk_score:run",
    "build-training-data": "src.modeling.build_training_table:run",
    "train-model": "src.modeling.train_model:run",
    "predict-risk": "src.modeling.predict_model:run",
    "generate-alerts": "src.alerts.generate_alerts:run",
}


def _resolve_action(target: str):
    module_name, function_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _dispatch(target: str, config_path: str | None) -> None:
    from src.utils.config import load_config

    config = load_config(config_path)
    _resolve_action(target)(config)


def _run_all(config_path: str | None) -> None:
    from src.utils.config import load_config

    config = load_config(config_path)
    for target in COMMAND_TARGETS.values():
        _resolve_action(target)(config)


def _launch_app(_config_path: str | None) -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app/app.py"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EarthRisk AI local pipeline CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="Path to the YAML config file.")

    for name in [*COMMAND_TARGETS.keys(), "run-all", "launch-app"]:
        subparsers.add_parser(name, parents=[common])

    args = parser.parse_args()
    if args.command == "run-all":
        _run_all(args.config)
    elif args.command == "launch-app":
        _launch_app(args.config)
    else:
        _dispatch(COMMAND_TARGETS[args.command], args.config)


if __name__ == "__main__":
    main()
