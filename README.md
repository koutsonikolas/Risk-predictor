# EarthRisk AI

EarthRisk AI is a local Python project for geospatial insurance-risk analysis in the Greater Phoenix area. It uses Google Earth Engine for remote-sensing access and preprocessing, H3 for spatial indexing, local parquet outputs for persistence, XGBoost for a forward-risk classifier, and Streamlit for the demo app.

## What The Project Does

- Builds a master H3 grid for Greater Phoenix.
- Aggregates Landsat, NLCD, Dynamic World, ERA5-Land, and local building-footprint features into that grid.
- Produces a deterministic climate and insurance risk score.
- Predicts which hex cells are likely to experience further risk growth over the next 12 months.
- Generates interpretable alerts and serves a local Streamlit app that reads only local parquet outputs.

## Repository Layout

```text
earthrisk-ai/
  configs/
  data/
    raw/
    interim/
    processed/
  notebooks/
  outputs/
  src/
    aoi/
    ingestion/
    preprocessing/
    features/
    scoring/
    modeling/
    alerts/
    app/
    utils/
  tests/
```

## Required External Inputs

### AOI boundary

Add the Greater Phoenix AOI file at:

`data/raw/aoi/greater_phoenix.geojson`

The loader will fail with a clear error until that file exists and contains a valid polygon or multipolygon.

### Building footprints

Add a local building-footprint file at one of:

- `data/raw/buildings/buildings.geojson`

The building aggregation step will fail with a clear error until one of those files exists.

### Google Earth Engine

Authenticate Earth Engine before running ingestion:

```bash
earthengine authenticate
python -c "import ee; ee.Initialize()"
```

If you use a service account instead, set `GOOGLE_APPLICATION_CREDENTIALS` and optionally `EE_PROJECT` in `.env`.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
```

## Pipeline Commands

These commands are exact and runnable once the required raw inputs and Earth Engine auth are in place.

```bash
python -m src.cli build-grid --config configs/phoenix.yaml
python -m src.cli ingest-landsat --config configs/phoenix.yaml
python -m src.cli ingest-nlcd --config configs/phoenix.yaml
python -m src.cli ingest-dynamic-world --config configs/phoenix.yaml
python -m src.cli ingest-era5 --config configs/phoenix.yaml
python -m src.cli ingest-buildings --config configs/phoenix.yaml
python -m src.cli build-feature-store --config configs/phoenix.yaml
python -m src.cli score-risk --config configs/phoenix.yaml
python -m src.cli build-training-data --config configs/phoenix.yaml
python -m src.cli train-model --config configs/phoenix.yaml
python -m src.cli predict-risk --config configs/phoenix.yaml
python -m src.cli generate-alerts --config configs/phoenix.yaml
streamlit run src/app/app.py
```

To run the core pipeline end to end after the raw inputs exist:

```bash
python -m src.cli run-all --config configs/phoenix.yaml
streamlit run src/app/app.py
```

## Output Files

- `data/processed/grids/phoenix_h3_res8.parquet`
- `data/processed/features/landsat_hex_year.parquet`
- `data/processed/features/nlcd_hex_year.parquet`
- `data/processed/features/dynamic_world_hex_year.parquet`
- `data/processed/features/era5_hex_year.parquet`
- `data/processed/features/buildings_hex_static.parquet`
- `data/processed/features/feature_store_hex_year.parquet`
- `data/processed/scoring/risk_scores_hex_year.parquet`
- `data/processed/modeling/training_table.parquet`
- `data/processed/modeling/predictions_current.parquet`
- `data/processed/alerts/alerts_current.parquet`
- `outputs/models/risk_increase_xgb.pkl`
- `outputs/models/feature_importance.csv`

## Notes On Data Availability

- Dynamic World coverage starts in 2015, so earlier years are emitted with null Dynamic World metrics.
- NLCD release years are mapped to the nearest available historical NLCD year on or before each requested analysis year.
- The Streamlit app reads only local parquet outputs and never calls Earth Engine directly.
