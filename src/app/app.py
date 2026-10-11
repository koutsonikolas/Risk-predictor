"""Local Streamlit demo app for EarthRisk AI."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.utils.paths import get_paths


st.set_page_config(page_title="EarthRisk AI", layout="wide")
PATHS = get_paths()


@st.cache_data
def load_grid() -> gpd.GeoDataFrame:
    grid_files = sorted(PATHS.processed_grids_dir.glob("phoenix_h3_res*.parquet"))
    if not grid_files:
        raise FileNotFoundError("Master grid not found. Run `python -m src.cli build-grid` first.")
    return gpd.read_parquet(grid_files[0])


@st.cache_data
def load_parquet(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_parquet(path)


def value_to_color(value: float, minimum: float, maximum: float) -> list[int]:
    if maximum <= minimum:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
    red = int(40 + 215 * ratio)
    green = int(170 - 110 * ratio)
    blue = int(130 - 80 * ratio)
    return [red, green, blue, 180]


def polygon_rows(frame: gpd.GeoDataFrame, value_column: str) -> list[dict[str, object]]:
    minimum = float(frame[value_column].min())
    maximum = float(frame[value_column].max())
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "hex_id": row.hex_id,
                "polygon": [[lon, lat] for lon, lat in row.geometry.exterior.coords],
                "value": float(getattr(row, value_column)),
                "fill_color": value_to_color(float(getattr(row, value_column)), minimum, maximum),
            }
        )
    return rows


def render_polygon_map(frame: gpd.GeoDataFrame, value_column: str, tooltip_label: str) -> None:
    rows = polygon_rows(frame, value_column)
    view_state = pdk.ViewState(
        latitude=float(frame["centroid_lat"].mean()),
        longitude=float(frame["centroid_lon"].mean()),
        zoom=8.5,
        pitch=0,
    )
    layer = pdk.Layer(
        "PolygonLayer",
        data=rows,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255, 40],
        line_width_min_pixels=0.5,
        pickable=True,
        stroked=True,
        filled=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": f"Hex: {{hex_id}}\n{tooltip_label}: {{value}}"},
            map_style=None,
        ),
        use_container_width=True,
    )


def show_aoi_overview(grid: gpd.GeoDataFrame, risk_scores: pd.DataFrame) -> None:
    latest_year = int(risk_scores["year"].max())
    latest = risk_scores[risk_scores["year"] == latest_year]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hex Cells", f"{len(grid):,}")
    col2.metric("Latest Year", latest_year)
    col3.metric("Mean Risk", f"{latest['risk_score'].mean():.1f}")
    col4.metric("High-Risk Hexes", f"{(latest['risk_score'] >= 70).sum():,}")

    yearly = (
        risk_scores.groupby("year")[["risk_score", "heat_score", "vegetation_loss_score", "urban_expansion_score"]]
        .mean()
        .reset_index()
        .set_index("year")
    )
    st.line_chart(yearly, use_container_width=True)


def show_change_map(grid: gpd.GeoDataFrame, risk_scores: pd.DataFrame) -> None:
    years = sorted(risk_scores["year"].unique())
    if len(years) < 2:
        st.warning("At least two years of scored output are required for the change map.")
        return

    latest_year = years[-1]
    previous_year = years[-2]
    latest = risk_scores[risk_scores["year"] == latest_year][["hex_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_latest"}
    )
    previous = risk_scores[risk_scores["year"] == previous_year][["hex_id", "risk_score"]].rename(
        columns={"risk_score": "risk_score_previous"}
    )
    delta = latest.merge(previous, on="hex_id", how="left")
    delta["risk_change"] = delta["risk_score_latest"] - delta["risk_score_previous"]
    mapped = grid.merge(delta[["hex_id", "risk_change"]], on="hex_id", how="left").fillna({"risk_change": 0.0})

    st.subheader(f"Risk change from {previous_year} to {latest_year}")
    render_polygon_map(mapped, "risk_change", "Risk Change")
    st.dataframe(
        delta.sort_values("risk_change", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )


def show_current_risk_map(grid: gpd.GeoDataFrame, risk_scores: pd.DataFrame) -> None:
    latest_year = int(risk_scores["year"].max())
    latest = risk_scores[risk_scores["year"] == latest_year]
    mapped = grid.merge(latest[["hex_id", "risk_score"]], on="hex_id", how="left")
    st.subheader(f"Current risk map for {latest_year}")
    render_polygon_map(mapped, "risk_score", "Risk Score")
    st.dataframe(
        latest.sort_values("risk_score", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )


def show_forward_risk_alerts(grid: gpd.GeoDataFrame, alerts: pd.DataFrame, predictions: pd.DataFrame) -> None:
    mapped = grid.merge(predictions, on="hex_id", how="left").fillna({"prob_risk_increase_12m": 0.0})
    st.subheader("Forward-risk probabilities")
    render_polygon_map(mapped, "prob_risk_increase_12m", "12M Increase Probability")
    st.dataframe(alerts, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("EarthRisk AI")
    st.caption("Local Greater Phoenix climate and insurance-risk monitoring")

    try:
        grid = load_grid()
        risk_scores = load_parquet(PATHS.risk_scores)
        predictions = load_parquet(PATHS.predictions_current)
        alerts = load_parquet(PATHS.alerts_current)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    view = st.sidebar.radio(
        "View",
        ["AOI overview", "Change map", "Current risk map", "Forward-risk alerts"],
    )

    if view == "AOI overview":
        show_aoi_overview(grid, risk_scores)
    elif view == "Change map":
        show_change_map(grid, risk_scores)
    elif view == "Current risk map":
        show_current_risk_map(grid, risk_scores)
    else:
        show_forward_risk_alerts(grid, alerts, predictions)


if __name__ == "__main__":
    main()
