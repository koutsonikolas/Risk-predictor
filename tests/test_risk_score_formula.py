import pandas as pd


def test_risk_score_formula_weights_sum_correctly():
    row = pd.Series(
        {
            "heat_score": 80.0,
            "vegetation_loss_score": 60.0,
            "urban_expansion_score": 50.0,
            "land_stress_score": 40.0,
            "asset_exposure_score": 30.0,
        }
    )
    score = (
        0.35 * row["heat_score"]
        + 0.25 * row["vegetation_loss_score"]
        + 0.20 * row["urban_expansion_score"]
        + 0.10 * row["land_stress_score"]
        + 0.10 * row["asset_exposure_score"]
    )
    assert score == 60.0
