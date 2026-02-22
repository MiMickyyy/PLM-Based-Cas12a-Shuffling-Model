import pandas as pd

from cas12a_shuffling_model.calibration.calibrator import (
    CalibrationConfig,
    apply_calibration,
    fit_calibrator,
)
from cas12a_shuffling_model.search.rank_pipeline import greedy_diversity_select, hamming_distance


def test_fit_and_apply_calibration():
    active = pd.DataFrame(
        {
            "global_score": [-5.0, -4.8, -4.9],
            "junction_mean": [-6.0, -5.7, -5.8],
            "junction_min": [-8.0, -7.6, -7.7],
        }
    )
    background = pd.DataFrame(
        {
            "global_score": [-12.0, -11.5, -10.8, -12.2],
            "junction_mean": [-13.0, -12.7, -12.1, -13.5],
            "junction_min": [-16.0, -15.2, -14.8, -16.5],
        }
    )
    art = fit_calibrator(active_df=active, background_df=background, cfg=CalibrationConfig())
    out = apply_calibration(active, art)
    assert "calibrated_prob" in out.columns
    assert "calibrated_score" in out.columns
    assert "passes_s_min" in out.columns


def test_hamming_distance():
    assert hamming_distance("AAAAAAAAAAA", "AAAAAAAALAA") == 1


def test_greedy_diversity_select():
    df = pd.DataFrame(
        [
            {"combo_compact": "AAAAAAAAAAA", "calibrated_prob": 0.99},
            {"combo_compact": "AAAAAAAAAAL", "calibrated_prob": 0.98},
            {"combo_compact": "LLLLLLLLLLL", "calibrated_prob": 0.97},
        ]
    )
    out = greedy_diversity_select(df, top_k=2, min_hamming=2)
    assert len(out) == 2
    assert "AAAAAAAAAAA" in set(out["combo_compact"].tolist())
    assert "LLLLLLLLLLL" in set(out["combo_compact"].tolist())

