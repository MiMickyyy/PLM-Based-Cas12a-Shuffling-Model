from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cas12a_shuffling_model.calibration.thresholds import compute_s_min_threshold

logger = logging.getLogger(__name__)


FEATURES = ["global_score", "junction_mean", "junction_min"]


@dataclass(frozen=True)
class CalibrationConfig:
    c: float = 1.0
    class_weight_positive: float = 1.0
    class_weight_background: float = 1.0
    s_min_quantile: float = 0.1


def _extract_feature_matrix(df: pd.DataFrame, features: Sequence[str] = FEATURES) -> np.ndarray:
    arr = df[list(features)].astype(float).to_numpy()
    return arr


def fit_calibrator(
    *,
    active_df: pd.DataFrame,
    background_df: pd.DataFrame,
    cfg: CalibrationConfig,
) -> dict[str, Any]:
    if len(active_df) == 0:
        raise ValueError("active_df is empty")
    if len(background_df) == 0:
        raise ValueError("background_df is empty")

    pos = active_df.copy()
    neg = background_df.copy()
    pos["label"] = 1
    neg["label"] = 0
    train_df = pd.concat([pos, neg], axis=0, ignore_index=True)

    X = _extract_feature_matrix(train_df, FEATURES)
    y = train_df["label"].astype(int).to_numpy()

    sample_weight = np.where(
        y == 1, float(cfg.class_weight_positive), float(cfg.class_weight_background)
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(cfg.c),
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=13,
                ),
            ),
        ]
    )
    model.fit(X, y, logreg__sample_weight=sample_weight)

    s_min = compute_s_min_threshold(
        active_junction_min=active_df["junction_min"].astype(float).tolist(),
        quantile=float(cfg.s_min_quantile),
    )
    artifact = {
        "model": model,
        "features": list(FEATURES),
        "s_min_threshold": float(s_min),
        "config": {
            "C": float(cfg.c),
            "class_weight_positive": float(cfg.class_weight_positive),
            "class_weight_background": float(cfg.class_weight_background),
            "s_min_quantile": float(cfg.s_min_quantile),
        },
        "train_summary": {
            "n_active": int(len(active_df)),
            "n_background": int(len(background_df)),
            "active_feature_means": {
                k: float(active_df[k].astype(float).mean()) for k in FEATURES
            },
            "background_feature_means": {
                k: float(background_df[k].astype(float).mean()) for k in FEATURES
            },
        },
    }
    return artifact


def apply_calibration(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    features = artifact["features"]
    model = artifact["model"]
    X = _extract_feature_matrix(out, features)
    prob = model.predict_proba(X)[:, 1]
    out["calibrated_prob"] = prob
    out["calibrated_score"] = prob

    s_min = float(artifact.get("s_min_threshold", float("nan")))
    if np.isfinite(s_min):
        out["passes_s_min"] = out["junction_min"].astype(float) >= s_min
    else:
        out["passes_s_min"] = True
    return out


def save_calibration_artifact(artifact: dict[str, Any], out_dir: str, run_id: str) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "calibration_model.joblib"
    meta_path = out_path / "calibration_meta.json"

    joblib.dump(artifact["model"], model_path)
    meta = {k: v for k, v in artifact.items() if k != "model"}
    meta["run_id"] = run_id
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "meta_path": str(meta_path)}


def load_calibration_artifact(model_path: str, meta_path: str) -> dict[str, Any]:
    model = joblib.load(model_path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return {"model": model, **meta}

