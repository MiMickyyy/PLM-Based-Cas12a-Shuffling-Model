from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.calibration.calibrator import (
    CalibrationConfig,
    FEATURES,
    apply_calibration,
    fit_calibrator,
    save_calibration_artifact,
)
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
    score_rows_with_teacher,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _has_teacher_features(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in FEATURES)


def _ensure_teacher_scored(
    *,
    df: pd.DataFrame,
    scorer,
    validated_domains,
    combo_col: str = "combo_compact",
    seq_col: str = "sequence_aa",
) -> pd.DataFrame:
    if _has_teacher_features(df):
        return df.copy()
    return score_rows_with_teacher(
        rows_df=df,
        scorer=scorer,
        validated_domains=validated_domains,
        combo_col=combo_col,
        seq_col=seq_col,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--active-csv", default=None)
    ap.add_argument("--background-csv", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--background-size", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)

    active_csv = args.active_csv or cfg.get("paths", {}).get("out_active_dir")
    if active_csv and Path(active_csv).is_dir():
        active_csv = str(Path(active_csv) / "active_chimeras_reconstructed.csv")
    if not active_csv:
        raise SystemExit("Missing active CSV path")

    background_csv = args.background_csv or cfg.get("distill", {}).get("output_csv")
    if not background_csv:
        raise SystemExit("Missing background CSV path")

    cal_cfg = cfg.get("calibration", {})
    out_dir = args.out_dir or cal_cfg.get("output_dir")
    if not out_dir:
        out_dir = "cas12a_shuffling_model/outputs/calibration"

    bg_size = int(args.background_size) if args.background_size else int(cal_cfg.get("background_size", 300))

    run_id = f"cal_{int(time.time())}"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    validated_domains = load_validated_domains_dict(vd_path)
    teacher_scorer = build_teacher_scorer_from_config(cfg, device=args.device)

    active_df_raw = pd.read_csv(active_csv)
    active_df = _ensure_teacher_scored(
        df=active_df_raw,
        scorer=teacher_scorer,
        validated_domains=validated_domains,
    )
    if "combo_compact" in active_df.columns:
        active_combos = set(active_df["combo_compact"].astype(str).tolist())
    else:
        active_combos = set()

    background_df_raw = pd.read_csv(background_csv)
    if "combo_compact" in background_df_raw.columns and active_combos:
        background_df_raw = background_df_raw[
            ~background_df_raw["combo_compact"].astype(str).isin(active_combos)
        ].copy()
    if bg_size > 0 and len(background_df_raw) > bg_size:
        background_df_raw = background_df_raw.sample(n=bg_size, random_state=int(cfg.get("seed", 13)))

    background_df = _ensure_teacher_scored(
        df=background_df_raw,
        scorer=teacher_scorer,
        validated_domains=validated_domains,
    )

    cal = CalibrationConfig(
        c=float(cal_cfg.get("C", 1.0)),
        class_weight_positive=float(cal_cfg.get("class_weight_positive", 1.0)),
        class_weight_background=float(cal_cfg.get("class_weight_background", 0.2)),
        s_min_quantile=float(cal_cfg.get("s_min_quantile", 0.1)),
    )
    artifact = fit_calibrator(active_df=active_df, background_df=background_df, cfg=cal)
    paths = save_calibration_artifact(artifact, out_dir=str(run_dir), run_id=run_id)

    act_out = apply_calibration(active_df, artifact)
    bg_out = apply_calibration(background_df, artifact)
    act_out.to_csv(run_dir / "active_scored_calibrated.csv", index=False)
    bg_out.to_csv(run_dir / "background_scored_calibrated.csv", index=False)

    report = {
        "run_id": run_id,
        "active_csv": str(active_csv),
        "background_csv": str(background_csv),
        "model_path": paths["model_path"],
        "meta_path": paths["meta_path"],
        "n_active": int(len(active_df)),
        "n_background": int(len(background_df)),
        "active_calibrated_prob_mean": float(act_out["calibrated_prob"].mean()),
        "background_calibrated_prob_mean": float(bg_out["calibrated_prob"].mean()),
    }
    (run_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Calibration done. run_dir=%s", run_dir)


if __name__ == "__main__":
    main()

