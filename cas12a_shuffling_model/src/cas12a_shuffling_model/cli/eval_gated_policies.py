from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.active_prior import load_active_codes, parse_slot_weights
from cas12a_shuffling_model.composition.gated_diagnostics import (
    distance_stratified_active_analysis,
    evaluate_policy_variants,
    leave_one_active_out_eval,
    missing_active_diagnosis,
    novelty_vs_score_analysis,
)
from cas12a_shuffling_model.composition.gated_policy import GatedPolicyConfig
from cas12a_shuffling_model.composition.table_io import read_table
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _base_policy_from_config(cfg: dict) -> GatedPolicyConfig:
    slot = cfg.get("slot_search", {})
    gp = slot.get("gated_policy", {})
    rr = slot.get("rerank", {})
    return GatedPolicyConfig(
        policy_mode=str(gp.get("policy_mode", rr.get("policy_mode", "global_fixed"))),
        gate_signal=str(gp.get("gate_signal", rr.get("gate_signal", "kernel_similarity_to_actives"))),
        alpha_far=float(gp.get("alpha_far", rr.get("alpha_far", 0.60))),
        alpha_near=float(gp.get("alpha_near", rr.get("alpha_near", 0.15))),
        similarity_beta=float(gp.get("similarity_beta", rr.get("similarity_beta", rr.get("active_prior_beta", 0.15)))),
        kernel_gamma=float(gp.get("kernel_gamma", rr.get("kernel_gamma", rr.get("active_prior_gamma", 0.7)))),
        density_radius=int(gp.get("density_radius", rr.get("density_radius", 3))),
        density_gamma=float(gp.get("density_gamma", rr.get("density_gamma", 1.0))),
        hard_distance_threshold=int(gp.get("hard_distance_threshold", rr.get("hard_distance_threshold", 3))),
        hard_similarity_threshold=float(gp.get("hard_similarity_threshold", rr.get("hard_similarity_threshold", 0.55))),
        soft_center=(float(gp["soft_center"]) if gp.get("soft_center", None) is not None else (float(rr["soft_center"]) if rr.get("soft_center", None) is not None else None)),
        soft_scale=float(gp.get("soft_scale", rr.get("soft_scale", 8.0))),
        recall_policy=str(gp.get("recall_policy", rr.get("recall_policy", "scan_teacher_mix"))),
        recall_teacher_weight=float(gp.get("recall_teacher_weight", rr.get("recall_teacher_weight", 0.70))),
        recall_active_weight=float(gp.get("recall_active_weight", rr.get("recall_active_weight", 0.10))),
        recall_scan_weight=float(gp.get("recall_scan_weight", rr.get("recall_scan_weight", 0.20))),
        recall_pool_size=(
            int(gp["recall_pool_size"])
            if gp.get("recall_pool_size", None) is not None
            else (int(rr["recall_pool_size"]) if rr.get("recall_pool_size", None) is not None else 100000)
        ),
        teacher_usage_mode=str(gp.get("teacher_usage_mode", rr.get("teacher_usage_mode", "none"))),
        teacher_plausibility_quantile=float(
            gp.get("teacher_plausibility_quantile", rr.get("teacher_plausibility_quantile", 0.05))
        ),
        teacher_plausibility_penalty=float(
            gp.get("teacher_plausibility_penalty", rr.get("teacher_plausibility_penalty", 0.50))
        ),
        slot_weights=parse_slot_weights(gp.get("slot_weights", rr.get("active_prior_slot_weights"))),
    )


def _policy_variants(base: GatedPolicyConfig) -> dict[str, GatedPolicyConfig]:
    variants: dict[str, GatedPolicyConfig] = {}
    variants["global_fixed"] = GatedPolicyConfig(
        **{**base.__dict__, "policy_mode": "global_fixed", "teacher_usage_mode": "none"}
    )
    variants["hard_gated"] = GatedPolicyConfig(
        **{**base.__dict__, "policy_mode": "hard_gated", "teacher_usage_mode": "recall_only"}
    )
    variants["soft_gated"] = GatedPolicyConfig(
        **{**base.__dict__, "policy_mode": "soft_gated", "teacher_usage_mode": "recall_only"}
    )
    variants["teacher_recall_only"] = GatedPolicyConfig(
        **{**base.__dict__, "policy_mode": "soft_gated", "teacher_usage_mode": "recall_only"}
    )
    variants["teacher_plausibility_filter"] = GatedPolicyConfig(
        **{**base.__dict__, "policy_mode": "soft_gated", "teacher_usage_mode": "plausibility_filter"}
    )
    return variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--input-table", required=True)
    ap.add_argument("--active-table", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--write-variant-tables", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    base_policy = _base_policy_from_config(cfg)
    variants = _policy_variants(base_policy)

    run_dir = args.out_dir
    if not run_dir:
        run_dir = str(Path("cas12a_shuffling_model/outputs/gated_eval") / f"run_{int(time.time())}")
    out_path = Path(run_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    table_df = read_table(args.input_table)
    active_codes = load_active_codes(args.active_table)
    rerank_top_ks = tuple(int(x) for x in cfg.get("slot_search", {}).get("rerank", {}).get("active_eval_top_ks", [50, 100]))
    recall_top_ks = tuple(int(x) for x in cfg.get("slot_search", {}).get("rerank", {}).get("recall_eval_top_ks", [20000, 50000, 100000]))

    comparison_df, details = evaluate_policy_variants(
        table_df=table_df,
        active_codes=active_codes,
        variants=variants,
        rerank_top_ks=rerank_top_ks,
        recall_top_ks=recall_top_ks,
    )
    comparison_csv = out_path / "policy_comparison.csv"
    comparison_json = out_path / "policy_comparison.json"
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(
        json.dumps(comparison_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    best_policy = "soft_gated"
    if len(comparison_df) > 0:
        tmp = comparison_df.copy()
        for c in ("rerank_hits_at_50", "rerank_hits_at_100", "rerank_best_rank_active"):
            if c not in tmp.columns:
                tmp[c] = 0
        tmp["rerank_best_rank_active"] = pd.to_numeric(tmp["rerank_best_rank_active"], errors="coerce").fillna(1e9)
        tmp = tmp.sort_values(
            ["rerank_hits_at_50", "rerank_hits_at_100", "rerank_best_rank_active"],
            ascending=[False, False, True],
        )
        best_policy = str(tmp.iloc[0]["policy_name"])
    best_result = details[best_policy]

    if bool(args.write_variant_tables):
        for name, result in details.items():
            table_path = out_path / f"policy_{name}_rerank_all.csv"
            result.rerank_pool.to_csv(table_path, index=False)

    loo_df = leave_one_active_out_eval(
        table_df=table_df,
        active_codes=active_codes,
        cfg=variants[best_policy],
        rerank_top_ks=rerank_top_ks,
    )
    loo_csv = out_path / "leave_one_active_out.csv"
    loo_df.to_csv(loo_csv, index=False)

    dist_df = distance_stratified_active_analysis(
        ranked_df=best_result.rerank_pool,
        active_codes=active_codes,
        score_col="final_gated_score",
    )
    dist_csv = out_path / "distance_stratified_active.csv"
    dist_df.to_csv(dist_csv, index=False)

    novelty_df = novelty_vs_score_analysis(
        ranked_df=best_result.rerank_pool,
        active_codes=active_codes,
        score_cols=["final_gated_score", "rerank_stage_score", "score_teacher", "score_active"],
        top_k=1000,
    )
    novelty_csv = out_path / "novelty_vs_score.csv"
    novelty_df.to_csv(novelty_csv, index=False)

    missing_df = missing_active_diagnosis(
        ranked_df=best_result.rerank_pool,
        active_codes=active_codes,
        recall_pool_size=int(variants[best_policy].recall_pool_size or len(best_result.recall_sorted)),
    )
    missing_csv = out_path / "missing_active_diagnosis.csv"
    missing_df.to_csv(missing_csv, index=False)

    report = {
        "input_table": args.input_table,
        "active_table": args.active_table,
        "best_policy": best_policy,
        "policy_comparison_csv": str(comparison_csv),
        "leave_one_active_out_csv": str(loo_csv),
        "distance_stratified_csv": str(dist_csv),
        "novelty_vs_score_csv": str(novelty_csv),
        "missing_active_diagnosis_csv": str(missing_csv),
        "n_active_total": int(len(active_codes)),
        "n_missing_in_best_policy": int(np.sum(missing_df["fail_stage"] == "recall_stage")) if len(missing_df) > 0 else 0,
    }
    report_json = out_path / "gated_eval_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Gated policy evaluation report saved: %s", report_json)


if __name__ == "__main__":
    main()
