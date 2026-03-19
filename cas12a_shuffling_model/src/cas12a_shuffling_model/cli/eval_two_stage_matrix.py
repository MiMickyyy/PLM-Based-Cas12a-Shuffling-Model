from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.active_prior import active_ranking_summary, load_active_codes
from cas12a_shuffling_model.composition.assistant_ranker import AssistantRankerScorer
from cas12a_shuffling_model.composition.gated_diagnostics import (
    distance_stratified_active_analysis,
    missing_active_diagnosis,
    novelty_vs_score_analysis,
)
from cas12a_shuffling_model.composition.table_io import read_table
from cas12a_shuffling_model.composition.two_stage_policy import TwoStagePolicyConfig, apply_two_stage_policy
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Combo:
    name: str
    recall_policy: str
    final_rerank_policy: str
    recall_teacher_weight: float = 0.35
    recall_scan_weight: float = 0.65
    recall_active_weight: float = 0.00
    recall_diversity_weight: float = 0.05


def _default_combos() -> list[_Combo]:
    return [
        _Combo("scan_only_active_only", "scan_only", "active_only"),
        _Combo("teacher_recall_active_only", "teacher_recall", "active_only"),
        _Combo("teacher_plausibility_filter_active_only", "teacher_plausibility_filter", "active_only"),
        _Combo("teacher_recall_plus_diversity_active_only", "teacher_recall_plus_diversity", "active_only"),
        _Combo("teacher_recall_active_heavy", "teacher_recall", "active_heavy"),
    ]


def _parse_combos(text: str | None) -> list[_Combo]:
    if not text or not str(text).strip():
        return _default_combos()
    out: list[_Combo] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) < 3:
            raise ValueError(f"Invalid combo spec: {item}; expected name:recall:rerank")
        name, recall, rerank = parts[:3]
        out.append(_Combo(name=name, recall_policy=recall, final_rerank_policy=rerank))
    return out


def _compute_suppression(
    *,
    shortlist_df: pd.DataFrame,
    rerank_df: pd.DataFrame,
    active_codes: list[str],
    top_k: int,
) -> dict[str, object]:
    actives = set(active_codes)
    if len(actives) == 0:
        return {}
    if "s_scan_score" in shortlist_df.columns:
        scan_top = shortlist_df.sort_values("s_scan_score", ascending=False).head(int(top_k))
        scan_active = set(scan_top["slot_code_11"].astype(str)) & actives
    else:
        scan_active = set()
    rerank_top = rerank_df.sort_values("final_score", ascending=False).head(int(top_k))
    rerank_active = set(rerank_top["slot_code_11"].astype(str)) & actives
    suppressed = sorted(scan_active - rerank_active)
    return {
        "n_s_scan_top_active": int(len(scan_active)),
        "n_rerank_top_active": int(len(rerank_active)),
        "n_suppressed": int(len(suppressed)),
        "suppressed_active_codes": suppressed,
    }


def _leave_one_out_hits(
    *,
    scored_df: pd.DataFrame,
    active_codes: list[str],
    cfg: TwoStagePolicyConfig,
    recall_pool_size: int,
) -> dict[str, float]:
    if len(active_codes) == 0:
        return {"loo_hits@50": float("nan"), "loo_hits@100": float("nan")}
    hit50 = 0
    hit100 = 0
    for held in active_codes:
        anchors = [c for c in active_codes if c != held]
        s = apply_two_stage_policy(scored_df, active_codes=anchors, cfg=cfg)
        rec = s.sort_values("recall_stage_score", ascending=False).reset_index(drop=True)
        pool = rec.head(min(int(recall_pool_size), len(rec))).copy()
        rr = pool.sort_values("final_score", ascending=False).reset_index(drop=True)
        rank_map = {str(c): int(i + 1) for i, c in enumerate(rr["slot_code_11"].astype(str).tolist())}
        r = rank_map.get(held)
        if r is not None and r <= 50:
            hit50 += 1
        if r is not None and r <= 100:
            hit100 += 1
    return {"loo_hits@50": float(hit50), "loo_hits@100": float(hit100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--shortlist-table", required=True)
    ap.add_argument("--assistant-checkpoint", required=True)
    ap.add_argument("--active-table", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--recall-pool-size", type=int, default=100000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--combos", default=None, help="name:recall:rerank,name:recall:rerank")
    ap.add_argument("--recall-teacher-weight", type=float, default=0.35)
    ap.add_argument("--recall-scan-weight", type=float, default=0.65)
    ap.add_argument("--recall-active-weight", type=float, default=0.0)
    ap.add_argument("--recall-diversity-weight", type=float, default=0.05)
    ap.add_argument("--disable-teacher-auto-flip", action="store_true")
    ap.add_argument("--disable-teacher-guardrail", action="store_true")
    ap.add_argument("--teacher-guardrail-min-frac", type=float, default=0.90)
    ap.add_argument("--disable-loo", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    c = cfg.get("slot_search", {}).get("rerank", {})
    combos = _parse_combos(args.combos)
    run_dir = args.out_dir or str(
        Path("cas12a_shuffling_model/outputs/two_stage_matrix") / f"run_{int(time.time())}"
    )
    out_path = Path(run_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shortlist = read_table(args.shortlist_table)
    active_codes = load_active_codes(args.active_table)
    scorer = AssistantRankerScorer(args.assistant_checkpoint, device=args.device)
    scored_base = scorer.score_dataframe(shortlist, batch_size=int(c.get("batch_size", 2048)), dual_head_alpha=c.get("dual_head_alpha"))

    rows = []
    combo_missing: dict[str, set[str]] = {}
    per_combo_outputs: dict[str, dict[str, str]] = {}
    for combo in combos:
        combo_dir = out_path / combo.name
        combo_dir.mkdir(parents=True, exist_ok=True)
        policy = TwoStagePolicyConfig(
            recall_policy=combo.recall_policy,
            final_rerank_policy=combo.final_rerank_policy,
            recall_teacher_weight=float(args.recall_teacher_weight),
            recall_scan_weight=float(args.recall_scan_weight),
            recall_active_weight=float(args.recall_active_weight),
            recall_diversity_weight=float(args.recall_diversity_weight),
            recall_pool_size=int(args.recall_pool_size),
            active_similarity_mode=str(c.get("active_prior_mode", "kernel_density_over_actives")),
            active_similarity_beta=float(c.get("active_prior_beta", 0.15)),
            active_similarity_gamma=float(c.get("active_prior_gamma", 0.70)),
            teacher_plausibility_quantile=float(c.get("teacher_plausibility_quantile", 0.05)),
            teacher_plausibility_penalty=float(c.get("teacher_plausibility_penalty", 0.50)),
            teacher_auto_flip=not bool(args.disable_teacher_auto_flip),
            teacher_guardrail=not bool(args.disable_teacher_guardrail),
            teacher_guardrail_min_frac=float(args.teacher_guardrail_min_frac),
        )
        scored = apply_two_stage_policy(scored_base, active_codes=active_codes, cfg=policy)
        recall_sorted = scored.sort_values("recall_stage_score", ascending=False).reset_index(drop=True)
        pool = recall_sorted.head(min(int(args.recall_pool_size), len(recall_sorted))).copy()
        rerank_sorted = pool.sort_values("final_score", ascending=False).reset_index(drop=True)
        rerank_sorted["final_rank"] = np.arange(1, len(rerank_sorted) + 1, dtype=np.int64)

        recall_summary = active_ranking_summary(
            df=recall_sorted,
            score_col="recall_stage_score",
            active_codes=active_codes,
            top_ks=(20000, 50000, 100000),
            distance_ks=(100,),
        )
        rerank_summary = active_ranking_summary(
            df=rerank_sorted,
            score_col="final_score",
            active_codes=active_codes,
            top_ks=(50, 100),
            distance_ks=(50, 100),
        )
        suppression = _compute_suppression(
            shortlist_df=shortlist,
            rerank_df=rerank_sorted,
            active_codes=active_codes,
            top_k=int(args.top_k),
        )
        missing_df = missing_active_diagnosis(
            ranked_df=rerank_sorted,
            active_codes=active_codes,
            recall_pool_size=int(args.recall_pool_size),
        )
        missing_pool_codes = set(
            missing_df.loc[missing_df["fail_stage"] == "recall_stage", "active_code"].astype(str).tolist()
        )
        combo_missing[combo.name] = missing_pool_codes
        novelty_df = novelty_vs_score_analysis(
            ranked_df=rerank_sorted,
            active_codes=active_codes,
            score_cols=["final_score"],
            top_k=1000,
        )
        dist_df = distance_stratified_active_analysis(
            ranked_df=rerank_sorted,
            active_codes=active_codes,
            score_col="final_score",
        )
        if args.disable_loo:
            loo = {"loo_hits@50": float("nan"), "loo_hits@100": float("nan")}
        else:
            loo = _leave_one_out_hits(
                scored_df=scored_base,
                active_codes=active_codes,
                cfg=policy,
                recall_pool_size=int(args.recall_pool_size),
            )

        recall_csv = combo_dir / "recall_ranked_all.csv"
        rerank_csv = combo_dir / "rerank_all.csv"
        top_csv = combo_dir / "rerank_top.csv"
        missing_csv = combo_dir / "missing_active_diagnosis.csv"
        novelty_csv = combo_dir / "novelty_stats.csv"
        dist_csv = combo_dir / "distance_stratified_active.csv"
        summary_json = combo_dir / "combo_summary.json"
        recall_sorted.to_csv(recall_csv, index=False)
        rerank_sorted.to_csv(rerank_csv, index=False)
        rerank_sorted.head(100).to_csv(top_csv, index=False)
        missing_df.to_csv(missing_csv, index=False)
        novelty_df.to_csv(novelty_csv, index=False)
        dist_df.to_csv(dist_csv, index=False)

        summary = {
            "combo": combo.name,
            "recall_policy": combo.recall_policy,
            "final_rerank_policy": combo.final_rerank_policy,
            "recall_summary": recall_summary,
            "rerank_summary": rerank_summary,
            "suppression": suppression,
            "missing_count_in_pool": int(len(missing_pool_codes)),
            "missing_active_codes_in_pool": sorted(missing_pool_codes),
            "teacher_recall_flipped": int(scored["teacher_recall_flipped"].iloc[0]) if "teacher_recall_flipped" in scored.columns and len(scored) else 0,
            "teacher_recall_guardrail_fallback": int(scored["teacher_recall_guardrail_fallback"].iloc[0]) if "teacher_recall_guardrail_fallback" in scored.columns and len(scored) else 0,
            **loo,
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        row = {
            "combo": combo.name,
            "recall_policy": combo.recall_policy,
            "final_rerank_policy": combo.final_rerank_policy,
            "present@20k": recall_summary.get("hits_at_20000"),
            "present@50k": recall_summary.get("hits_at_50000"),
            "present@100k": recall_summary.get("hits_at_100000"),
            "missing_count_in_pool": int(len(missing_pool_codes)),
            "hits@50": rerank_summary.get("hits_at_50"),
            "hits@100": rerank_summary.get("hits_at_100"),
            "best_rank_active": rerank_summary.get("best_rank_active"),
            "median_rank_active": rerank_summary.get("median_rank_active"),
            "n_suppressed": suppression.get("n_suppressed", np.nan),
            "teacher_recall_flipped": summary["teacher_recall_flipped"],
            "teacher_recall_guardrail_fallback": summary["teacher_recall_guardrail_fallback"],
            "loo_hits@50": loo["loo_hits@50"],
            "loo_hits@100": loo["loo_hits@100"],
            "novelty_median_dist_top1k": float(novelty_df["median_dist"].iloc[0]) if len(novelty_df) > 0 else np.nan,
        }
        rows.append(row)
        per_combo_outputs[combo.name] = {
            "recall_csv": str(recall_csv),
            "rerank_csv": str(rerank_csv),
            "top_csv": str(top_csv),
            "summary_json": str(summary_json),
        }
        logger.info(
            "combo=%s present@100k=%s hits@50=%s hits@100=%s missing_pool=%d",
            combo.name,
            str(row["present@100k"]),
            str(row["hits@50"]),
            str(row["hits@100"]),
            int(row["missing_count_in_pool"]),
        )

    baseline = "scan_only_active_only"
    base_missing = combo_missing.get(baseline, set())
    for row in rows:
        missing = combo_missing.get(str(row["combo"]), set())
        rescued = sorted(base_missing - missing)
        row["rescued_vs_scan_only_count"] = int(len(rescued))
        row["rescued_vs_scan_only_codes"] = ";".join(rescued)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        ["hits@50", "hits@100", "present@100k", "missing_count_in_pool", "best_rank_active"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    matrix_csv = out_path / "matrix_summary.csv"
    matrix_json = out_path / "matrix_summary.json"
    summary_df.to_csv(matrix_csv, index=False)
    matrix_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")

    report = {
        "shortlist_table": args.shortlist_table,
        "assistant_checkpoint": args.assistant_checkpoint,
        "active_table": args.active_table,
        "recall_pool_size": int(args.recall_pool_size),
        "combos": [c.__dict__ for c in combos],
        "matrix_summary_csv": str(matrix_csv),
        "matrix_summary_json": str(matrix_json),
        "per_combo_outputs": per_combo_outputs,
        "best_combo": summary_df.iloc[0]["combo"] if len(summary_df) > 0 else None,
    }
    report_json = out_path / "two_stage_matrix_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Two-stage matrix report saved: %s", report_json)


if __name__ == "__main__":
    main()
