from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.active_prior import (
    parse_beta_list,
    parse_slot_weights,
    load_active_codes,
)
from cas12a_shuffling_model.composition.rerank import RerankConfig, rerank_shortlist
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
    with_teacher_overrides,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--shortlist-table", required=True)
    ap.add_argument("--assistant-checkpoint", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dual-head-alpha", type=float, default=None)
    ap.add_argument("--active-table", default=None)
    ap.add_argument("--use-two-stage-policy", action="store_true")
    ap.add_argument("--no-two-stage-policy", action="store_true")
    ap.add_argument(
        "--recall-policy",
        choices=[
            "scan_only",
            "teacher_recall",
            "teacher_plausibility_filter",
            "teacher_recall_plus_diversity",
            "teacher_head",
            "teacher_active_mix",
            "scan_teacher_mix",
            "active_only",
        ],
        default=None,
    )
    ap.add_argument("--final-rerank-policy", choices=["active_only", "active_heavy"], default=None)
    ap.add_argument("--recall-diversity-weight", type=float, default=None)
    ap.add_argument("--policy-mode", choices=["global_fixed", "hard_gated", "soft_gated"], default=None)
    ap.add_argument(
        "--gate-signal",
        choices=["min_hamming_to_active", "kernel_similarity_to_actives", "active_density_score"],
        default=None,
    )
    ap.add_argument("--alpha-far", type=float, default=None)
    ap.add_argument("--alpha-near", type=float, default=None)
    ap.add_argument("--similarity-beta", type=float, default=None)
    ap.add_argument("--kernel-gamma", type=float, default=None)
    ap.add_argument("--density-radius", type=int, default=None)
    ap.add_argument("--density-gamma", type=float, default=None)
    ap.add_argument("--hard-distance-threshold", type=int, default=None)
    ap.add_argument("--hard-similarity-threshold", type=float, default=None)
    ap.add_argument("--soft-center", type=float, default=None)
    ap.add_argument("--soft-scale", type=float, default=None)
    ap.add_argument("--recall-teacher-weight", type=float, default=None)
    ap.add_argument("--recall-active-weight", type=float, default=None)
    ap.add_argument("--recall-scan-weight", type=float, default=None)
    ap.add_argument("--recall-pool-size", type=int, default=None)
    ap.add_argument("--teacher-usage-mode", choices=["none", "recall_only", "plausibility_filter"], default=None)
    ap.add_argument("--teacher-plausibility-quantile", type=float, default=None)
    ap.add_argument("--teacher-plausibility-penalty", type=float, default=None)
    ap.add_argument(
        "--active-prior-mode",
        default=None,
        choices=["none", "min_hamming_similarity", "weighted_slot_similarity", "kernel_density_over_actives"],
    )
    ap.add_argument("--active-prior-beta", type=float, default=None)
    ap.add_argument("--active-prior-gamma", type=float, default=None)
    ap.add_argument("--active-prior-slot-weights", default=None)
    ap.add_argument("--active-prior-beta-sweep", default=None, help="comma list, e.g. 0.0,0.05,0.1")
    ap.add_argument("--active-prior-base-score-col", default=None)
    ap.add_argument("--include-sequence", action="store_true")
    ap.add_argument("--teacher-audit", action="store_true")
    ap.add_argument("--teacher-audit-top-k", type=int, default=None)
    ap.add_argument("--teacher-audit-batch-size", type=int, default=None)
    ap.add_argument("--teacher-model-name-or-path", default=None)
    ap.add_argument("--teacher-model-source", choices=["hf", "local"], default=None)
    ap.add_argument("--teacher-adapter-path", default=None)
    ap.add_argument("--teacher-model-revision", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    cfg = with_teacher_overrides(
        cfg,
        model_name_or_path=args.teacher_model_name_or_path,
        model_source=args.teacher_model_source,
        adapter_path=args.teacher_adapter_path,
        model_revision=args.teacher_model_revision,
    )
    c = cfg.get("slot_search", {}).get("rerank", {})
    gp = cfg.get("slot_search", {}).get("gated_policy", {})

    run_dir = args.out_dir or c.get("output_dir")
    if not run_dir:
        run_dir = str(Path("cas12a_shuffling_model/outputs/rerank") / f"run_{int(time.time())}")
    else:
        run_dir = str(Path(run_dir) / f"run_{int(time.time())}")

    domains = None
    include_sequence = bool(args.include_sequence or c.get("include_sequence", False))
    teacher_audit = bool(args.teacher_audit or c.get("teacher_audit", False))
    if include_sequence or teacher_audit:
        vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
        domains = load_validated_domains_dict(vd_path)
    teacher_scorer = build_teacher_scorer_from_config(cfg, device=args.device) if teacher_audit else None
    active_table = str(args.active_table or c.get("active_table") or cfg.get("paths", {}).get("sequence_results", "")).strip()
    if active_table:
        try:
            active_codes = load_active_codes(active_table)
        except FileNotFoundError:
            logger.warning("Active table not found: %s; rerank will run without active prior", active_table)
            active_codes = []
    else:
        active_codes = []

    outputs = rerank_shortlist(
        shortlist_table=args.shortlist_table,
        assistant_checkpoint=args.assistant_checkpoint,
        out_dir=run_dir,
        cfg=RerankConfig(
            batch_size=int(args.batch_size if args.batch_size is not None else c.get("batch_size", 2048)),
            top_k=int(args.top_k if args.top_k is not None else c.get("top_k", 50)),
            include_sequence=include_sequence,
            dual_head_alpha=(float(args.dual_head_alpha) if args.dual_head_alpha is not None else c.get("dual_head_alpha")),
            use_two_stage_policy=(
                False
                if args.no_two_stage_policy
                else (
                    True
                    if args.use_two_stage_policy
                    else bool(c.get("use_two_stage_policy", True))
                )
            ),
            recall_policy=str(
                args.recall_policy
                if args.recall_policy is not None
                else c.get("recall_policy", gp.get("recall_policy", "teacher_recall"))
            ),
            final_rerank_policy=str(
                args.final_rerank_policy
                if args.final_rerank_policy is not None
                else c.get("final_rerank_policy", "active_only")
            ),
            recall_diversity_weight=float(
                args.recall_diversity_weight
                if args.recall_diversity_weight is not None
                else c.get("recall_diversity_weight", gp.get("recall_diversity_weight", 0.05))
            ),
            policy_mode=str(args.policy_mode if args.policy_mode is not None else gp.get("policy_mode", c.get("policy_mode", "global_fixed"))),
            gate_signal=str(
                args.gate_signal
                if args.gate_signal is not None
                else gp.get("gate_signal", c.get("gate_signal", "kernel_similarity_to_actives"))
            ),
            alpha_far=float(
                args.alpha_far if args.alpha_far is not None else gp.get("alpha_far", c.get("alpha_far", 0.60))
            ),
            alpha_near=float(
                args.alpha_near if args.alpha_near is not None else gp.get("alpha_near", c.get("alpha_near", 0.15))
            ),
            similarity_beta=float(
                args.similarity_beta
                if args.similarity_beta is not None
                else gp.get("similarity_beta", c.get("active_prior_beta", 0.15))
            ),
            kernel_gamma=float(
                args.kernel_gamma if args.kernel_gamma is not None else gp.get("kernel_gamma", c.get("active_prior_gamma", 0.7))
            ),
            density_radius=int(
                args.density_radius if args.density_radius is not None else gp.get("density_radius", c.get("density_radius", 3))
            ),
            density_gamma=float(
                args.density_gamma if args.density_gamma is not None else gp.get("density_gamma", c.get("density_gamma", 1.0))
            ),
            hard_distance_threshold=int(
                args.hard_distance_threshold
                if args.hard_distance_threshold is not None
                else gp.get("hard_distance_threshold", c.get("hard_distance_threshold", 3))
            ),
            hard_similarity_threshold=float(
                args.hard_similarity_threshold
                if args.hard_similarity_threshold is not None
                else gp.get("hard_similarity_threshold", c.get("hard_similarity_threshold", 0.55))
            ),
            soft_center=(
                float(args.soft_center)
                if args.soft_center is not None
                else (
                    float(gp.get("soft_center"))
                    if gp.get("soft_center", None) is not None
                    else (float(c.get("soft_center")) if c.get("soft_center", None) is not None else None)
                )
            ),
            soft_scale=float(
                args.soft_scale if args.soft_scale is not None else gp.get("soft_scale", c.get("soft_scale", 8.0))
            ),
            recall_teacher_weight=float(
                args.recall_teacher_weight
                if args.recall_teacher_weight is not None
                else gp.get("recall_teacher_weight", c.get("recall_teacher_weight", 0.70))
            ),
            recall_active_weight=float(
                args.recall_active_weight
                if args.recall_active_weight is not None
                else gp.get("recall_active_weight", c.get("recall_active_weight", 0.10))
            ),
            recall_scan_weight=float(
                args.recall_scan_weight
                if args.recall_scan_weight is not None
                else gp.get("recall_scan_weight", c.get("recall_scan_weight", 0.20))
            ),
            recall_pool_size=(
                int(args.recall_pool_size)
                if args.recall_pool_size is not None
                else (
                    int(gp.get("recall_pool_size"))
                    if gp.get("recall_pool_size", None) is not None
                    else (
                        int(c.get("recall_pool_size"))
                        if c.get("recall_pool_size", None) is not None
                        else None
                    )
                )
            ),
            teacher_usage_mode=str(
                args.teacher_usage_mode
                if args.teacher_usage_mode is not None
                else gp.get("teacher_usage_mode", c.get("teacher_usage_mode", "none"))
            ),
            teacher_plausibility_quantile=float(
                args.teacher_plausibility_quantile
                if args.teacher_plausibility_quantile is not None
                else gp.get("teacher_plausibility_quantile", c.get("teacher_plausibility_quantile", 0.05))
            ),
            teacher_plausibility_penalty=float(
                args.teacher_plausibility_penalty
                if args.teacher_plausibility_penalty is not None
                else gp.get("teacher_plausibility_penalty", c.get("teacher_plausibility_penalty", 0.50))
            ),
            active_prior_mode=str(
                args.active_prior_mode if args.active_prior_mode is not None else c.get("active_prior_mode", "none")
            ),
            active_prior_beta=float(
                args.active_prior_beta if args.active_prior_beta is not None else c.get("active_prior_beta", 0.0)
            ),
            active_prior_gamma=float(
                args.active_prior_gamma if args.active_prior_gamma is not None else c.get("active_prior_gamma", 0.7)
            ),
            active_prior_slot_weights=parse_slot_weights(
                args.active_prior_slot_weights
                if args.active_prior_slot_weights is not None
                else c.get("active_prior_slot_weights")
            ),
            active_prior_beta_sweep=tuple(
                parse_beta_list(
                    args.active_prior_beta_sweep
                    if args.active_prior_beta_sweep is not None
                    else c.get("active_prior_beta_sweep")
                )
            ),
            active_prior_base_score_col=str(
                args.active_prior_base_score_col
                if args.active_prior_base_score_col is not None
                else c.get("active_prior_base_score_col", "assistant_score")
            ),
            active_eval_top_ks=tuple(int(x) for x in c.get("active_eval_top_ks", [50, 100])),
            recall_eval_top_ks=tuple(int(x) for x in c.get("recall_eval_top_ks", [20000, 50000, 100000])),
            teacher_audit=teacher_audit,
            teacher_audit_top_k=int(
                args.teacher_audit_top_k if args.teacher_audit_top_k is not None else c.get("teacher_audit_top_k", 200)
            ),
            teacher_audit_batch_size=int(
                args.teacher_audit_batch_size
                if args.teacher_audit_batch_size is not None
                else c.get("teacher_audit_batch_size", 8)
            ),
        ),
        active_codes=active_codes,
        validated_domains=domains,
        device=args.device,
        teacher_scorer=teacher_scorer,
    )
    logger.info("Rerank done. outputs=%s", outputs)


if __name__ == "__main__":
    main()
