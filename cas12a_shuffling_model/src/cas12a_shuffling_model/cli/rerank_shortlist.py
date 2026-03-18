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
