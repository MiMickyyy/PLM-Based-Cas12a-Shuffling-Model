from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.teacher_export import (
    TeacherExportConfig,
    export_teacher_labels,
)
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
    ap.add_argument("--input-table", default=None)
    ap.add_argument("--out-table", default=None)
    ap.add_argument("--active-table", default=None, help="Path to active combo table (e.g., Sequence_Result.xlsx)")
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--score-batch-size", type=int, default=None)
    ap.add_argument("--normalize-length-bin-size", type=int, default=None)
    ap.add_argument("--normalize-min-group-size", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--teacher-model-name-or-path", default=None)
    ap.add_argument("--teacher-model-source", choices=["hf", "local"], default=None)
    ap.add_argument("--teacher-adapter-path", default=None)
    ap.add_argument("--teacher-model-revision", default=None)
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
    exp_cfg = cfg.get("slot_search", {}).get("teacher_export", {})

    out_table = args.out_table or exp_cfg.get("output_table")
    if not out_table:
        base = exp_cfg.get("output_dir", "cas12a_shuffling_model/outputs/teacher_export")
        out_table = str(Path(base) / f"teacher_labels_{int(time.time())}.parquet")

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    domains = load_validated_domains_dict(vd_path)
    scorer = build_teacher_scorer_from_config(cfg, device=args.device)
    run_cfg = TeacherExportConfig(
        n_samples=int(args.n_samples if args.n_samples is not None else exp_cfg.get("n_samples", 10000)),
        seed=int(args.seed if args.seed is not None else cfg.get("seed", 13)),
        score_batch_size=int(
            args.score_batch_size if args.score_batch_size is not None else exp_cfg.get("score_batch_size", 8)
        ),
        normalize_length_bin_size=int(
            args.normalize_length_bin_size
            if args.normalize_length_bin_size is not None
            else exp_cfg.get("normalize_length_bin_size", 50)
        ),
        normalize_min_group_size=int(
            args.normalize_min_group_size
            if args.normalize_min_group_size is not None
            else exp_cfg.get("normalize_min_group_size", 64)
        ),
        resume=not bool(args.no_resume),
        include_junction_features=bool(exp_cfg.get("include_junction_features", True)),
        include_active_rows=bool(exp_cfg.get("include_active_rows", True)),
    )
    saved = export_teacher_labels(
        scorer=scorer,
        out_table=out_table,
        validated_domains=domains,
        cfg=run_cfg,
        input_table=args.input_table,
        active_table=str(args.active_table or cfg.get("paths", {}).get("sequence_results", "")) or None,
        slot_code_col=str(exp_cfg.get("slot_code_col", "slot_code_11")),
        sequence_col=str(exp_cfg.get("sequence_col", "sequence_aa")),
    )
    logger.info("Teacher labels exported: %s", saved)


if __name__ == "__main__":
    main()
