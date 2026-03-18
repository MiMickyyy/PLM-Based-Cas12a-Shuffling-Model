from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.active_prior import load_active_codes
from cas12a_shuffling_model.composition.local_expand import LocalExpandConfig, expand_local_neighborhood
from cas12a_shuffling_model.composition.active_prior import parse_beta_list, parse_slot_weights
from cas12a_shuffling_model.composition.rerank import RerankConfig, rerank_shortlist
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--seed-table", required=True)
    ap.add_argument("--active-table", default=None)
    ap.add_argument("--out-table", default=None)
    ap.add_argument("--seed-score-col", default=None)
    ap.add_argument("--top-seed-count", type=int, default=None)
    ap.add_argument("--active-distance-cap", type=int, default=None)
    ap.add_argument("--no-active-seeds", action="store_true")
    ap.add_argument("--no-hamming-1", action="store_true")
    ap.add_argument("--hamming-2-top-seeds", type=int, default=None)
    ap.add_argument("--keep-original-codes", action="store_true")
    ap.add_argument("--assistant-checkpoint", default=None, help="Optional: rerank expanded candidates directly")
    ap.add_argument("--rerank-out-dir", default=None)
    ap.add_argument("--rerank-top-k", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    c = cfg.get("slot_search", {}).get("local_expand", {})

    out_table = args.out_table
    if not out_table:
        out_dir = c.get("output_dir", "cas12a_shuffling_model/outputs/local_expand")
        out_table = str(Path(out_dir) / f"local_expanded_{int(time.time())}.csv")

    active_table = str(args.active_table or c.get("active_table") or cfg.get("paths", {}).get("sequence_results", "")).strip()
    if not active_table:
        raise ValueError("Missing --active-table and config slot_search.local_expand.active_table")
    active_codes = load_active_codes(active_table)

    summary = expand_local_neighborhood(
        seed_table=args.seed_table,
        active_codes=active_codes,
        out_table=out_table,
        cfg=LocalExpandConfig(
            seed_score_col=str(args.seed_score_col or c.get("seed_score_col", "final_score")),
            top_seed_count=int(args.top_seed_count if args.top_seed_count is not None else c.get("top_seed_count", 500)),
            active_distance_cap=int(
                args.active_distance_cap if args.active_distance_cap is not None else c.get("active_distance_cap", 3)
            ),
            include_active_seeds=not bool(args.no_active_seeds or c.get("no_active_seeds", False)),
            include_hamming_1=not bool(args.no_hamming_1 or c.get("no_hamming_1", False)),
            include_hamming_2_top_seeds=int(
                args.hamming_2_top_seeds
                if args.hamming_2_top_seeds is not None
                else c.get("hamming_2_top_seeds", 100)
            ),
            drop_original_codes=not bool(args.keep_original_codes or c.get("keep_original_codes", False)),
        ),
    )
    logger.info("Local expansion done. summary=%s", summary)

    if args.assistant_checkpoint:
        rc = cfg.get("slot_search", {}).get("rerank", {})
        rerank_out = args.rerank_out_dir
        if not rerank_out:
            rerank_out = str(Path(Path(out_table).parent) / "local_expanded_rerank")
        outputs = rerank_shortlist(
            shortlist_table=out_table,
            assistant_checkpoint=args.assistant_checkpoint,
            out_dir=rerank_out,
            cfg=RerankConfig(
                batch_size=int(rc.get("batch_size", 2048)),
                top_k=int(args.rerank_top_k if args.rerank_top_k is not None else rc.get("top_k", 50)),
                include_sequence=False,
                dual_head_alpha=rc.get("dual_head_alpha"),
                active_prior_mode=str(rc.get("active_prior_mode", "kernel_density_over_actives")),
                active_prior_beta=float(rc.get("active_prior_beta", 0.15)),
                active_prior_gamma=float(rc.get("active_prior_gamma", 0.7)),
                active_prior_slot_weights=parse_slot_weights(rc.get("active_prior_slot_weights")),
                active_prior_beta_sweep=tuple(parse_beta_list(rc.get("active_prior_beta_sweep"))),
                active_prior_base_score_col=str(rc.get("active_prior_base_score_col", "assistant_score")),
                active_eval_top_ks=tuple(int(x) for x in rc.get("active_eval_top_ks", [50, 100])),
                teacher_audit=False,
            ),
            active_codes=active_codes,
            validated_domains=None,
            device=args.device,
            teacher_scorer=None,
        )
        logger.info("Local expanded rerank done. outputs=%s", outputs)


if __name__ == "__main__":
    main()
