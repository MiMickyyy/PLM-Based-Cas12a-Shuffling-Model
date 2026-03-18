from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.active_prior import load_active_codes
from cas12a_shuffling_model.composition.hard_negative_mining import (
    HardNegativeMiningConfig,
    build_active_local_training_table,
)
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--base-table", required=True, help="Teacher-label table used for assistant training")
    ap.add_argument("--rerank-table", required=True, help="Rerank all table used for mining")
    ap.add_argument("--active-table", default=None, help="Sequence_Result(.xlsx/.csv) with known actives")
    ap.add_argument("--out-table", default=None)
    ap.add_argument("--out-hard-negatives", default=None)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--top-pool-size", type=int, default=None)
    ap.add_argument("--max-negatives", type=int, default=None)
    ap.add_argument("--min-distance", type=int, default=None)
    ap.add_argument("--max-distance", type=int, default=None)
    ap.add_argument("--active-score-quantile", type=float, default=None)
    ap.add_argument("--no-include-missing", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    m = cfg.get("slot_search", {}).get("hard_negative_mining", {})

    out_table = args.out_table
    if not out_table:
        out_base = m.get("output_dir", "cas12a_shuffling_model/outputs/hard_negative_mining")
        out_table = str(Path(out_base) / f"assistant_train_active_local_{int(time.time())}.csv")

    active_table = str(args.active_table or m.get("active_table") or cfg.get("paths", {}).get("sequence_results", "")).strip()
    if not active_table:
        raise ValueError("Missing --active-table and config slot_search.hard_negative_mining.active_table")
    active_codes = load_active_codes(active_table)

    summary = build_active_local_training_table(
        base_table=args.base_table,
        rerank_table=args.rerank_table,
        active_codes=active_codes,
        out_table=out_table,
        out_hard_negatives=args.out_hard_negatives,
        cfg=HardNegativeMiningConfig(
            score_col=str(args.score_col or m.get("score_col", "assistant_score")),
            top_pool_size=int(args.top_pool_size if args.top_pool_size is not None else m.get("top_pool_size", 100000)),
            max_negatives=int(args.max_negatives if args.max_negatives is not None else m.get("max_negatives", 20000)),
            min_distance=int(args.min_distance if args.min_distance is not None else m.get("min_distance", 1)),
            max_distance=int(args.max_distance if args.max_distance is not None else m.get("max_distance", 4)),
            active_score_quantile=float(
                args.active_score_quantile
                if args.active_score_quantile is not None
                else m.get("active_score_quantile", 0.50)
            ),
            include_missing_from_base=not bool(args.no_include_missing or m.get("no_include_missing", False)),
            local_target_col=str(m.get("local_target_col", "active_local_target")),
            hard_negative_col=str(m.get("hard_negative_col", "is_hard_negative")),
            distance_col=str(m.get("distance_col", "local_active_distance")),
            active_col=str(m.get("active_col", "is_active")),
        ),
    )
    logger.info("Hard-negative mining done. summary=%s", summary)


if __name__ == "__main__":
    main()
