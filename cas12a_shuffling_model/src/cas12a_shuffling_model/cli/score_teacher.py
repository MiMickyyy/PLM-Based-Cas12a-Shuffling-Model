from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
    score_rows_with_teacher,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--combo-column", default="combo_compact")
    ap.add_argument("--sequence-column", default="sequence_aa")
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    config = load_yaml(args.config)

    in_df = pd.read_csv(args.input_csv)
    if args.limit is not None and args.limit > 0:
        in_df = in_df.head(args.limit).copy()

    validated_domains = None
    if args.combo_column in in_df.columns:
        vd_path = resolve_validated_domains_path(config, cli_path=args.validated_domains)
        validated_domains = load_validated_domains_dict(vd_path)
        logger.info("Loaded validated domains from %s", vd_path)

    scorer = build_teacher_scorer_from_config(config, device=args.device)
    out_df = score_rows_with_teacher(
        rows_df=in_df,
        scorer=scorer,
        validated_domains=validated_domains,
        combo_col=args.combo_column,
        seq_col=args.sequence_column,
    )

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    cache_hits = int(out_df["teacher_cache_hit"].sum()) if "teacher_cache_hit" in out_df.columns else 0
    logger.info("Scored rows=%d, cache_hits=%d, output=%s", len(out_df), cache_hits, args.output_csv)


if __name__ == "__main__":
    main()

