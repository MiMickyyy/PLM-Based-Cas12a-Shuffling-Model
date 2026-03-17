from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.rerank import RerankConfig, rerank_shortlist
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.scoring_utils import (
    load_validated_domains_dict,
    resolve_validated_domains_path,
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
    ap.add_argument("--include-sequence", action="store_true")
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    c = cfg.get("slot_search", {}).get("rerank", {})

    run_dir = args.out_dir or c.get("output_dir")
    if not run_dir:
        run_dir = str(Path("cas12a_shuffling_model/outputs/rerank") / f"run_{int(time.time())}")
    else:
        run_dir = str(Path(run_dir) / f"run_{int(time.time())}")

    domains = None
    include_sequence = bool(args.include_sequence or c.get("include_sequence", False))
    if include_sequence:
        vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
        domains = load_validated_domains_dict(vd_path)

    outputs = rerank_shortlist(
        shortlist_table=args.shortlist_table,
        assistant_checkpoint=args.assistant_checkpoint,
        out_dir=run_dir,
        cfg=RerankConfig(
            batch_size=int(args.batch_size if args.batch_size is not None else c.get("batch_size", 2048)),
            top_k=int(args.top_k if args.top_k is not None else c.get("top_k", 50)),
            include_sequence=include_sequence,
        ),
        validated_domains=domains,
        device=args.device,
    )
    logger.info("Rerank done. outputs=%s", outputs)


if __name__ == "__main__":
    main()

