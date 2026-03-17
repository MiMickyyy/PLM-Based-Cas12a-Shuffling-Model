from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.full_scan import FullScanConfig, scan_full_space
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--slot-scorer-checkpoint", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--shortlist-size", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--save-all-scores", action="store_true")
    ap.add_argument("--no-decomposition", action="store_true")
    ap.add_argument("--progress-every-batches", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--end-index", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    c = cfg.get("slot_search", {}).get("scan", {})

    run_dir = args.out_dir or c.get("output_dir")
    if not run_dir:
        run_dir = str(Path("cas12a_shuffling_model/outputs/full_scan") / f"run_{int(time.time())}")
    else:
        run_dir = str(Path(run_dir) / f"run_{int(time.time())}")

    run_cfg = FullScanConfig(
        batch_size=int(args.batch_size if args.batch_size is not None else c.get("batch_size", 65536)),
        shortlist_size=int(
            args.shortlist_size if args.shortlist_size is not None else c.get("shortlist_size", 20000)
        ),
        top_k=int(args.top_k if args.top_k is not None else c.get("top_k", 50)),
        save_all_scores=bool(args.save_all_scores or c.get("save_all_scores", False)),
        include_decomposition=not bool(args.no_decomposition),
        progress_every_batches=int(
            args.progress_every_batches
            if args.progress_every_batches is not None
            else c.get("progress_every_batches", 4)
        ),
        start_index=int(args.start_index if args.start_index is not None else c.get("start_index", 0)),
        end_index=int(args.end_index) if args.end_index is not None else c.get("end_index"),
    )
    outputs = scan_full_space(
        slot_scorer_checkpoint=args.slot_scorer_checkpoint,
        out_dir=run_dir,
        cfg=run_cfg,
        device=args.device,
    )
    logger.info("Full scan done. outputs=%s", outputs)


if __name__ == "__main__":
    main()

