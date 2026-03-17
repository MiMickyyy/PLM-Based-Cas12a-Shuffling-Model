from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.slot_scorer import SlotScorerConfig
from cas12a_shuffling_model.composition.train_slot_scorer import (
    SlotScorerTrainConfig,
    train_slot_scorer,
)
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--data-table", required=True)
    ap.add_argument("--assistant-checkpoint", default=None)
    ap.add_argument("--cache-scored-table", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--active-table", default=None, help="Path to active combo table (e.g., Sequence_Result.xlsx)")
    ap.add_argument("--active-loss-weight", type=float, default=None)
    ap.add_argument("--active-sample-weight", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    c = cfg.get("slot_search", {}).get("slot_scorer", {})
    m = c.get("model", {})
    t = c.get("train", {})

    run_dir = args.out_dir or c.get("output_dir")
    if not run_dir:
        run_dir = str(Path("cas12a_shuffling_model/outputs/slot_scorer") / f"run_{int(time.time())}")
    else:
        run_dir = str(Path(run_dir) / f"run_{int(time.time())}")

    model_cfg = SlotScorerConfig(
        slot_embed_dim=int(m.get("slot_embed_dim", 8)),
        mlp_hidden_dim=int(m.get("mlp_hidden_dim", 64)),
        mlp_layers=int(m.get("mlp_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
        enable_pairwise=bool(m.get("enable_pairwise", True)),
    )
    train_cfg = SlotScorerTrainConfig(
        seed=int(cfg.get("seed", 13)),
        batch_size=int(args.batch_size if args.batch_size is not None else t.get("batch_size", 1024)),
        epochs=int(args.epochs if args.epochs is not None else t.get("epochs", 40)),
        lr=float(args.lr if args.lr is not None else t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        grad_clip=float(t.get("grad_clip", 1.0)),
        val_fraction=float(t.get("val_fraction", 0.2)),
        top_weight=float(t.get("top_weight", 1.0)),
        corr_weight=float(t.get("corr_weight", 0.4)),
        pair_weight=float(t.get("pair_weight", 0.2)),
        pair_min_gap=float(t.get("pair_min_gap", 0.01)),
        pair_near_tie_gap=float(t.get("pair_near_tie_gap", 0.01)),
        pair_easy_ratio=float(t.get("pair_easy_ratio", 0.2)),
        pair_medium_ratio=float(t.get("pair_medium_ratio", 0.5)),
        pair_hard_ratio=float(t.get("pair_hard_ratio", 0.3)),
        pair_pairs_per_batch=int(t.get("pair_pairs_per_batch", 1024)),
        pair_margin=float(t.get("pair_margin", 0.0)),
        pair_margin_alpha=float(t.get("pair_margin_alpha", 0.5)),
        pair_margin_min=float(t.get("pair_margin_min", 0.0)),
        pair_margin_max=float(t.get("pair_margin_max", 0.3)),
        top_fraction=float(t.get("top_fraction", 0.10)),
        top_pairs_per_batch=int(t.get("top_pairs_per_batch", 512)),
        top_margin=float(t.get("top_margin", 0.0)),
        hard_neg_weight=float(t.get("hard_neg_weight", 0.3)),
        hard_neg_top_fraction=float(t.get("hard_neg_top_fraction", 0.10)),
        hard_neg_pairs_per_batch=int(t.get("hard_neg_pairs_per_batch", 512)),
        hard_neg_margin=float(t.get("hard_neg_margin", 0.05)),
        oversample_top_fraction=float(t.get("oversample_top_fraction", 0.10)),
        oversample_weight=float(t.get("oversample_weight", 2.0)),
        active_codes_path=str(args.active_table or t.get("active_codes_path") or cfg.get("paths", {}).get("sequence_results", "")) or None,
        active_sample_weight=float(
            args.active_sample_weight if args.active_sample_weight is not None else t.get("active_sample_weight", 1.0)
        ),
        active_force_train=bool(t.get("active_force_train", True)),
        active_loss_weight=float(
            args.active_loss_weight if args.active_loss_weight is not None else t.get("active_loss_weight", 0.0)
        ),
        active_pairs_per_batch=int(t.get("active_pairs_per_batch", 512)),
        active_margin=float(t.get("active_margin", 0.10)),
        active_min_target_gap=float(t.get("active_min_target_gap", 0.0)),
        target_col=str(args.target_col or t.get("target_col", "assistant_score")),
        best_metric=str(t.get("best_metric", "global_corr_chimera")),
        num_workers=int(t.get("num_workers", 0)),
        device=args.device,
        cpu_threads=t.get("cpu_threads"),
        interop_threads=t.get("interop_threads"),
    )
    summary = train_slot_scorer(
        data_table=args.data_table,
        out_dir=run_dir,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        assistant_checkpoint=args.assistant_checkpoint,
        cache_scored_table=args.cache_scored_table,
    )
    logger.info(
        "Slot scorer training finished. run_dir=%s best_epoch=%s best_metric=%s best_metrics=%s",
        run_dir,
        summary.get("best_epoch"),
        summary.get("best_metric"),
        summary.get("best_metrics"),
    )


if __name__ == "__main__":
    main()
