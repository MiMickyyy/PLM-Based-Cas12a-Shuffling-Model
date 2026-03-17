from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.composition.train_assistant import (
    AssistantModelConfig,
    AssistantTrainConfig,
    train_assistant_ranker,
)
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _feature_cols_from_config(cfg: dict) -> list[str]:
    cols = cfg.get("slot_search", {}).get("assistant_ranker", {}).get("feature_cols")
    if cols is None:
        return ["teacher_junction_mean", "teacher_junction_min", "length"]
    return [str(c) for c in cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/slot_search.yaml")
    ap.add_argument("--data-table", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--feature-cols", default=None, help="comma-separated feature columns; empty => slot-only")
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    c = cfg.get("slot_search", {}).get("assistant_ranker", {})
    m = c.get("model", {})
    t = c.get("train", {})

    run_dir = args.out_dir or c.get("output_dir")
    if not run_dir:
        run_dir = str(
            Path("cas12a_shuffling_model/outputs/assistant_ranker")
            / f"run_{int(time.time())}"
        )
    else:
        run_dir = str(Path(run_dir) / f"run_{int(time.time())}")

    model_cfg = AssistantModelConfig(
        slot_embed_dim=int(m.get("slot_embed_dim", 16)),
        hidden_dim=int(m.get("hidden_dim", 128)),
        num_layers=int(m.get("num_layers", 3)),
        dropout=float(m.get("dropout", 0.1)),
        use_extra_features=bool(m.get("use_extra_features", True)),
    )
    train_cfg = AssistantTrainConfig(
        seed=int(cfg.get("seed", 13)),
        batch_size=int(args.batch_size if args.batch_size is not None else t.get("batch_size", 256)),
        epochs=int(args.epochs if args.epochs is not None else t.get("epochs", 30)),
        lr=float(args.lr if args.lr is not None else t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        grad_clip=float(t.get("grad_clip", 1.0)),
        val_fraction=float(t.get("val_fraction", 0.2)),
        top_weight=float(t.get("top_weight", 1.0)),
        corr_weight=float(t.get("corr_weight", 0.3)),
        pair_weight=float(t.get("pair_weight", 0.2)),
        pair_min_gap=float(t.get("pair_min_gap", 0.01)),
        pair_near_tie_gap=float(t.get("pair_near_tie_gap", 0.01)),
        pair_easy_ratio=float(t.get("pair_easy_ratio", 0.2)),
        pair_medium_ratio=float(t.get("pair_medium_ratio", 0.5)),
        pair_hard_ratio=float(t.get("pair_hard_ratio", 0.3)),
        pair_pairs_per_batch=int(t.get("pair_pairs_per_batch", 512)),
        pair_margin=float(t.get("pair_margin", 0.0)),
        pair_margin_alpha=float(t.get("pair_margin_alpha", 0.5)),
        pair_margin_min=float(t.get("pair_margin_min", 0.0)),
        pair_margin_max=float(t.get("pair_margin_max", 0.3)),
        top_fraction=float(t.get("top_fraction", 0.1)),
        top_pairs_per_batch=int(t.get("top_pairs_per_batch", 256)),
        top_margin=float(t.get("top_margin", 0.0)),
        chimera_only=bool(t.get("chimera_only", True)),
        target_col=str(args.target_col or t.get("target_col", "teacher_seq_score_norm")),
        experimental_label_col=t.get("experimental_label_col"),
        experimental_weight=float(t.get("experimental_weight", 0.0)),
        oversample_top_fraction=float(t.get("oversample_top_fraction", 0.10)),
        oversample_weight=float(t.get("oversample_weight", 2.0)),
        best_metric=str(t.get("best_metric", "global_corr_chimera")),
        num_workers=int(t.get("num_workers", 0)),
        device=args.device,
        cpu_threads=t.get("cpu_threads"),
        interop_threads=t.get("interop_threads"),
    )
    if args.feature_cols is None:
        feature_cols = _feature_cols_from_config(cfg)
    elif not args.feature_cols.strip():
        feature_cols = []
    else:
        feature_cols = [x.strip() for x in args.feature_cols.split(",") if x.strip()]

    summary = train_assistant_ranker(
        data_table=args.data_table,
        out_dir=run_dir,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        feature_cols=feature_cols,
    )
    logger.info(
        "Assistant training finished. run_dir=%s best_epoch=%s best_metric=%s best_metrics=%s",
        run_dir,
        summary.get("best_epoch"),
        summary.get("best_metric"),
        summary.get("best_metrics"),
    )


if __name__ == "__main__":
    main()

