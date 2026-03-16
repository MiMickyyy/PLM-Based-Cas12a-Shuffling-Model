from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig
from cas12a_shuffling_model.teacher.scoring_utils import (
    load_validated_domains_dict,
    resolve_validated_domains_path,
)
from cas12a_shuffling_model.student.train_student import (
    StudentModelConfig,
    StudentTrainConfig,
    train_student_from_distill_csv,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _build_model_cfg(cfg: dict) -> StudentModelConfig:
    m = cfg.get("student", {}).get("model", {})
    return StudentModelConfig(
        model_type=str(m.get("model_type", "gru")),
        embed_dim=int(m.get("embed_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        num_layers=int(m.get("num_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
        num_heads=int(m.get("num_heads", 4)),
        ff_dim=int(m.get("ff_dim", 512)),
        max_positions=int(m.get("max_positions", 4096)),
    )


def _build_train_cfg(cfg: dict, device: str | None = None) -> StudentTrainConfig:
    t = cfg.get("student", {}).get("train", {})
    return StudentTrainConfig(
        seed=int(cfg.get("seed", 13)),
        batch_size=int(t.get("batch_size", 8)),
        epochs=int(t.get("epochs", 3)),
        lr=float(t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        grad_clip=float(t.get("grad_clip", 1.0)),
        val_fraction=float(t.get("val_fraction", 0.2)),
        nll_weight=float(t.get("nll_weight", 1.0)),
        global_weight=float(t.get("global_weight", 1.0)),
        junction_weight=float(t.get("junction_weight", 1.0)),
        natural_global_weight=float(t.get("natural_global_weight", 1.0)),
        chimera_global_weight=float(t.get("chimera_global_weight", 1.0)),
        natural_junction_weight=float(t.get("natural_junction_weight", 1.0)),
        chimera_junction_weight=float(t.get("chimera_junction_weight", 1.0)),
        correlation_weight=float(t.get("correlation_weight", 0.0)),
        correlation_on_chimera_only=bool(t.get("correlation_on_chimera_only", True)),
        pairwise_weight=float(t.get("pairwise_weight", 0.0)),
        pairwise_margin=float(t.get("pairwise_margin", 0.0)),
        pairwise_pairs_per_batch=int(t.get("pairwise_pairs_per_batch", 64)),
        pairwise_min_teacher_diff=float(t.get("pairwise_min_teacher_diff", 0.01)),
        pairwise_ignore_close_diff=float(t.get("pairwise_ignore_close_diff", 0.0)),
        pairwise_hard_ratio=float(t.get("pairwise_hard_ratio", 0.5)),
        pairwise_medium_ratio=float(t.get("pairwise_medium_ratio", 0.5)),
        pairwise_easy_ratio=float(t.get("pairwise_easy_ratio", 0.2)),
        pairwise_length_bin_size=int(t.get("pairwise_length_bin_size", 64)),
        pairwise_margin_alpha=float(t.get("pairwise_margin_alpha", 0.5)),
        pairwise_margin_min=float(t.get("pairwise_margin_min", 0.0)),
        pairwise_margin_max=float(t.get("pairwise_margin_max", 0.3)),
        pairwise_on_chimera_only=bool(t.get("pairwise_on_chimera_only", True)),
        pairwise_warmup_epochs=int(t.get("pairwise_warmup_epochs", 0)),
        pairwise_ramp_epochs=int(t.get("pairwise_ramp_epochs", 0)),
        stage_a_natural_epochs=int(t.get("stage_a_natural_epochs", 0)),
        stage_a_batch_size=t.get("stage_a_batch_size"),
        stage_b_chimera_only=bool(t.get("stage_b_chimera_only", False)),
        stage_b_lr_scale=float(t.get("stage_b_lr_scale", 1.0)),
        normalize_teacher_global=bool(t.get("normalize_teacher_global", False)),
        normalize_length_bin_size=int(t.get("normalize_length_bin_size", 64)),
        normalize_min_group_size=int(t.get("normalize_min_group_size", 32)),
        nll_final_weight=t.get("nll_final_weight"),
        nll_decay_start_epoch=int(t.get("nll_decay_start_epoch", 1)),
        nll_decay_end_epoch=int(t.get("nll_decay_end_epoch", 1)),
        top_loss_weight=float(t.get("top_loss_weight", 0.0)),
        top_fraction=float(t.get("top_fraction", 0.1)),
        top_margin=float(t.get("top_margin", 0.0)),
        top_pairs_per_batch=int(t.get("top_pairs_per_batch", 64)),
        top_on_chimera_only=bool(t.get("top_on_chimera_only", True)),
        topk_fracs=tuple(float(x) for x in t.get("topk_fracs", [0.01, 0.05, 0.10])),
        best_metric=str(t.get("best_metric", "val_loss")),
        best_metric_mode=str(t.get("best_metric_mode", "auto")),
        balance_source_types=bool(t.get("balance_source_types", False)),
        num_workers=int(t.get("num_workers", 0)),
        device=device,
        cpu_threads=t.get("cpu_threads"),
        interop_threads=t.get("interop_threads"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--distill-csv", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--model-type", choices=["gru", "transformer"], default=None)
    ap.add_argument("--embed-dim", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--ff-dim", type=int, default=None)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--nll-weight", type=float, default=None)
    ap.add_argument("--global-weight", type=float, default=None)
    ap.add_argument("--junction-weight", type=float, default=None)
    ap.add_argument("--natural-global-weight", type=float, default=None)
    ap.add_argument("--chimera-global-weight", type=float, default=None)
    ap.add_argument("--natural-junction-weight", type=float, default=None)
    ap.add_argument("--chimera-junction-weight", type=float, default=None)
    ap.add_argument("--correlation-weight", type=float, default=None)
    ap.add_argument("--correlation-on-all-sources", action="store_true")
    ap.add_argument("--pairwise-weight", type=float, default=None)
    ap.add_argument("--pairwise-margin", type=float, default=None)
    ap.add_argument("--pairwise-pairs-per-batch", type=int, default=None)
    ap.add_argument("--pairwise-min-teacher-diff", type=float, default=None)
    ap.add_argument("--pairwise-ignore-close-diff", type=float, default=None)
    ap.add_argument("--pairwise-warmup-epochs", type=int, default=None)
    ap.add_argument("--pairwise-ramp-epochs", type=int, default=None)
    ap.add_argument("--pairwise-on-all-sources", action="store_true")
    ap.add_argument("--stage-a-natural-epochs", type=int, default=None)
    ap.add_argument("--stage-b-chimera-only", action="store_true")
    ap.add_argument("--stage-b-use-all-sources", action="store_true")
    ap.add_argument("--stage-b-lr-scale", type=float, default=None)
    ap.add_argument("--nll-final-weight", type=float, default=None)
    ap.add_argument("--nll-decay-start-epoch", type=int, default=None)
    ap.add_argument("--nll-decay-end-epoch", type=int, default=None)
    ap.add_argument("--top-loss-weight", type=float, default=None)
    ap.add_argument("--normalize-teacher-global", action="store_true")
    ap.add_argument("--normalize-length-bin-size", type=int, default=None)
    ap.add_argument("--best-metric", type=str, default=None)
    ap.add_argument("--best-metric-mode", choices=["auto", "max", "min"], default=None)
    ap.add_argument("--balance-source-types", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)

    distill_csv = args.distill_csv or cfg.get("student", {}).get("distill_csv")
    if not distill_csv:
        distill_csv = cfg.get("distill", {}).get("output_csv")
    if not distill_csv:
        raise SystemExit("Missing distill CSV path; pass --distill-csv or set student.distill_csv")

    run_dir = args.out_dir or cfg.get("student", {}).get("output_dir")
    if not run_dir:
        run_dir = "cas12a_shuffling_model/outputs/student"
    run_dir = str(Path(run_dir) / f"run_{int(time.time())}")

    model_cfg = _build_model_cfg(cfg)
    if args.model_type is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "model_type": str(args.model_type)})
    if args.embed_dim is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "embed_dim": int(args.embed_dim)})
    if args.hidden_dim is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "hidden_dim": int(args.hidden_dim)})
    if args.num_layers is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "num_layers": int(args.num_layers)})
    if args.dropout is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "dropout": float(args.dropout)})
    if args.num_heads is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "num_heads": int(args.num_heads)})
    if args.ff_dim is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "ff_dim": int(args.ff_dim)})
    if args.max_positions is not None:
        model_cfg = StudentModelConfig(**{**model_cfg.__dict__, "max_positions": int(args.max_positions)})

    train_cfg = _build_train_cfg(cfg, device=args.device)
    if args.epochs is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "epochs": int(args.epochs)})
    if args.batch_size is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "batch_size": int(args.batch_size)})
    if args.nll_weight is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "nll_weight": float(args.nll_weight)})
    if args.global_weight is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "global_weight": float(args.global_weight)})
    if args.junction_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "junction_weight": float(args.junction_weight)}
        )
    if args.natural_global_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "natural_global_weight": float(args.natural_global_weight)}
        )
    if args.chimera_global_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "chimera_global_weight": float(args.chimera_global_weight)}
        )
    if args.natural_junction_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "natural_junction_weight": float(args.natural_junction_weight)}
        )
    if args.chimera_junction_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "chimera_junction_weight": float(args.chimera_junction_weight)}
        )
    if args.correlation_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "correlation_weight": float(args.correlation_weight)}
        )
    if args.correlation_on_all_sources:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "correlation_on_chimera_only": False}
        )
    if args.pairwise_weight is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "pairwise_weight": float(args.pairwise_weight)})
    if args.pairwise_margin is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "pairwise_margin": float(args.pairwise_margin)})
    if args.pairwise_pairs_per_batch is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "pairwise_pairs_per_batch": int(args.pairwise_pairs_per_batch)}
        )
    if args.pairwise_min_teacher_diff is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "pairwise_min_teacher_diff": float(args.pairwise_min_teacher_diff)}
        )
    if args.pairwise_ignore_close_diff is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "pairwise_ignore_close_diff": float(args.pairwise_ignore_close_diff)}
        )
    if args.pairwise_warmup_epochs is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "pairwise_warmup_epochs": int(args.pairwise_warmup_epochs)}
        )
    if args.pairwise_ramp_epochs is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "pairwise_ramp_epochs": int(args.pairwise_ramp_epochs)}
        )
    if args.pairwise_on_all_sources:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "pairwise_on_chimera_only": False})
    if args.stage_a_natural_epochs is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "stage_a_natural_epochs": int(args.stage_a_natural_epochs)}
        )
    if args.stage_b_chimera_only:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "stage_b_chimera_only": True}
        )
    if args.stage_b_use_all_sources:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "stage_b_chimera_only": False}
        )
    if args.stage_b_lr_scale is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "stage_b_lr_scale": float(args.stage_b_lr_scale)}
        )
    if args.nll_final_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "nll_final_weight": float(args.nll_final_weight)}
        )
    if args.nll_decay_start_epoch is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "nll_decay_start_epoch": int(args.nll_decay_start_epoch)}
        )
    if args.nll_decay_end_epoch is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "nll_decay_end_epoch": int(args.nll_decay_end_epoch)}
        )
    if args.top_loss_weight is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "top_loss_weight": float(args.top_loss_weight)}
        )
    if args.normalize_teacher_global:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "normalize_teacher_global": True}
        )
    if args.normalize_length_bin_size is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "normalize_length_bin_size": int(args.normalize_length_bin_size)}
        )
    if args.best_metric is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "best_metric": str(args.best_metric)})
    if args.best_metric_mode is not None:
        train_cfg = StudentTrainConfig(
            **{**train_cfg.__dict__, "best_metric_mode": str(args.best_metric_mode)}
        )
    if args.balance_source_types:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "balance_source_types": True})

    teacher_window = cfg.get("teacher", {}).get("junction_window", {})
    window = JunctionWindowConfig(
        left=int(teacher_window.get("left", 25)),
        right=int(teacher_window.get("right", 25)),
    )

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    validated_domains = load_validated_domains_dict(vd_path)

    try:
        summary = train_student_from_distill_csv(
            distill_csv=distill_csv,
            validated_domains=validated_domains,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            window=window,
            out_dir=run_dir,
        )
    except RuntimeError as e:
        msg = str(e)
        auto_device = args.device is None
        if auto_device and ("mps" in msg.lower() or "metal" in msg.lower()):
            logger.warning("Student training on MPS failed; retry on CPU. reason=%s", msg)
            cpu_train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "device": "cpu"})
            summary = train_student_from_distill_csv(
                distill_csv=distill_csv,
                validated_domains=validated_domains,
                model_cfg=model_cfg,
                train_cfg=cpu_train_cfg,
                window=window,
                out_dir=run_dir,
            )
        else:
            raise
    logger.info("Student training finished. Run dir: %s", run_dir)
    logger.info(
        "Best epoch=%s, best_metric=%s, best_metric_value=%s, best_val_loss=%.6f",
        summary.get("best_epoch"),
        summary.get("best_metric"),
        summary.get("best_metric_value"),
        summary.get("best_val_loss", float("nan")),
    )


if __name__ == "__main__":
    main()
