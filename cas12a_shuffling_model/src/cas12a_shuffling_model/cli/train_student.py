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
        embed_dim=int(m.get("embed_dim", 128)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        num_layers=int(m.get("num_layers", 2)),
        dropout=float(m.get("dropout", 0.1)),
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
        num_workers=int(t.get("num_workers", 0)),
        device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--distill-csv", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
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
    train_cfg = _build_train_cfg(cfg, device=args.device)
    if args.epochs is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "epochs": int(args.epochs)})
    if args.batch_size is not None:
        train_cfg = StudentTrainConfig(**{**train_cfg.__dict__, "batch_size": int(args.batch_size)})

    teacher_window = cfg.get("teacher", {}).get("junction_window", {})
    window = JunctionWindowConfig(
        left=int(teacher_window.get("left", 25)),
        right=int(teacher_window.get("right", 25)),
    )

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    validated_domains = load_validated_domains_dict(vd_path)

    summary = train_student_from_distill_csv(
        distill_csv=distill_csv,
        validated_domains=validated_domains,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        window=window,
        out_dir=run_dir,
    )
    logger.info("Student training finished. Run dir: %s", run_dir)
    logger.info("Best epoch=%s, best_val_loss=%.6f", summary["best_epoch"], summary["best_val_loss"])


if __name__ == "__main__":
    main()
