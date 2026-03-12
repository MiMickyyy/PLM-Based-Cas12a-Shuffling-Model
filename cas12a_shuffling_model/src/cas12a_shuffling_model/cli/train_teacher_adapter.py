from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.teacher.adapter_training import (
    TeacherAdaptConfig,
    train_teacher_adapter,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _build_cfg(root_cfg: dict, args) -> TeacherAdaptConfig:
    adapt = root_cfg.get("teacher_adapt", {})
    paths = root_cfg.get("paths", {})
    return TeacherAdaptConfig(
        seed=int(args.seed if args.seed is not None else root_cfg.get("seed", 13)),
        base_model_name_or_path=str(
            args.base_model_name_or_path
            or adapt.get("base_model_name_or_path")
            or root_cfg.get("teacher", {}).get("model_name_or_path")
            or root_cfg.get("teacher", {}).get("model_name", "nferruz/ProtGPT2")
        ),
        base_model_revision=args.base_model_revision
        or adapt.get("base_model_revision")
        or root_cfg.get("teacher", {}).get("model_revision"),
        add_spaces=bool(adapt.get("add_spaces", root_cfg.get("teacher", {}).get("add_spaces", True))),
        method=str(args.method or adapt.get("method", "auto")),
        partial_last_n_layers=int(adapt.get("partial_last_n_layers", 2)),
        lora_r=int(adapt.get("lora_r", 8)),
        lora_alpha=int(adapt.get("lora_alpha", 16)),
        lora_dropout=float(adapt.get("lora_dropout", 0.05)),
        min_len=int(adapt.get("min_len", 300)),
        max_len_filter=adapt.get("max_len_filter"),
        deduplicate=bool(adapt.get("deduplicate", False)),
        train_val_fraction=float(adapt.get("train_val_fraction", 0.1)),
        max_train_sequences=adapt.get("max_train_sequences"),
        max_val_sequences=adapt.get("max_val_sequences"),
        max_length=adapt.get("max_length", 2048),
        batch_size=int(args.batch_size if args.batch_size is not None else adapt.get("batch_size", 1)),
        eval_batch_size=int(adapt.get("eval_batch_size", 1)),
        grad_accum_steps=int(
            args.grad_accum_steps if args.grad_accum_steps is not None else adapt.get("grad_accum_steps", 8)
        ),
        epochs=int(args.epochs if args.epochs is not None else adapt.get("epochs", 1)),
        lr=float(args.lr if args.lr is not None else adapt.get("lr", 1e-4)),
        weight_decay=float(adapt.get("weight_decay", 1e-4)),
        max_grad_norm=float(adapt.get("max_grad_norm", 1.0)),
        num_workers=int(adapt.get("num_workers", 0)),
        cpu_threads=adapt.get("cpu_threads"),
        interop_threads=adapt.get("interop_threads"),
        log_every_steps=int(adapt.get("log_every_steps", 10)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base-model-name-or-path", default=None)
    ap.add_argument("--base-model-revision", default=None)
    ap.add_argument("--method", choices=["auto", "lora", "partial", "full"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--deduplicate", action="store_true", default=None)
    ap.add_argument("--allow-duplicates", action="store_true")
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)

    fasta = args.fasta or cfg.get("paths", {}).get("atlas_fasta") or "cas12a.fasta"
    out_base = args.out_dir or cfg.get("teacher_adapt", {}).get(
        "output_dir", "cas12a_shuffling_model/outputs/teacher_adapt"
    )
    run_id = args.run_id or f"adapt_{int(time.time())}"
    run_dir = Path(out_base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    adapt_cfg = _build_cfg(cfg, args)
    if args.deduplicate is True:
        adapt_cfg = TeacherAdaptConfig(**{**adapt_cfg.__dict__, "deduplicate": True})
    if args.allow_duplicates:
        adapt_cfg = TeacherAdaptConfig(**{**adapt_cfg.__dict__, "deduplicate": False})
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "input_config": args.config,
                "fasta": str(fasta),
                "device": args.device,
                "resume": bool(args.resume),
                "teacher_adapt": adapt_cfg.__dict__,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    try:
        summary = train_teacher_adapter(
            fasta_path=str(fasta),
            cfg=adapt_cfg,
            out_dir=str(run_dir),
            device=args.device,
            resume=bool(args.resume),
        )
    except RuntimeError as e:
        msg = str(e)
        auto_device = args.device is None
        if auto_device and ("mps" in msg.lower() or "metal" in msg.lower()):
            logger.warning("Teacher adaptation on MPS failed; retry on CPU. reason=%s", msg)
            summary = train_teacher_adapter(
                fasta_path=str(fasta),
                cfg=adapt_cfg,
                out_dir=str(run_dir),
                device="cpu",
                resume=bool(args.resume),
            )
        else:
            raise
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Teacher adaptation finished. run_dir=%s", run_dir)
    logger.info(
        "method=%s best_val_loss=%.4f best_val_ppl=%.4f adapted_model_path=%s adapter_path=%s",
        summary["method_used"],
        summary["best_val_loss"],
        summary["best_val_ppl"],
        summary.get("adapted_model_path"),
        summary.get("adapter_path"),
    )


if __name__ == "__main__":
    main()
