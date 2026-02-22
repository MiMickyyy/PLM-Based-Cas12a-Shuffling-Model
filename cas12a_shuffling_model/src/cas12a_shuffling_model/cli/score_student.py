from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)
from cas12a_shuffling_model.student.score_student import StudentScorer
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig
from cas12a_shuffling_model.teacher.scoring_utils import (
    load_validated_domains_dict,
    resolve_validated_domains_path,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _resolve_checkpoint(config: dict, cli_checkpoint: str | None = None) -> str:
    if cli_checkpoint:
        return cli_checkpoint
    student_cfg = config.get("student", {})
    ckpt = student_cfg.get("checkpoint")
    if ckpt:
        return str(ckpt)
    out_dir = student_cfg.get("output_dir")
    if out_dir:
        p = Path(out_dir)
        if p.is_file():
            return str(p)
        candidates = sorted(p.glob("run_*/student_best.pt"), reverse=True)
        if candidates:
            return str(candidates[0])
    raise ValueError("Cannot resolve student checkpoint. Pass --checkpoint explicitly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--sequence-aa", default=None, help="Single-sequence scoring mode")
    ap.add_argument("--combo-compact", default=None, help="Single-sequence scoring via combo reconstruction")
    ap.add_argument("--input-csv", default=None, help="Batch scoring mode")
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--combo-column", default="combo_compact")
    ap.add_argument("--sequence-column", default="sequence_aa")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    window_cfg = cfg.get("teacher", {}).get("junction_window", {})
    window = JunctionWindowConfig(
        left=int(window_cfg.get("left", 25)),
        right=int(window_cfg.get("right", 25)),
    )
    checkpoint = _resolve_checkpoint(cfg, cli_checkpoint=args.checkpoint)
    scorer = StudentScorer(checkpoint_path=checkpoint, window=window, device=args.device)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        validated_domains = None
        if args.combo_column in df.columns:
            vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
            validated_domains = load_validated_domains_dict(vd_path)
        out_df = scorer.score_batch_rows(
            rows_df=df,
            validated_domains=validated_domains,
            combo_col=args.combo_column,
            seq_col=args.sequence_column,
        )
        if not args.output_csv:
            raise SystemExit("Batch mode requires --output-csv")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        logger.info("Scored rows=%d, output=%s", len(out_df), args.output_csv)
        return

    # Single-sequence mode
    seq = (args.sequence_aa or "").strip().upper()
    if not seq and args.combo_compact:
        combo = validate_combo_compact(args.combo_compact)
        vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
        validated_domains = load_validated_domains_dict(vd_path)
        seq = build_sequence_from_combo(combo, validated_domains)
        dlen = domain_lengths_from_combo(combo, validated_domains)
    else:
        dlen = None

    if not seq:
        raise SystemExit("Provide either --sequence-aa or --combo-compact, or use --input-csv")

    score = scorer.score_one(sequence_aa=seq, domain_lengths=dlen)
    out = {
        "sequence_hash": score.sequence_hash,
        "global_score": score.global_score,
        "junction_mean": score.junction_mean,
        "junction_min": score.junction_min,
    }
    for i, v in enumerate(score.junction_scores, start=1):
        out[f"junction_{i:02d}"] = v

    if args.output_csv:
        df = pd.DataFrame([{**out, "sequence_aa": seq, "combo_compact": args.combo_compact or ""}])
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        logger.info("Wrote single-sequence student score: %s", args.output_csv)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

