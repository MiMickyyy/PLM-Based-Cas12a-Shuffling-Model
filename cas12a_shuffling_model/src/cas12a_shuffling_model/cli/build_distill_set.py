from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.domain.chimera_builder import extract_active_rows
from cas12a_shuffling_model.io.cas12a_corpus import read_cas12a_fasta, sample_sequences
from cas12a_shuffling_model.io.loaders import load_yaml, read_sequence_results_table
from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)
from cas12a_shuffling_model.search.sampler import sample_combo_compacts
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
    with_teacher_overrides,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _actives_combo_list(config: dict) -> list[str]:
    act_cfg = config.get("actives", {})
    seq_path = config.get("paths", {}).get("sequence_results")
    if not seq_path:
        return []
    df = read_sequence_results_table(seq_path)
    active_df, slot_cols = extract_active_rows(
        df,
        slot_columns=act_cfg.get("slot_columns"),
        allowed_letters=act_cfg.get("allowed_letters", ["A", "L", "F", "M"]),
    )
    combos = []
    for _, row in active_df.iterrows():
        combo = validate_combo_compact("".join(str(row[c]).strip().upper() for c in slot_cols))
        combos.append(combo)
    return combos


def _rows_from_chimera_combos(
    *,
    combos: list[str],
    domains: dict[tuple[str, int], str],
    scorer,
    run_id: str,
    seed: int,
    teacher_batch_size: int,
) -> list[dict]:
    rows = []
    combos = sorted(set(combos))
    seqs = [build_sequence_from_combo(c, domains) for c in combos]
    dlen = [domain_lengths_from_combo(c, domains) for c in combos]
    scores = scorer.score_many(
        seqs_aa=seqs,
        domain_lengths_list=dlen,
        batch_size=teacher_batch_size,
    )
    for combo, seq, score in zip(combos, seqs, scores):
        rec = {
            "source_run_id": run_id,
            "seed": seed,
            "source_type": "chimera",
            "combo_compact": combo,
            "sequence_aa": seq,
            "sequence_hash": score.seq_hash,
            "global_score": score.global_score,
            "junction_mean": score.junction_mean,
            "junction_min": score.junction_min,
            "teacher_cache_hit": bool(score.from_cache),
        }
        for i, v in enumerate(score.junction_scores, start=1):
            rec[f"junction_{i:02d}"] = v
        rows.append(rec)
    return rows


def _rows_from_natural_sequences(
    *,
    sequences: list[str],
    scorer,
    run_id: str,
    seed: int,
    teacher_batch_size: int,
    with_junction: bool,
) -> list[dict]:
    rows = []
    scores = scorer.score_many(
        seqs_aa=sequences,
        domain_lengths_list=[None] * len(sequences),
        batch_size=teacher_batch_size,
    )
    for seq, score in zip(sequences, scores):
        if with_junction:
            jvals = list(score.junction_scores)
            jmean = score.junction_mean
            jmin = score.junction_min
        else:
            jvals = [float("nan")] * 10
            jmean = float("nan")
            jmin = float("nan")

        rec = {
            "source_run_id": run_id,
            "seed": seed,
            "source_type": "natural",
            "combo_compact": "",
            "sequence_aa": seq,
            "sequence_hash": score.seq_hash,
            "global_score": score.global_score,
            "junction_mean": jmean,
            "junction_min": jmin,
            "teacher_cache_hit": bool(score.from_cache),
        }
        for i, v in enumerate(jvals, start=1):
            rec[f"junction_{i:02d}"] = v
        rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--source-mode", choices=["chimera", "natural", "mixed"], default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="legacy alias for chimera sample size")
    ap.add_argument("--chimera-samples", type=int, default=None)
    ap.add_argument("--natural-samples", type=int, default=None)
    ap.add_argument("--include-actives", action="store_true")
    ap.add_argument("--natural-fasta", default=None)
    ap.add_argument("--natural-min-len", type=int, default=None)
    ap.add_argument("--natural-max-len", type=int, default=None)
    ap.add_argument("--natural-deduplicate", action="store_true")
    ap.add_argument("--natural-allow-duplicates", action="store_true")
    ap.add_argument("--natural-with-junction", action="store_true")
    ap.add_argument("--teacher-model-name-or-path", default=None)
    ap.add_argument("--teacher-model-source", choices=["hf", "local"], default=None)
    ap.add_argument("--teacher-adapter-path", default=None)
    ap.add_argument("--teacher-model-revision", default=None)
    ap.add_argument("--teacher-batch-size", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    cfg = with_teacher_overrides(
        cfg,
        model_name_or_path=args.teacher_model_name_or_path,
        model_source=args.teacher_model_source,
        adapter_path=args.teacher_adapter_path,
        model_revision=args.teacher_model_revision,
    )
    dist_cfg = cfg.get("distill", {})

    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 13))
    source_mode = str(args.source_mode or dist_cfg.get("source_mode", "chimera")).lower()
    if source_mode not in {"chimera", "natural", "mixed"}:
        raise SystemExit(f"Invalid source_mode: {source_mode}")

    chimera_n = (
        int(args.chimera_samples)
        if args.chimera_samples is not None
        else int(
            args.n_samples
            if args.n_samples is not None
            else dist_cfg.get("chimera_samples", dist_cfg.get("n_samples", 64))
        )
    )
    natural_n = (
        int(args.natural_samples)
        if args.natural_samples is not None
        else int(dist_cfg.get("natural_samples", 128))
    )
    teacher_batch_size = (
        int(args.teacher_batch_size)
        if args.teacher_batch_size is not None
        else int(dist_cfg.get("teacher_batch_size", 1))
    )
    natural_with_junction = bool(
        args.natural_with_junction or dist_cfg.get("natural_with_junction", False)
    )
    if args.natural_allow_duplicates:
        natural_deduplicate = False
    elif args.natural_deduplicate:
        natural_deduplicate = True
    else:
        natural_deduplicate = bool(dist_cfg.get("natural_deduplicate", True))

    out_csv = args.out_csv
    if not out_csv:
        out_csv = dist_cfg.get("output_csv")
    if not out_csv:
        out_processed = cfg.get("paths", {}).get("out_processed_dir")
        if not out_processed:
            raise ValueError("Missing output path: set --out-csv or distill.output_csv")
        out_csv = str(Path(out_processed) / "distill_teacher_scores.csv")

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    domains = load_validated_domains_dict(vd_path)
    scorer = build_teacher_scorer_from_config(cfg, device=args.device)

    run_id = f"distill_{int(time.time())}_{seed}"
    rows: list[dict] = []

    if source_mode in {"chimera", "mixed"}:
        combos = sample_combo_compacts(n=chimera_n, seed=seed)
        if args.include_actives or bool(dist_cfg.get("include_actives", False)):
            combos.extend(_actives_combo_list(cfg))
        rows.extend(
            _rows_from_chimera_combos(
                combos=combos,
                domains=domains,
                scorer=scorer,
                run_id=run_id,
                seed=seed,
                teacher_batch_size=teacher_batch_size,
            )
        )

    if source_mode in {"natural", "mixed"}:
        fasta_path = (
            args.natural_fasta
            or dist_cfg.get("natural_fasta")
            or cfg.get("paths", {}).get("atlas_fasta")
        )
        if not fasta_path:
            raise SystemExit("natural source requested but natural_fasta/paths.atlas_fasta is missing")
        natural_max_len = (
            int(args.natural_max_len)
            if args.natural_max_len is not None
            else dist_cfg.get("natural_max_len")
        )
        nat_records = read_cas12a_fasta(
            fasta_path=fasta_path,
            min_len=int(
                args.natural_min_len
                if args.natural_min_len is not None
                else dist_cfg.get("natural_min_len", 300)
            ),
            max_len=int(natural_max_len) if natural_max_len is not None else None,
            deduplicate=natural_deduplicate,
        )
        nat_picked = sample_sequences(nat_records, n=natural_n, seed=seed + 17)
        nat_seqs = [r.sequence_aa for r in nat_picked]
        rows.extend(
            _rows_from_natural_sequences(
                sequences=nat_seqs,
                scorer=scorer,
                run_id=run_id,
                seed=seed,
                teacher_batch_size=teacher_batch_size,
                with_junction=natural_with_junction,
            )
        )

    out_df = pd.DataFrame(rows).reset_index(drop=True)
    ordered = [
        "source_run_id",
        "seed",
        "source_type",
        "combo_compact",
        "sequence_aa",
        "sequence_hash",
        "global_score",
        "junction_mean",
        "junction_min",
        "teacher_cache_hit",
    ] + [f"junction_{i:02d}" for i in range(1, 11)]
    for col in ordered:
        if col not in out_df.columns:
            out_df[col] = math.nan if col.startswith("junction_") else ""
    out_df = out_df[ordered]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info(
        "Wrote distill set rows=%d chimera=%d natural=%d cache_hits=%d output=%s",
        len(out_df),
        int((out_df["source_type"] == "chimera").sum()),
        int((out_df["source_type"] == "natural").sum()),
        int(out_df["teacher_cache_hit"].sum()) if "teacher_cache_hit" in out_df.columns else 0,
        out_csv,
    )


if __name__ == "__main__":
    main()
