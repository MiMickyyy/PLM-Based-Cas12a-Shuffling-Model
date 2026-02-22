from __future__ import annotations

import argparse
import heapq
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from cas12a_shuffling_model.calibration.calibrator import apply_calibration, load_calibration_artifact
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.search.rank_pipeline import greedy_diversity_select
from cas12a_shuffling_model.search.sampler import combo_from_index, sample_combo_compacts
from cas12a_shuffling_model.student.score_student import StudentScorer
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
    score_rows_with_teacher,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

TOTAL_COMBOS = 4**11
ALPHABET = ("A", "L", "F", "M")


def _latest_calibration_paths(base_dir: str) -> tuple[str, str]:
    root = Path(base_dir)
    runs = sorted([p for p in root.glob("cal_*") if p.is_dir()], reverse=True)
    for r in runs:
        model_path = r / "calibration_model.joblib"
        meta_path = r / "calibration_meta.json"
        if model_path.exists() and meta_path.exists():
            return str(model_path), str(meta_path)
    raise FileNotFoundError(f"No calibration artifacts found under {base_dir}")


def _resolve_student_checkpoint(cfg: dict, cli_checkpoint: str | None) -> str:
    if cli_checkpoint:
        return cli_checkpoint
    student_cfg = cfg.get("student", {})
    ckpt = student_cfg.get("checkpoint")
    if ckpt:
        return str(ckpt)
    out_dir = student_cfg.get("output_dir", "cas12a_shuffling_model/outputs/student")
    runs = sorted(Path(out_dir).glob("run_*/student_best.pt"), reverse=True)
    if not runs:
        raise FileNotFoundError("No student checkpoint found; pass --student-checkpoint")
    return str(runs[0])


def _student_rank_score(
    row: dict[str, Any], w_global: float, w_jmean: float, w_jmin: float
) -> float:
    return (
        w_global * float(row["global_score"])
        + w_jmean * float(row["junction_mean"])
        + w_jmin * float(row["junction_min"])
    )


def _slot_lookup(validated_domains: dict[tuple[str, int], str]) -> tuple[list[dict[str, str]], list[dict[str, int]]]:
    letter_to_parent = {"A": "As", "L": "Lb", "F": "Fn", "M": "Mb2"}
    seq_lookup: list[dict[str, str]] = []
    len_lookup: list[dict[str, int]] = []
    for slot in range(1, 12):
        seq_d = {}
        len_d = {}
        for letter, parent in letter_to_parent.items():
            seq = validated_domains[(parent, slot)]
            seq_d[letter] = seq
            len_d[letter] = len(seq)
        seq_lookup.append(seq_d)
        len_lookup.append(len_d)
    return seq_lookup, len_lookup


def _seq_and_lengths_from_combo(
    combo: str, seq_lookup: list[dict[str, str]], len_lookup: list[dict[str, int]]
) -> tuple[str, list[int]]:
    seq_parts = []
    lens = []
    for i, ch in enumerate(combo):
        seq_parts.append(seq_lookup[i][ch])
        lens.append(len_lookup[i][ch])
    return "".join(seq_parts), lens


def _save_exhaustive_checkpoint(
    *,
    checkpoint_path: Path,
    next_index: int,
    total: int,
    processed: int,
    heap_rows: list[tuple[float, str, dict]],
    started_at: float,
) -> None:
    payload = {
        "next_index": int(next_index),
        "total": int(total),
        "processed": int(processed),
        "started_at": float(started_at),
        "heap": [
            {
                "student_rank_score": float(score),
                "combo_compact": combo,
                "row": row,
            }
            for score, combo, row in heap_rows
        ],
    }
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")


def _load_exhaustive_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    heap_rows = []
    for item in payload.get("heap", []):
        heap_rows.append(
            (
                float(item["student_rank_score"]),
                str(item["combo_compact"]),
                dict(item["row"]),
            )
        )
    return {
        "next_index": int(payload.get("next_index", 0)),
        "total": int(payload.get("total", TOTAL_COMBOS)),
        "processed": int(payload.get("processed", 0)),
        "started_at": float(payload.get("started_at", time.time())),
        "heap_rows": heap_rows,
    }


def _run_exhaustive_student_scan(
    *,
    run_dir: Path,
    student_scorer: StudentScorer,
    validated_domains: dict[tuple[str, int], str],
    shortlist_size: int,
    student_batch_size: int,
    progress_every: int,
    checkpoint_every_batches: int,
    resume: bool,
    w_global: float,
    w_jmean: float,
    w_jmin: float,
) -> pd.DataFrame:
    checkpoint_path = run_dir / "student_exhaustive_checkpoint.json"
    shortlist_csv = run_dir / "student_shortlist.csv"
    summary_json = run_dir / "student_exhaustive_summary.json"

    seq_lookup, len_lookup = _slot_lookup(validated_domains)
    heap_rows: list[tuple[float, str, dict]] = []
    started_at = time.time()
    next_index = 0
    processed = 0

    if resume and shortlist_csv.exists() and summary_json.exists():
        logger.info("Reuse existing exhaustive shortlist: %s", shortlist_csv)
        return pd.read_csv(shortlist_csv)

    if resume and checkpoint_path.exists():
        state = _load_exhaustive_checkpoint(checkpoint_path)
        next_index = state["next_index"]
        processed = state["processed"]
        started_at = state["started_at"]
        heap_rows = state["heap_rows"]
        logger.info(
            "Resume exhaustive scan from index=%d/%d (processed=%d, heap=%d)",
            next_index,
            TOTAL_COMBOS,
            processed,
            len(heap_rows),
        )

    n_batches = 0
    while next_index < TOTAL_COMBOS:
        end_index = min(next_index + student_batch_size, TOTAL_COMBOS)
        combos = [combo_from_index(i, slots=11, alphabet=ALPHABET) for i in range(next_index, end_index)]

        seqs = []
        lens = []
        for combo in combos:
            seq, dlen = _seq_and_lengths_from_combo(combo, seq_lookup, len_lookup)
            seqs.append(seq)
            lens.append(dlen)

        scores = student_scorer.score_sequences(
            sequences_aa=seqs,
            domain_lengths_list=lens,
            batch_size=student_batch_size,
        )

        for combo, score in zip(combos, scores):
            rank_score = (
                w_global * float(score.global_score)
                + w_jmean * float(score.junction_mean)
                + w_jmin * float(score.junction_min)
            )

            if len(heap_rows) >= shortlist_size and rank_score <= heap_rows[0][0]:
                continue

            row = {
                "combo_compact": combo,
                "sequence_hash": score.sequence_hash,
                "global_score": score.global_score,
                "junction_mean": score.junction_mean,
                "junction_min": score.junction_min,
                "student_rank_score": rank_score,
            }
            for i, v in enumerate(score.junction_scores, start=1):
                row[f"junction_{i:02d}"] = v

            item = (rank_score, combo, row)
            if len(heap_rows) < shortlist_size:
                heapq.heappush(heap_rows, item)
            else:
                heapq.heapreplace(heap_rows, item)

        processed += (end_index - next_index)
        next_index = end_index
        n_batches += 1

        if processed % progress_every == 0 or next_index >= TOTAL_COMBOS:
            elapsed = time.time() - started_at
            rate = processed / max(1.0, elapsed)
            eta = (TOTAL_COMBOS - processed) / max(1e-9, rate)
            logger.info(
                "Exhaustive scan progress: %d/%d (%.2f%%), rate=%.1f combos/s, eta=%.1f min, heap=%d",
                processed,
                TOTAL_COMBOS,
                100.0 * processed / TOTAL_COMBOS,
                rate,
                eta / 60.0,
                len(heap_rows),
            )

        if n_batches % checkpoint_every_batches == 0 or next_index >= TOTAL_COMBOS:
            _save_exhaustive_checkpoint(
                checkpoint_path=checkpoint_path,
                next_index=next_index,
                total=TOTAL_COMBOS,
                processed=processed,
                heap_rows=heap_rows,
                started_at=started_at,
            )

    # Final shortlist
    shortlist_rows = [item[2] for item in heap_rows]
    shortlist_df = pd.DataFrame(shortlist_rows)
    shortlist_df = shortlist_df.sort_values("student_rank_score", ascending=False).reset_index(drop=True)
    shortlist_df.to_csv(shortlist_csv, index=False)

    summary = {
        "mode": "exhaustive",
        "total_combos": TOTAL_COMBOS,
        "processed_combos": processed,
        "shortlist_size": int(len(shortlist_df)),
        "started_at": started_at,
        "finished_at": time.time(),
        "runtime_seconds": time.time() - started_at,
        "checkpoint_path": str(checkpoint_path),
        "shortlist_csv": str(shortlist_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Exhaustive student scan done. processed=%d", processed)
    return shortlist_df


def _run_sampled_student_scan(
    *,
    run_dir: Path,
    student_scorer: StudentScorer,
    validated_domains: dict[tuple[str, int], str],
    sample_size: int,
    shortlist_size: int,
    seed: int,
    resume: bool,
    w_global: float,
    w_jmean: float,
    w_jmin: float,
) -> pd.DataFrame:
    sample_scored_csv = run_dir / "student_sample_scored.csv"
    shortlist_csv = run_dir / "student_shortlist.csv"
    if resume and shortlist_csv.exists():
        return pd.read_csv(shortlist_csv)

    if resume and sample_scored_csv.exists():
        student_df = pd.read_csv(sample_scored_csv)
    else:
        combos = sample_combo_compacts(n=sample_size, seed=seed)
        sample_df = pd.DataFrame({"combo_compact": combos})
        student_df = student_scorer.score_batch_rows(
            rows_df=sample_df,
            validated_domains=validated_domains,
            combo_col="combo_compact",
            seq_col="sequence_aa",
        )
        student_df["student_rank_score"] = student_df.apply(
            lambda r: _student_rank_score(r.to_dict(), w_global=w_global, w_jmean=w_jmean, w_jmin=w_jmin),
            axis=1,
        )
        student_df.to_csv(sample_scored_csv, index=False)

    shortlist_df = (
        student_df.sort_values("student_rank_score", ascending=False)
        .head(shortlist_size)
        .reset_index(drop=True)
    )
    shortlist_df.to_csv(shortlist_csv, index=False)
    return shortlist_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--student-checkpoint", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--calibration-model", default=None)
    ap.add_argument("--calibration-meta", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default=None, help="Reuse existing run folder when resuming")
    ap.add_argument("--mode", choices=["sampled", "exhaustive"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--shortlist-size", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--teacher-rerank-size", type=int, default=None)
    ap.add_argument("--min-hamming", type=int, default=None)
    ap.add_argument("--student-batch-size", type=int, default=None)
    ap.add_argument("--teacher-batch-size", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=None)
    ap.add_argument("--checkpoint-every-batches", type=int, default=None)
    ap.add_argument("--device", default=None, help="student device; teacher auto-detect")
    ap.add_argument("--teacher-device", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_yaml(args.config)
    search_cfg = cfg.get("search", {})
    cal_cfg = cfg.get("calibration", {})

    mode = args.mode or str(search_cfg.get("mode", "sampled"))
    seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 13))
    sample_size = int(args.sample_size) if args.sample_size is not None else int(search_cfg.get("sample_size", 20000))
    shortlist_size = (
        int(args.shortlist_size) if args.shortlist_size is not None else int(search_cfg.get("shortlist_size", 2000))
    )
    teacher_rerank_size = (
        int(args.teacher_rerank_size)
        if args.teacher_rerank_size is not None
        else int(search_cfg.get("teacher_rerank_size", 300))
    )
    top_k = int(args.top_k) if args.top_k is not None else int(search_cfg.get("top_k", 50))
    min_hamming = int(args.min_hamming) if args.min_hamming is not None else int(search_cfg.get("min_hamming", 2))
    student_batch_size = (
        int(args.student_batch_size)
        if args.student_batch_size is not None
        else int(search_cfg.get("student_batch_size", 16))
    )
    teacher_batch_size = (
        int(args.teacher_batch_size)
        if args.teacher_batch_size is not None
        else int(search_cfg.get("teacher_batch_size", 4))
    )
    progress_every = (
        int(args.progress_every) if args.progress_every is not None else int(search_cfg.get("progress_every", 50000))
    )
    checkpoint_every_batches = (
        int(args.checkpoint_every_batches)
        if args.checkpoint_every_batches is not None
        else int(search_cfg.get("checkpoint_every_batches", 200))
    )
    weights = search_cfg.get("student_rank_weights", {})
    w_global = float(weights.get("global", 1.0))
    w_jmean = float(weights.get("junction_mean", 0.5))
    w_jmin = float(weights.get("junction_min", 0.5))

    out_dir = args.out_dir or search_cfg.get("output_dir", "cas12a_shuffling_model/outputs/ranking")
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"rank_{int(time.time())}_{seed}"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    vd_path = resolve_validated_domains_path(cfg, cli_path=args.validated_domains)
    validated_domains = load_validated_domains_dict(vd_path)

    checkpoint = _resolve_student_checkpoint(cfg, args.student_checkpoint)
    window_cfg = cfg.get("teacher", {}).get("junction_window", {})
    window = JunctionWindowConfig(left=int(window_cfg.get("left", 25)), right=int(window_cfg.get("right", 25)))

    student_scorer = StudentScorer(checkpoint_path=checkpoint, window=window, device=args.device)
    teacher_scorer = build_teacher_scorer_from_config(cfg, device=args.teacher_device)

    if args.calibration_model and args.calibration_meta:
        cal_model_path, cal_meta_path = args.calibration_model, args.calibration_meta
    else:
        cal_base = cal_cfg.get("output_dir", "cas12a_shuffling_model/outputs/calibration")
        cal_model_path, cal_meta_path = _latest_calibration_paths(cal_base)
    cal_artifact = load_calibration_artifact(cal_model_path, cal_meta_path)

    shortlist_csv = run_dir / "student_shortlist.csv"
    teacher_reranked_csv = run_dir / "teacher_reranked.csv"
    candidate_top_csv = run_dir / "candidate_top.csv"
    candidate_all_csv = run_dir / "candidate_all_scored.csv"

    if mode == "exhaustive":
        shortlist_df = _run_exhaustive_student_scan(
            run_dir=run_dir,
            student_scorer=student_scorer,
            validated_domains=validated_domains,
            shortlist_size=shortlist_size,
            student_batch_size=student_batch_size,
            progress_every=progress_every,
            checkpoint_every_batches=checkpoint_every_batches,
            resume=args.resume,
            w_global=w_global,
            w_jmean=w_jmean,
            w_jmin=w_jmin,
        )
    else:
        shortlist_df = _run_sampled_student_scan(
            run_dir=run_dir,
            student_scorer=student_scorer,
            validated_domains=validated_domains,
            sample_size=sample_size,
            shortlist_size=shortlist_size,
            seed=seed,
            resume=args.resume,
            w_global=w_global,
            w_jmean=w_jmean,
            w_jmin=w_jmin,
        )

    if args.resume and teacher_reranked_csv.exists():
        teacher_df = pd.read_csv(teacher_reranked_csv)
    else:
        rerank_df = shortlist_df.head(teacher_rerank_size).copy()
        teacher_df = score_rows_with_teacher(
            rows_df=rerank_df,
            scorer=teacher_scorer,
            validated_domains=validated_domains,
            combo_col="combo_compact",
            seq_col="sequence_aa",
            batch_size=teacher_batch_size,
        )
        teacher_df.to_csv(teacher_reranked_csv, index=False)

    cal_df = apply_calibration(teacher_df, cal_artifact)
    cal_df["source_run_id"] = run_id
    cal_df["seed"] = seed
    cal_df["diversity_selected"] = False

    selected = greedy_diversity_select(
        cal_df,
        top_k=top_k,
        score_col="calibrated_prob",
        combo_col="combo_compact",
        min_hamming=min_hamming,
    )
    selected["diversity_selected"] = True
    selected = selected.sort_values("calibrated_prob", ascending=False).reset_index(drop=True)
    selected["final_rank"] = range(1, len(selected) + 1)

    selected_keys = set(selected["combo_compact"].astype(str).tolist())
    cal_df["diversity_selected"] = cal_df["combo_compact"].astype(str).isin(selected_keys)

    ordered_cols = [
        "final_rank",
        "combo_compact",
        "sequence_aa",
        "sequence_hash",
        "global_score",
        "junction_mean",
        "junction_min",
        "calibrated_score",
        "calibrated_prob",
        "passes_s_min",
        "diversity_selected",
        "source_run_id",
        "seed",
    ] + [f"junction_{i:02d}" for i in range(1, 11)]
    for c in ordered_cols:
        if c not in selected.columns:
            selected[c] = ""
        if c not in cal_df.columns:
            cal_df[c] = ""

    selected = selected[ordered_cols]
    cal_df = cal_df[ordered_cols]
    selected.to_csv(candidate_top_csv, index=False)
    cal_df.to_csv(candidate_all_csv, index=False)

    run_meta = {
        "run_id": run_id,
        "mode": mode,
        "seed": seed,
        "sample_size": sample_size if mode == "sampled" else None,
        "total_combos": TOTAL_COMBOS if mode == "exhaustive" else None,
        "shortlist_size": shortlist_size,
        "teacher_rerank_size": teacher_rerank_size,
        "top_k": top_k,
        "min_hamming": min_hamming,
        "student_batch_size": student_batch_size,
        "teacher_batch_size": teacher_batch_size,
        "progress_every": progress_every,
        "checkpoint_every_batches": checkpoint_every_batches,
        "student_rank_weights": {
            "global": w_global,
            "junction_mean": w_jmean,
            "junction_min": w_jmin,
        },
        "student_checkpoint": checkpoint,
        "calibration_model": cal_model_path,
        "calibration_meta": cal_meta_path,
        "outputs": {
            "student_shortlist": str(shortlist_csv),
            "teacher_reranked": str(teacher_reranked_csv),
            "candidate_top": str(candidate_top_csv),
            "candidate_all_scored": str(candidate_all_csv),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    logger.info("Ranking completed. run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
