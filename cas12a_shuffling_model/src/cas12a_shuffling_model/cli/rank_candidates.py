from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from cas12a_shuffling_model.calibration.calibrator import apply_calibration, load_calibration_artifact
from cas12a_shuffling_model.io.loaders import load_yaml
from cas12a_shuffling_model.search.combo_compact import build_sequence_from_combo, validate_combo_compact
from cas12a_shuffling_model.search.rank_pipeline import greedy_diversity_select
from cas12a_shuffling_model.search.sampler import combo_from_index, sample_combo_compacts
from cas12a_shuffling_model.student.score_student import StudentScorer
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
    score_rows_with_teacher,
    with_teacher_overrides,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

TOTAL_COMBOS = 4**11
ALPHABET = ("A", "L", "F", "M")
UNION_METRICS_DEFAULT = ("student_rank_score", "global_score", "junction_mean", "junction_min")


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


def _heap_push_topk(
    heap_rows: list[tuple[float, str, dict]],
    item: tuple[float, str, dict],
    *,
    max_size: int,
) -> None:
    if len(heap_rows) < max_size:
        heapq.heappush(heap_rows, item)
        return
    if item[0] > heap_rows[0][0]:
        heapq.heapreplace(heap_rows, item)


def _build_union_shortlist_from_metric_heaps(
    metric_heaps: dict[str, list[tuple[float, str, dict]]],
) -> pd.DataFrame:
    rows_by_combo: dict[str, dict[str, Any]] = {}
    sources_by_combo: dict[str, set[str]] = {}
    for metric_name, heap_rows in metric_heaps.items():
        for _, combo, row in heap_rows:
            if combo not in rows_by_combo:
                rows_by_combo[combo] = dict(row)
                sources_by_combo[combo] = set()
            sources_by_combo[combo].add(metric_name)

    out_rows: list[dict[str, Any]] = []
    for combo, row in rows_by_combo.items():
        rec = dict(row)
        rec["shortlist_sources"] = ",".join(sorted(sources_by_combo.get(combo, set())))
        out_rows.append(rec)
    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows)
    sort_cols = [c for c in ("student_rank_score", "global_score", "junction_mean", "junction_min") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return df.reset_index(drop=True)


def _select_rerank_candidates_union(
    *,
    shortlist_df: pd.DataFrame,
    teacher_rerank_size: int,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if len(shortlist_df) <= teacher_rerank_size:
        return shortlist_df.copy().reset_index(drop=True)

    combo_col = "combo_compact"
    available_metrics = [m for m in metrics if m in shortlist_df.columns]
    if not available_metrics:
        return shortlist_df.head(teacher_rerank_size).copy().reset_index(drop=True)

    metric_lists: list[list[str]] = []
    for metric in available_metrics:
        metric_vals = pd.to_numeric(shortlist_df[metric], errors="coerce")
        metric_df = shortlist_df[metric_vals.notna()].copy()
        metric_df = metric_df.sort_values(metric, ascending=False)
        metric_lists.append(metric_df[combo_col].astype(str).tolist())

    selected: list[str] = []
    selected_set: set[str] = set()
    pointers = [0] * len(metric_lists)

    while len(selected) < teacher_rerank_size:
        progressed = False
        for idx, combos in enumerate(metric_lists):
            while pointers[idx] < len(combos):
                combo = combos[pointers[idx]]
                pointers[idx] += 1
                if combo in selected_set:
                    continue
                selected.append(combo)
                selected_set.add(combo)
                progressed = True
                break
            if len(selected) >= teacher_rerank_size:
                break
        if not progressed:
            break

    if len(selected) < teacher_rerank_size:
        fallback = shortlist_df.sort_values("student_rank_score", ascending=False)
        for combo in fallback[combo_col].astype(str).tolist():
            if combo in selected_set:
                continue
            selected.append(combo)
            selected_set.add(combo)
            if len(selected) >= teacher_rerank_size:
                break

    order_map = {combo: i for i, combo in enumerate(selected)}
    out = shortlist_df[shortlist_df[combo_col].astype(str).isin(selected_set)].copy()
    out["__rr_order"] = out[combo_col].astype(str).map(order_map)
    out = out.sort_values("__rr_order").drop(columns=["__rr_order"])
    return out.head(teacher_rerank_size).reset_index(drop=True)


def _select_rerank_candidates_hybrid(
    *,
    shortlist_df: pd.DataFrame,
    teacher_rerank_size: int,
    exploit_ratio: float,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if len(shortlist_df) <= teacher_rerank_size:
        return shortlist_df.copy().reset_index(drop=True)

    ratio = float(exploit_ratio)
    ratio = min(1.0, max(0.0, ratio))
    exploit_n = int(round(teacher_rerank_size * ratio))
    exploit_n = max(1, min(teacher_rerank_size, exploit_n))

    top_exploit = shortlist_df.sort_values("student_rank_score", ascending=False).head(exploit_n).copy()
    if len(top_exploit) >= teacher_rerank_size:
        return top_exploit.head(teacher_rerank_size).reset_index(drop=True)

    chosen = set(top_exploit["combo_compact"].astype(str).tolist())
    remain = shortlist_df[~shortlist_df["combo_compact"].astype(str).isin(chosen)].copy()
    explore_n = teacher_rerank_size - len(top_exploit)
    explore = _select_rerank_candidates_union(
        shortlist_df=remain,
        teacher_rerank_size=explore_n,
        metrics=metrics,
    )
    out = pd.concat([top_exploit, explore], ignore_index=True)
    out = out.drop_duplicates(subset=["combo_compact"]).head(teacher_rerank_size)
    return out.reset_index(drop=True)


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


def _resolve_active_probe_csv(cfg: dict, cli_path: str | None = None) -> str | None:
    if cli_path:
        return str(cli_path)
    search_cfg = cfg.get("search", {})
    configured = search_cfg.get("active_probe_csv")
    if configured:
        return str(configured)
    out_active = cfg.get("paths", {}).get("out_active_dir")
    if out_active:
        p = Path(out_active)
        if p.is_dir():
            return str(p / "active_chimeras_reconstructed.csv")
        return str(p)
    return None


def _load_active_probe_rows(
    *,
    active_csv: str,
    validated_domains: dict[tuple[str, int], str],
) -> pd.DataFrame:
    if not active_csv or not Path(active_csv).exists():
        return pd.DataFrame(columns=["combo_compact", "sequence_aa"])
    df = pd.read_csv(active_csv)
    if len(df) == 0:
        return pd.DataFrame(columns=["combo_compact", "sequence_aa"])

    combo_vals: list[str] = []
    if "combo_compact" in df.columns:
        for v in df["combo_compact"].astype(str).tolist():
            vv = v.strip().upper()
            if not vv or vv == "NAN":
                continue
            combo_vals.append(validate_combo_compact(vv))
    else:
        slot_names = [str(i) for i in range(1, 12)]
        slot_cols = [c for c in df.columns if str(c).strip() in slot_names]
        if len(slot_cols) < 11:
            raise ValueError(
                f"Active probe CSV missing combo_compact or 11 slot columns: {active_csv}"
            )
        slot_cols = sorted(slot_cols, key=lambda x: int(str(x).strip()))
        for _, row in df.iterrows():
            combo = "".join(str(row[c]).strip().upper() for c in slot_cols[:11])
            combo_vals.append(validate_combo_compact(combo))

    out = pd.DataFrame({"combo_compact": combo_vals})
    if "sequence_aa" in df.columns:
        seq_map: dict[str, str] = {}
        for combo, seq in zip(
            df.get("combo_compact", pd.Series([""] * len(df))).astype(str).tolist(),
            df["sequence_aa"].astype(str).tolist(),
        ):
            combo_raw = combo.strip().upper()
            if not combo_raw or combo_raw == "NAN":
                continue
            try:
                combo_norm = validate_combo_compact(combo_raw)
            except ValueError:
                continue
            seq_map[combo_norm] = seq.strip().upper()
        out["sequence_aa"] = out["combo_compact"].map(seq_map).fillna("")
    else:
        out["sequence_aa"] = ""

    out = out.drop_duplicates(subset=["combo_compact"]).reset_index(drop=True)
    out["sequence_aa"] = out.apply(
        lambda r: r["sequence_aa"]
        if str(r["sequence_aa"]).strip()
        else build_sequence_from_combo(str(r["combo_compact"]), validated_domains),
        axis=1,
    )
    return out


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
    union_metric_heaps: dict[str, list[tuple[float, str, dict]]] | None = None,
) -> None:
    union_serialized: dict[str, list[dict[str, Any]]] = {}
    for metric_name, items in (union_metric_heaps or {}).items():
        union_serialized[str(metric_name)] = [
            {
                "score": float(score),
                "combo_compact": combo,
                "row": row,
            }
            for score, combo, row in items
        ]
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
        "union_metric_heaps": union_serialized,
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
    union_metric_heaps: dict[str, list[tuple[float, str, dict]]] = {}
    for metric_name, items in dict(payload.get("union_metric_heaps", {})).items():
        metric_rows: list[tuple[float, str, dict]] = []
        for item in items:
            metric_rows.append(
                (
                    float(item["score"]),
                    str(item["combo_compact"]),
                    dict(item["row"]),
                )
            )
        union_metric_heaps[str(metric_name)] = metric_rows
    return {
        "next_index": int(payload.get("next_index", 0)),
        "total": int(payload.get("total", TOTAL_COMBOS)),
        "processed": int(payload.get("processed", 0)),
        "started_at": float(payload.get("started_at", time.time())),
        "heap_rows": heap_rows,
        "union_metric_heaps": union_metric_heaps,
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
    shortlist_strategy: str = "weighted",
    union_metrics: Sequence[str] = UNION_METRICS_DEFAULT,
    union_per_metric_size: int | None = None,
) -> pd.DataFrame:
    checkpoint_path = run_dir / "student_exhaustive_checkpoint.json"
    shortlist_csv = run_dir / "student_shortlist.csv"
    summary_json = run_dir / "student_exhaustive_summary.json"

    seq_lookup, len_lookup = _slot_lookup(validated_domains)
    heap_rows: list[tuple[float, str, dict]] = []
    union_metric_heaps: dict[str, list[tuple[float, str, dict]]] = {
        m: [] for m in union_metrics
    }
    started_at = time.time()
    next_index = 0
    processed = 0
    per_metric_keep = int(union_per_metric_size or shortlist_size)
    per_metric_keep = max(1, per_metric_keep)
    shortlist_strategy = str(shortlist_strategy).strip().lower()

    if resume and shortlist_csv.exists() and summary_json.exists():
        logger.info("Reuse existing exhaustive shortlist: %s", shortlist_csv)
        return pd.read_csv(shortlist_csv)

    if resume and checkpoint_path.exists():
        state = _load_exhaustive_checkpoint(checkpoint_path)
        next_index = state["next_index"]
        processed = state["processed"]
        started_at = state["started_at"]
        heap_rows = state["heap_rows"]
        loaded_union = state.get("union_metric_heaps", {})
        for metric_name in union_metric_heaps.keys():
            if metric_name in loaded_union:
                union_metric_heaps[metric_name] = loaded_union[metric_name]
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
            if shortlist_strategy == "union":
                _heap_push_topk(heap_rows, item, max_size=shortlist_size)
                for metric_name in union_metric_heaps.keys():
                    metric_val = row.get(metric_name)
                    if metric_val is None:
                        continue
                    try:
                        metric_score = float(metric_val)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(metric_score):
                        continue
                    _heap_push_topk(
                        union_metric_heaps[metric_name],
                        (metric_score, combo, row),
                        max_size=per_metric_keep,
                    )
            else:
                _heap_push_topk(heap_rows, item, max_size=shortlist_size)

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
                union_metric_heaps=union_metric_heaps if shortlist_strategy == "union" else None,
            )

    # Final shortlist
    if shortlist_strategy == "union":
        union_df = _build_union_shortlist_from_metric_heaps(union_metric_heaps)
        if len(union_df) == 0:
            shortlist_rows = [item[2] for item in heap_rows]
            shortlist_df = pd.DataFrame(shortlist_rows)
        else:
            shortlist_df = union_df
            logger.info(
                "Union shortlist built from metrics=%s; rows=%d",
                list(union_metric_heaps.keys()),
                len(shortlist_df),
            )
    else:
        shortlist_rows = [item[2] for item in heap_rows]
        shortlist_df = pd.DataFrame(shortlist_rows)
    shortlist_df = shortlist_df.sort_values("student_rank_score", ascending=False).reset_index(drop=True)
    shortlist_df.to_csv(shortlist_csv, index=False)

    summary = {
        "mode": "exhaustive",
        "total_combos": TOTAL_COMBOS,
        "processed_combos": processed,
        "shortlist_size": int(len(shortlist_df)),
        "shortlist_strategy": shortlist_strategy,
        "union_metrics": list(union_metric_heaps.keys()) if shortlist_strategy == "union" else [],
        "union_per_metric_size": per_metric_keep if shortlist_strategy == "union" else None,
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
    ap.add_argument("--shortlist-strategy", choices=["weighted", "union"], default=None)
    ap.add_argument("--union-per-metric-size", type=int, default=None)
    ap.add_argument("--rerank-selection", choices=["weighted", "union", "hybrid"], default=None)
    ap.add_argument("--rerank-exploit-ratio", type=float, default=None)
    ap.add_argument("--active-probe-csv", default=None)
    ap.add_argument("--disable-active-probe", action="store_true")
    ap.add_argument("--device", default=None, help="student device; teacher auto-detect")
    ap.add_argument("--teacher-device", default=None)
    ap.add_argument("--teacher-model-name-or-path", default=None)
    ap.add_argument("--teacher-model-source", choices=["hf", "local"], default=None)
    ap.add_argument("--teacher-adapter-path", default=None)
    ap.add_argument("--teacher-model-revision", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
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
    shortlist_strategy = str(
        args.shortlist_strategy if args.shortlist_strategy is not None else search_cfg.get("shortlist_strategy", "weighted")
    ).strip().lower()
    union_per_metric_size = (
        int(args.union_per_metric_size)
        if args.union_per_metric_size is not None
        else int(search_cfg.get("union_per_metric_size", shortlist_size))
    )
    rerank_selection = str(
        args.rerank_selection if args.rerank_selection is not None else search_cfg.get("rerank_selection", shortlist_strategy)
    ).strip().lower()
    rerank_exploit_ratio = (
        float(args.rerank_exploit_ratio)
        if args.rerank_exploit_ratio is not None
        else float(search_cfg.get("rerank_exploit_ratio", 0.7))
    )
    active_probe_enabled = (
        not bool(args.disable_active_probe)
        and bool(search_cfg.get("active_probe", True))
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
    try:
        shutil.copy2(cal_model_path, run_dir / "calibration_model.joblib")
        shutil.copy2(cal_meta_path, run_dir / "calibration_meta.json")
    except OSError as e:
        logger.warning("Failed to copy calibration artifacts into run dir: %s", e)

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
            shortlist_strategy=shortlist_strategy,
            union_per_metric_size=union_per_metric_size,
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

    rerank_df: pd.DataFrame
    if args.resume and teacher_reranked_csv.exists():
        teacher_df = pd.read_csv(teacher_reranked_csv)
        rerank_df = teacher_df.copy()
    else:
        if rerank_selection == "union":
            rerank_df = _select_rerank_candidates_union(
                shortlist_df=shortlist_df,
                teacher_rerank_size=teacher_rerank_size,
                metrics=UNION_METRICS_DEFAULT,
            )
        elif rerank_selection == "hybrid":
            rerank_df = _select_rerank_candidates_hybrid(
                shortlist_df=shortlist_df,
                teacher_rerank_size=teacher_rerank_size,
                exploit_ratio=rerank_exploit_ratio,
                metrics=UNION_METRICS_DEFAULT,
            )
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

    probe_summary: dict[str, Any] | None = None
    if active_probe_enabled:
        active_probe_csv = _resolve_active_probe_csv(cfg, cli_path=args.active_probe_csv)
        if active_probe_csv and Path(active_probe_csv).exists():
            try:
                probe_rows = _load_active_probe_rows(
                    active_csv=active_probe_csv,
                    validated_domains=validated_domains,
                )
            except Exception as e:
                logger.warning("Failed to load active probe rows: %s", e)
                probe_rows = pd.DataFrame(columns=["combo_compact", "sequence_aa"])
        else:
            logger.warning("Active probe CSV not found; skip active probe diagnostics. path=%s", active_probe_csv)
            probe_rows = pd.DataFrame(columns=["combo_compact", "sequence_aa"])

        if len(probe_rows) > 0:
            shortlist_lookup = shortlist_df.set_index("combo_compact")
            probe_student_rows: list[dict[str, Any]] = []
            missing_rows: list[dict[str, Any]] = []
            for _, row in probe_rows.iterrows():
                combo = str(row["combo_compact"])
                seq = str(row.get("sequence_aa", "")).strip().upper()
                if combo in shortlist_lookup.index:
                    src = shortlist_lookup.loc[combo]
                    if isinstance(src, pd.DataFrame):
                        src = src.iloc[0]
                    rec = {
                        "combo_compact": combo,
                        "sequence_aa": seq or str(src.get("sequence_aa", "")),
                        "sequence_hash": src.get("sequence_hash", ""),
                        "global_score": src.get("global_score", float("nan")),
                        "junction_mean": src.get("junction_mean", float("nan")),
                        "junction_min": src.get("junction_min", float("nan")),
                        "student_rank_score": src.get("student_rank_score", float("nan")),
                        "in_student_shortlist": True,
                    }
                    probe_student_rows.append(rec)
                else:
                    missing_rows.append({"combo_compact": combo, "sequence_aa": seq})

            if missing_rows:
                missing_df = pd.DataFrame(missing_rows)
                scored_missing = student_scorer.score_batch_rows(
                    rows_df=missing_df,
                    validated_domains=validated_domains,
                    combo_col="combo_compact",
                    seq_col="sequence_aa",
                    batch_size=student_batch_size,
                )
                scored_missing["student_rank_score"] = scored_missing.apply(
                    lambda r: _student_rank_score(
                        r.to_dict(), w_global=w_global, w_jmean=w_jmean, w_jmin=w_jmin
                    ),
                    axis=1,
                )
                for _, r in scored_missing.iterrows():
                    probe_student_rows.append(
                        {
                            "combo_compact": str(r["combo_compact"]),
                            "sequence_aa": str(r.get("sequence_aa", "")),
                            "sequence_hash": str(r.get("sequence_hash", "")),
                            "global_score": float(r.get("global_score", float("nan"))),
                            "junction_mean": float(r.get("junction_mean", float("nan"))),
                            "junction_min": float(r.get("junction_min", float("nan"))),
                            "student_rank_score": float(r.get("student_rank_score", float("nan"))),
                            "in_student_shortlist": False,
                        }
                    )

            probe_student_df = pd.DataFrame(probe_student_rows)
            shortlist_rank_map = {
                str(c): i + 1 for i, c in enumerate(shortlist_df["combo_compact"].astype(str).tolist())
            }
            rerank_combo_set = set(teacher_df["combo_compact"].astype(str).tolist())
            probe_student_df["student_shortlist_rank"] = probe_student_df["combo_compact"].map(shortlist_rank_map)
            probe_student_df["in_teacher_rerank"] = probe_student_df["combo_compact"].astype(str).isin(rerank_combo_set)
            probe_student_df["in_final_top"] = probe_student_df["combo_compact"].astype(str).isin(selected_keys)
            probe_student_df.to_csv(run_dir / "active_probe_student_rank.csv", index=False)

            probe_teacher_df = score_rows_with_teacher(
                rows_df=probe_student_df[["combo_compact", "sequence_aa"]],
                scorer=teacher_scorer,
                validated_domains=validated_domains,
                combo_col="combo_compact",
                seq_col="sequence_aa",
                batch_size=teacher_batch_size,
            )
            probe_teacher_df = probe_teacher_df.merge(
                probe_student_df[
                    [
                        "combo_compact",
                        "student_rank_score",
                        "student_shortlist_rank",
                        "in_student_shortlist",
                        "in_teacher_rerank",
                        "in_final_top",
                    ]
                ],
                on="combo_compact",
                how="left",
            )
            probe_teacher_df.to_csv(run_dir / "active_probe_teacher_score.csv", index=False)

            probe_cal_df = apply_calibration(probe_teacher_df, cal_artifact)
            rerank_global_rank = {
                str(c): i + 1
                for i, c in enumerate(
                    teacher_df.sort_values("global_score", ascending=False)["combo_compact"].astype(str).tolist()
                )
            }
            rerank_cal_rank = {
                str(c): i + 1
                for i, c in enumerate(
                    cal_df.sort_values("calibrated_prob", ascending=False)["combo_compact"].astype(str).tolist()
                )
            }
            probe_cal_df["teacher_global_rank_in_rerank"] = probe_cal_df["combo_compact"].astype(str).map(rerank_global_rank)
            probe_cal_df["teacher_cal_rank_in_rerank"] = probe_cal_df["combo_compact"].astype(str).map(rerank_cal_rank)
            probe_cal_df.to_csv(run_dir / "active_probe_calibrated.csv", index=False)

            probe_summary = {
                "active_probe_csv": str(active_probe_csv),
                "n_probe": int(len(probe_cal_df)),
                "n_in_student_shortlist": int(probe_cal_df["in_student_shortlist"].astype(bool).sum()),
                "n_in_teacher_rerank": int(probe_cal_df["in_teacher_rerank"].astype(bool).sum()),
                "n_in_final_top": int(probe_cal_df["in_final_top"].astype(bool).sum()),
                "mean_calibrated_prob": float(probe_cal_df["calibrated_prob"].mean()),
                "median_calibrated_prob": float(probe_cal_df["calibrated_prob"].median()),
            }
            (run_dir / "active_probe_summary.json").write_text(
                json.dumps(probe_summary, indent=2), encoding="utf-8"
            )

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
        "shortlist_strategy": shortlist_strategy,
        "union_per_metric_size": union_per_metric_size if shortlist_strategy == "union" else None,
        "rerank_selection": rerank_selection,
        "rerank_exploit_ratio": rerank_exploit_ratio if rerank_selection == "hybrid" else None,
        "student_rank_weights": {
            "global": w_global,
            "junction_mean": w_jmean,
            "junction_min": w_jmin,
        },
        "student_checkpoint": checkpoint,
        "calibration_model": cal_model_path,
        "calibration_meta": cal_meta_path,
        "teacher": {
            "model_name_or_path": cfg.get("teacher", {}).get("model_name_or_path")
            or cfg.get("teacher", {}).get("model_name"),
            "model_source": cfg.get("teacher", {}).get("model_source", "hf"),
            "model_revision": cfg.get("teacher", {}).get("model_revision"),
            "adapter_path": cfg.get("teacher", {}).get("adapter_path"),
        },
        "outputs": {
            "student_shortlist": str(shortlist_csv),
            "teacher_reranked": str(teacher_reranked_csv),
            "candidate_top": str(candidate_top_csv),
            "candidate_all_scored": str(candidate_all_csv),
        },
        "active_probe": probe_summary,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    logger.info("Ranking completed. run_dir=%s", run_dir)


if __name__ == "__main__":
    main()
