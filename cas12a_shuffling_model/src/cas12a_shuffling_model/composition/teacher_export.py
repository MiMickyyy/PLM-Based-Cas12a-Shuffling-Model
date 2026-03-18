from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.chimera_repr import (
    canonicalize_chimera_table,
    load_active_code_counts,
    sample_slot_codes,
    slot_code_to_int_array,
    slot_columns,
)
from cas12a_shuffling_model.composition.table_io import read_table, write_table
from cas12a_shuffling_model.io.loaders import sha256_text
from cas12a_shuffling_model.search.combo_compact import domain_lengths_from_combo
from cas12a_shuffling_model.teacher.protgpt2_scorer import ProtGPT2Scorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherExportConfig:
    n_samples: int = 10000
    seed: int = 13
    score_batch_size: int = 8
    normalize_length_bin_size: int = 50
    normalize_min_group_size: int = 64
    resume: bool = True
    include_junction_features: bool = True
    include_active_rows: bool = True


def robust_normalize_teacher_scores(
    df: pd.DataFrame,
    *,
    raw_col: str = "teacher_seq_score_raw",
    length_col: str = "length",
    bin_size: int = 50,
    min_group_size: int = 64,
) -> pd.Series:
    if len(df) == 0:
        return pd.Series([], dtype=float)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    lengths = pd.to_numeric(df[length_col], errors="coerce")
    if raw.notna().sum() == 0:
        return out

    finite_raw = raw[np.isfinite(raw.to_numpy())]
    global_med = float(finite_raw.median())
    q1 = float(finite_raw.quantile(0.25))
    q3 = float(finite_raw.quantile(0.75))
    global_scale = (q3 - q1) / 1.349 if (q3 - q1) > 1e-8 else float(finite_raw.std())
    if not np.isfinite(global_scale) or global_scale <= 1e-8:
        global_scale = 1.0

    if lengths.notna().sum() == 0 or (lengths.max() - lengths.min()) < max(1, int(bin_size // 2)):
        return (raw - global_med) / global_scale

    bins = (lengths // max(1, int(bin_size))).astype("Int64") * max(1, int(bin_size))
    for b in sorted(bins.dropna().unique().tolist()):
        mask = bins == b
        vals = raw[mask]
        vals = vals[np.isfinite(vals.to_numpy())]
        if len(vals) >= int(min_group_size):
            med = float(vals.median())
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            scale = (q3 - q1) / 1.349 if (q3 - q1) > 1e-8 else float(vals.std())
        else:
            med, scale = global_med, global_scale
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = global_scale
        out.loc[mask] = (raw.loc[mask] - med) / scale
    out.loc[out.isna()] = (raw.loc[out.isna()] - global_med) / global_scale
    return out


def _sample_input_table(
    *,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    codes = sample_slot_codes(n=n_samples, seed=seed)
    return pd.DataFrame({"slot_code_11": codes})


def _domain_lengths_for_rows(
    canonical_df: pd.DataFrame,
    validated_domains: dict[tuple[str, int], str] | None,
) -> list[list[int] | None]:
    if validated_domains is None:
        return [None] * len(canonical_df)
    out: list[list[int]] = []
    for code in canonical_df["slot_code_11"].astype(str).tolist():
        out.append(domain_lengths_from_combo(code, validated_domains))
    return out


def _teacher_rows_from_scores(
    canonical_df: pd.DataFrame,
    scores,
    *,
    include_junction_features: bool,
) -> pd.DataFrame:
    rows = []
    for rec, score in zip(canonical_df.to_dict(orient="records"), scores):
        row = dict(rec)
        seq = str(row["full_protein_sequence"])
        row["sequence_hash"] = score.seq_hash or sha256_text(seq)
        row["teacher_seq_score_raw"] = float(score.global_score)
        row["teacher_cache_hit"] = bool(score.from_cache)
        if include_junction_features:
            row["teacher_junction_mean"] = float(score.junction_mean)
            row["teacher_junction_min"] = float(score.junction_min)
            for i, v in enumerate(score.junction_scores, start=1):
                row[f"teacher_junction_{i:02d}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def export_teacher_labels(
    *,
    scorer: ProtGPT2Scorer,
    out_table: str,
    validated_domains: dict[tuple[str, int], str] | None,
    cfg: TeacherExportConfig,
    input_table: str | None = None,
    active_table: str | None = None,
    slot_code_col: str = "slot_code_11",
    sequence_col: str = "sequence_aa",
) -> str:
    if input_table:
        src_df = read_table(input_table)
    else:
        src_df = _sample_input_table(n_samples=int(cfg.n_samples), seed=int(cfg.seed))

    active_counts: dict[str, int] = {}
    if bool(cfg.include_active_rows) and active_table:
        try:
            active_counts = load_active_code_counts(active_table)
        except FileNotFoundError:
            logger.warning("Active table not found for teacher export: %s", active_table)
            active_counts = {}
        if len(active_counts) > 0:
            active_df = pd.DataFrame(
                {
                    "slot_code_11": list(active_counts.keys()),
                    "is_active": 1,
                    "active_repeat_count": [int(active_counts[k]) for k in active_counts.keys()],
                    "source_type": "active",
                }
            )
            src_df = pd.concat([src_df, active_df], ignore_index=True)
            logger.info(
                "Teacher export appended active combos: unique=%d table=%s",
                len(active_counts),
                active_table,
            )
        else:
            logger.warning("Active table provided but no valid 11-slot combos parsed: %s", active_table)

    canonical = canonicalize_chimera_table(
        src_df,
        validated_domains=validated_domains,
        slot_code_col=slot_code_col,
        sequence_col=sequence_col,
    )
    if "source_type" not in canonical.columns:
        canonical["source_type"] = "chimera"
    else:
        src = canonical["source_type"].astype("string")
        canonical["source_type"] = src.fillna("chimera")
    canonical = canonical.sort_values("slot_code_11").reset_index(drop=True)
    canonical["sequence_hash"] = canonical["full_protein_sequence"].map(sha256_text)

    existing = pd.DataFrame()
    out_path = Path(out_table)
    if bool(cfg.resume) and out_path.exists():
        existing = read_table(out_path)
        if "sequence_hash" in existing.columns:
            seen = set(existing["sequence_hash"].astype(str).tolist())
            missing = canonical[~canonical["sequence_hash"].isin(seen)].copy()
            logger.info(
                "Teacher export resume: existing=%d missing=%d",
                len(existing),
                len(missing),
            )
        else:
            missing = canonical.copy()
    else:
        missing = canonical.copy()

    scored_new = pd.DataFrame()
    if len(missing) > 0:
        dlen = _domain_lengths_for_rows(missing, validated_domains)
        seqs = missing["full_protein_sequence"].astype(str).tolist()
        scores = scorer.score_many(
            seqs_aa=seqs,
            domain_lengths_list=dlen,
            batch_size=int(cfg.score_batch_size),
        )
        scored_new = _teacher_rows_from_scores(
            missing,
            scores,
            include_junction_features=bool(cfg.include_junction_features),
        )

    merged = pd.concat([existing, scored_new], ignore_index=True) if len(existing) > 0 else scored_new
    if len(merged) == 0:
        merged = canonical.copy()
        merged["teacher_seq_score_raw"] = np.nan
        merged["teacher_cache_hit"] = False
    merged = merged.drop_duplicates(subset=["slot_code_11"], keep="last").reset_index(drop=True)
    merged["teacher_seq_score_norm"] = robust_normalize_teacher_scores(
        merged,
        raw_col="teacher_seq_score_raw",
        length_col="length",
        bin_size=int(cfg.normalize_length_bin_size),
        min_group_size=int(cfg.normalize_min_group_size),
    )
    if len(active_counts) > 0:
        merged["active_repeat_count"] = merged["slot_code_11"].map(lambda c: int(active_counts.get(str(c), 0)))
        merged["is_active"] = (merged["active_repeat_count"] > 0).astype(int)
    else:
        if "active_repeat_count" not in merged.columns:
            merged["active_repeat_count"] = 0
        if "is_active" not in merged.columns:
            merged["is_active"] = 0

    for i, col in enumerate(slot_columns(), start=1):
        if col not in merged.columns:
            merged[col] = merged["slot_code_11"].map(lambda s: int(slot_code_to_int_array(str(s))[i - 1]))

    ordered_prefix = [
        "chimera_id",
        "slot_code_11",
        "combo_compact",
        *slot_columns(),
        "full_protein_sequence",
        "length",
        "sequence_hash",
        "teacher_seq_score_raw",
        "teacher_seq_score_norm",
        "teacher_cache_hit",
        "is_active",
        "active_repeat_count",
    ]
    if "teacher_junction_mean" in merged.columns:
        ordered_prefix.extend(
            ["teacher_junction_mean", "teacher_junction_min", *[f"teacher_junction_{i:02d}" for i in range(1, 11)]]
        )
    head = [c for c in ordered_prefix if c in merged.columns]
    tail = [c for c in merged.columns if c not in head]
    merged = merged[head + tail]
    saved = write_table(merged.sort_values("slot_code_11").reset_index(drop=True), out_table)
    return saved
