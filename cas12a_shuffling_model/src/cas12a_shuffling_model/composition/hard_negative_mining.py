from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.active_prior import (
    ensure_active_codes,
    min_hamming_distance,
)
from cas12a_shuffling_model.composition.chimera_repr import canonicalize_chimera_table
from cas12a_shuffling_model.composition.table_io import read_table, write_table


@dataclass(frozen=True)
class HardNegativeMiningConfig:
    score_col: str = "assistant_score"
    top_pool_size: int = 100000
    max_negatives: int = 20000
    min_distance: int = 1
    max_distance: int = 4
    active_score_quantile: float = 0.50
    include_missing_from_base: bool = True
    local_target_col: str = "active_local_target"
    hard_negative_col: str = "is_hard_negative"
    distance_col: str = "local_active_distance"
    active_col: str = "is_active"


def _pick_hard_negatives(
    *,
    rerank_df: pd.DataFrame,
    active_codes: Sequence[str],
    cfg: HardNegativeMiningConfig,
) -> pd.DataFrame:
    work = canonicalize_chimera_table(rerank_df, require_sequence=False)
    if cfg.score_col not in work.columns:
        raise KeyError(f"Missing score column in rerank table: {cfg.score_col}")
    work["slot_code_11"] = work["slot_code_11"].astype(str)
    work["_score"] = pd.to_numeric(work[cfg.score_col], errors="coerce")
    work = work[np.isfinite(work["_score"].to_numpy())].copy()
    work = work.sort_values("_score", ascending=False).head(int(cfg.top_pool_size)).copy()
    if len(work) == 0:
        return work

    active_set = set(ensure_active_codes(active_codes))
    work[cfg.active_col] = work["slot_code_11"].map(lambda c: 1 if c in active_set else 0).astype(int)
    work[cfg.distance_col] = min_hamming_distance(work["slot_code_11"].tolist(), list(active_set)).astype(int)
    eligible = work[
        (work[cfg.active_col] == 0)
        & (work[cfg.distance_col] >= int(cfg.min_distance))
        & (work[cfg.distance_col] <= int(cfg.max_distance))
    ].copy()
    if len(eligible) == 0:
        return eligible

    active_scores = work.loc[work[cfg.active_col] == 1, "_score"].to_numpy(dtype=np.float64)
    if active_scores.size > 0:
        thr = float(np.quantile(active_scores, float(cfg.active_score_quantile)))
        hard = eligible[eligible["_score"] >= thr].copy()
    else:
        hard = eligible.copy()
    if len(hard) == 0:
        hard = eligible.sort_values("_score", ascending=False).head(min(len(eligible), int(cfg.max_negatives))).copy()
    else:
        hard = hard.sort_values("_score", ascending=False).head(int(cfg.max_negatives)).copy()

    hard[cfg.hard_negative_col] = 1
    hard["hard_negative_reason"] = "active_local_basin"
    hard[cfg.local_target_col] = pd.to_numeric(hard["_score"], errors="coerce")
    return hard


def build_active_local_training_table(
    *,
    base_table: str,
    rerank_table: str,
    active_codes: Sequence[str],
    out_table: str,
    out_hard_negatives: str | None = None,
    cfg: HardNegativeMiningConfig = HardNegativeMiningConfig(),
) -> dict[str, object]:
    base_df = canonicalize_chimera_table(read_table(base_table), require_sequence=False)
    rerank_df = canonicalize_chimera_table(read_table(rerank_table), require_sequence=False)
    active_set = set(ensure_active_codes(active_codes))

    hard = _pick_hard_negatives(rerank_df=rerank_df, active_codes=active_codes, cfg=cfg)
    hard_codes = set(hard["slot_code_11"].astype(str).tolist())

    out_df = base_df.copy()
    out_df[cfg.active_col] = out_df["slot_code_11"].map(lambda c: 1 if str(c) in active_set else 0).astype(int)
    out_df[cfg.hard_negative_col] = out_df["slot_code_11"].map(lambda c: 1 if str(c) in hard_codes else 0).astype(int)
    dist_map = dict(zip(hard["slot_code_11"].astype(str), hard[cfg.distance_col].astype(float)))
    out_df[cfg.distance_col] = out_df["slot_code_11"].map(lambda c: dist_map.get(str(c), np.nan))
    if cfg.local_target_col in out_df.columns:
        out_df[cfg.local_target_col] = pd.to_numeric(out_df[cfg.local_target_col], errors="coerce")
    else:
        out_df[cfg.local_target_col] = np.nan
    score_map = dict(zip(hard["slot_code_11"].astype(str), pd.to_numeric(hard[cfg.score_col], errors="coerce")))
    target_map = dict(zip(hard["slot_code_11"].astype(str), pd.to_numeric(hard[cfg.local_target_col], errors="coerce")))
    out_df[cfg.local_target_col] = out_df["slot_code_11"].map(
        lambda c: target_map.get(str(c), score_map.get(str(c), np.nan))
    )

    appended = 0
    if bool(cfg.include_missing_from_base) and len(hard) > 0:
        missing = hard[~hard["slot_code_11"].astype(str).isin(set(out_df["slot_code_11"].astype(str)))].copy()
        if len(missing) > 0:
            missing_cols = {c: np.nan for c in out_df.columns if c not in missing.columns}
            for col, val in missing_cols.items():
                missing[col] = val
            missing[cfg.active_col] = 0
            missing[cfg.hard_negative_col] = 1
            if "combo_compact" not in missing.columns:
                missing["combo_compact"] = missing["slot_code_11"]
            appended = int(len(missing))
            out_df = pd.concat([out_df, missing[out_df.columns]], ignore_index=True)

    out_df = out_df.drop_duplicates(subset=["slot_code_11"], keep="first").reset_index(drop=True)
    saved_train = write_table(out_df, out_table)

    hard_out = out_hard_negatives
    if hard_out is None:
        hard_out = str(Path(out_table).with_name("active_local_hard_negatives.csv"))
    saved_hard = write_table(hard, hard_out)

    summary = {
        "base_table": base_table,
        "rerank_table": rerank_table,
        "out_table": saved_train,
        "hard_negatives_table": saved_hard,
        "n_base_rows": int(len(base_df)),
        "n_out_rows": int(len(out_df)),
        "n_active_total": int(len(active_set)),
        "n_hard_negatives": int(len(hard)),
        "n_hard_negatives_appended": int(appended),
        "config": asdict(cfg),
    }
    summary_path = Path(saved_train).with_name("hard_negative_mining_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
