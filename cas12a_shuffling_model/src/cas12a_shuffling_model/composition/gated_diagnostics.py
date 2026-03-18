from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.active_prior import active_ranking_summary, ensure_active_codes
from cas12a_shuffling_model.composition.gated_policy import GatedPolicyConfig, apply_gated_policy


@dataclass(frozen=True)
class PolicyEvalResult:
    name: str
    scored: pd.DataFrame
    recall_sorted: pd.DataFrame
    rerank_pool: pd.DataFrame


def _rank_map(df: pd.DataFrame, score_col: str) -> dict[str, int]:
    if score_col not in df.columns:
        return {}
    work = df.copy()
    work = work.sort_values(score_col, ascending=False).reset_index(drop=True)
    return {str(c): int(i + 1) for i, c in enumerate(work["slot_code_11"].astype(str).tolist())}


def _flatten_metrics(prefix: str, metrics: Mapping[str, object]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) or v is None:
            out[f"{prefix}_{k}"] = v  # type: ignore[assignment]
    return out


def evaluate_policy_once(
    *,
    table_df: pd.DataFrame,
    active_codes: Sequence[str],
    cfg: GatedPolicyConfig,
    rerank_top_ks: Sequence[int] = (50, 100),
    recall_top_ks: Sequence[int] = (20000, 50000, 100000),
) -> PolicyEvalResult:
    scored = apply_gated_policy(table_df, active_codes=active_codes, cfg=cfg)
    recall_sorted = scored.sort_values("recall_stage_score", ascending=False).reset_index(drop=True)
    pool_size = int(cfg.recall_pool_size) if cfg.recall_pool_size is not None else int(len(recall_sorted))
    pool_size = max(1, min(pool_size, int(len(recall_sorted))))
    rerank_pool = (
        recall_sorted.head(pool_size)
        .sort_values("final_gated_score", ascending=False)
        .reset_index(drop=True)
    )
    rerank_pool["final_rank"] = np.arange(1, len(rerank_pool) + 1, dtype=np.int64)
    recall_rank_map = _rank_map(recall_sorted, "recall_stage_score")
    rerank_pool["recall_stage_rank"] = rerank_pool["slot_code_11"].astype(str).map(recall_rank_map)
    _ = rerank_top_ks, recall_top_ks
    return PolicyEvalResult(
        name="",
        scored=scored,
        recall_sorted=recall_sorted,
        rerank_pool=rerank_pool,
    )


def evaluate_policy_variants(
    *,
    table_df: pd.DataFrame,
    active_codes: Sequence[str],
    variants: Mapping[str, GatedPolicyConfig],
    rerank_top_ks: Sequence[int] = (50, 100),
    recall_top_ks: Sequence[int] = (20000, 50000, 100000),
) -> tuple[pd.DataFrame, dict[str, PolicyEvalResult]]:
    active_norm = ensure_active_codes(active_codes)
    rows: list[dict[str, float | int | str | None]] = []
    details: dict[str, PolicyEvalResult] = {}
    for name, cfg in variants.items():
        result = evaluate_policy_once(
            table_df=table_df,
            active_codes=active_norm,
            cfg=cfg,
            rerank_top_ks=rerank_top_ks,
            recall_top_ks=recall_top_ks,
        )
        details[name] = PolicyEvalResult(
            name=name,
            scored=result.scored,
            recall_sorted=result.recall_sorted,
            rerank_pool=result.rerank_pool,
        )
        recall_summary = active_ranking_summary(
            df=result.recall_sorted,
            score_col="recall_stage_score",
            active_codes=active_norm,
            top_ks=tuple(int(x) for x in recall_top_ks),
            distance_ks=tuple(int(x) for x in rerank_top_ks),
        )
        rerank_summary = active_ranking_summary(
            df=result.rerank_pool,
            score_col="final_gated_score",
            active_codes=active_norm,
            top_ks=tuple(int(x) for x in rerank_top_ks),
            distance_ks=tuple(int(x) for x in rerank_top_ks),
        )
        row: dict[str, float | int | str | None] = {
            "policy_name": name,
            "policy_mode": cfg.policy_mode,
            "gate_signal": cfg.gate_signal,
            "teacher_usage_mode": cfg.teacher_usage_mode,
            "alpha_far": float(cfg.alpha_far),
            "alpha_near": float(cfg.alpha_near),
            "similarity_beta": float(cfg.similarity_beta),
            "recall_pool_size": int(cfg.recall_pool_size) if cfg.recall_pool_size is not None else int(len(result.recall_sorted)),
        }
        row.update(_flatten_metrics("recall", recall_summary))
        row.update(_flatten_metrics("rerank", rerank_summary))
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) > 0:
        sort_col = "rerank_hits_at_50" if "rerank_hits_at_50" in out.columns else out.columns[0]
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out, details


def leave_one_active_out_eval(
    *,
    table_df: pd.DataFrame,
    active_codes: Sequence[str],
    cfg: GatedPolicyConfig,
    rerank_top_ks: Sequence[int] = (50, 100),
) -> pd.DataFrame:
    active_norm = ensure_active_codes(active_codes)
    rows: list[dict[str, float | int | str | None]] = []
    for held in active_norm:
        anchors = [c for c in active_norm if c != held]
        result = evaluate_policy_once(
            table_df=table_df,
            active_codes=anchors,
            cfg=cfg,
            rerank_top_ks=rerank_top_ks,
            recall_top_ks=(20000, 50000, 100000),
        )
        rank_map = _rank_map(result.rerank_pool, "final_gated_score")
        held_rank = rank_map.get(held)
        rec = {
            "held_active": held,
            "held_rank": held_rank,
            "held_hit_top50": int(held_rank is not None and held_rank <= 50),
            "held_hit_top100": int(held_rank is not None and held_rank <= 100),
            "n_present_in_pool": int(sum(1 for c in active_norm if c in rank_map)),
        }
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["held_rank"] = pd.to_numeric(out["held_rank"], errors="coerce")
    return out


def distance_stratified_active_analysis(
    *,
    ranked_df: pd.DataFrame,
    active_codes: Sequence[str],
    score_col: str = "final_gated_score",
) -> pd.DataFrame:
    active_norm = ensure_active_codes(active_codes)
    if len(active_norm) == 0:
        return pd.DataFrame()
    work = ranked_df.copy()
    if score_col not in work.columns:
        raise KeyError(f"Missing score column: {score_col}")
    work = work.sort_values(score_col, ascending=False).reset_index(drop=True)
    rank_map = {str(c): int(i + 1) for i, c in enumerate(work["slot_code_11"].astype(str).tolist())}

    mat = np.asarray([[1 if ch == "A" else 2 if ch == "L" else 3 if ch == "F" else 4 for ch in code] for code in active_norm], dtype=np.int16)
    dmat = np.sum(mat[:, None, :] != mat[None, :, :], axis=2).astype(np.int64)
    np.fill_diagonal(dmat, 999)
    nearest = np.min(dmat, axis=1).astype(np.float64)
    q1 = float(np.quantile(nearest, 1 / 3))
    q2 = float(np.quantile(nearest, 2 / 3))

    def _bucket(d: float) -> str:
        if d <= q1:
            return "dense"
        if d <= q2:
            return "medium"
        return "isolated"

    rows = []
    for code, d in zip(active_norm, nearest.tolist()):
        r = rank_map.get(code)
        rows.append(
            {
                "active_code": code,
                "nearest_active_distance": float(d),
                "distance_bucket": _bucket(float(d)),
                "rank": r,
                "hit_top50": int(r is not None and r <= 50),
                "hit_top100": int(r is not None and r <= 100),
            }
        )
    return pd.DataFrame(rows).sort_values(["distance_bucket", "nearest_active_distance", "active_code"]).reset_index(drop=True)


def novelty_vs_score_analysis(
    *,
    ranked_df: pd.DataFrame,
    active_codes: Sequence[str],
    score_cols: Sequence[str],
    top_k: int = 1000,
) -> pd.DataFrame:
    active_norm = ensure_active_codes(active_codes)
    rows: list[dict[str, float | int | str]] = []
    for score_col in score_cols:
        if score_col not in ranked_df.columns:
            continue
        top = ranked_df.sort_values(score_col, ascending=False).head(int(max(1, top_k))).copy()
        if len(top) == 0:
            continue
        if len(active_norm) > 0:
            from cas12a_shuffling_model.composition.active_prior import min_hamming_distance

            d = min_hamming_distance(top["slot_code_11"].astype(str).tolist(), active_norm).astype(np.float64)
        else:
            d = np.full((len(top),), 11.0, dtype=np.float64)
        s = pd.to_numeric(top[score_col], errors="coerce").to_numpy(dtype=np.float64)
        corr = float(np.corrcoef(s, -d)[0, 1]) if len(top) > 2 and np.std(s) > 1e-8 and np.std(d) > 1e-8 else float("nan")
        rows.append(
            {
                "score_col": score_col,
                "top_k": int(len(top)),
                "score_dist_neg_corr": corr,
                "min_dist": float(np.min(d)),
                "median_dist": float(np.median(d)),
                "mean_dist": float(np.mean(d)),
                "within_hamming_1": int(np.sum(d <= 1)),
                "within_hamming_2": int(np.sum(d <= 2)),
                "within_hamming_4": int(np.sum(d <= 4)),
            }
        )
    return pd.DataFrame(rows)


def missing_active_diagnosis(
    *,
    ranked_df: pd.DataFrame,
    active_codes: Sequence[str],
    recall_pool_size: int = 100000,
) -> pd.DataFrame:
    active_norm = ensure_active_codes(active_codes)
    if len(active_norm) == 0:
        return pd.DataFrame()
    cols = [
        "s_scan_score",
        "score_teacher",
        "score_active",
        "recall_stage_score",
        "rerank_stage_score",
        "final_gated_score",
        "final_score",
    ]
    rank_maps = {c: _rank_map(ranked_df, c) for c in cols if c in ranked_df.columns}

    rows = []
    pool_n = int(min(int(recall_pool_size), len(ranked_df)))
    for code in active_norm:
        rec: dict[str, float | int | str | None] = {"active_code": code}
        present = code in set(ranked_df["slot_code_11"].astype(str).tolist())
        rec["present_in_input_table"] = int(present)
        for c, m in rank_maps.items():
            rec[f"rank_{c}"] = m.get(code)
        recall_rank = rec.get("rank_recall_stage_score")
        final_rank = rec.get("rank_final_gated_score", rec.get("rank_final_score"))
        if not present:
            rec["fail_stage"] = "recall_stage"
        elif isinstance(recall_rank, int) and int(recall_rank) > pool_n:
            rec["fail_stage"] = "recall_stage"
        elif isinstance(final_rank, int) and int(final_rank) > 100:
            rec["fail_stage"] = "rerank_stage"
        else:
            rec["fail_stage"] = "recovered"
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["fail_stage", "active_code"]).reset_index(drop=True)
