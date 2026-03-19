from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.active_prior import (
    active_similarity_score,
    codes_to_matrix,
    ensure_active_codes,
    min_hamming_distance,
)


@dataclass(frozen=True)
class TwoStagePolicyConfig:
    recall_policy: str = "teacher_recall"
    final_rerank_policy: str = "active_only"
    recall_teacher_weight: float = 0.70
    recall_scan_weight: float = 0.30
    recall_active_weight: float = 0.00
    recall_diversity_weight: float = 0.05
    recall_pool_size: int | None = 100000
    active_similarity_mode: str = "kernel_density_over_actives"
    active_similarity_beta: float = 0.15
    active_similarity_gamma: float = 0.70
    active_similarity_slot_weights: tuple[float, ...] | None = None
    teacher_plausibility_quantile: float = 0.05
    teacher_plausibility_penalty: float = 0.50
    teacher_plausibility_floor_mode: str = "global"
    teacher_auto_flip: bool = True
    teacher_guardrail: bool = True
    teacher_guardrail_min_frac: float = 0.90
    teacher_recall_max_abs_z: float = 2.5


def _as_float_array(values: pd.Series | Sequence[float], *, fill: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, fill)
    return arr.astype(np.float64)


def _robust_z(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(x)
    if int(np.sum(finite)) < 2:
        return np.zeros_like(x, dtype=np.float64)
    v = x[finite]
    med = float(np.median(v))
    q1 = float(np.quantile(v, 0.25))
    q3 = float(np.quantile(v, 0.75))
    scale = (q3 - q1) / 1.349 if (q3 - q1) > 1e-8 else float(np.std(v))
    if (not np.isfinite(scale)) or scale <= 1e-8:
        scale = 1.0
    out = (x - med) / scale
    out[~finite] = 0.0
    return out


def _resolve_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "assistant_teacher_head_score" in df.columns:
        score_teacher = _as_float_array(df["assistant_teacher_head_score"])
    elif "teacher_global_score" in df.columns:
        score_teacher = _as_float_array(df["teacher_global_score"])
    elif "assistant_score" in df.columns:
        score_teacher = _as_float_array(df["assistant_score"])
    else:
        score_teacher = np.zeros((len(df),), dtype=np.float64)

    if "assistant_active_head_score" in df.columns:
        score_active = _as_float_array(df["assistant_active_head_score"])
    elif "assistant_score" in df.columns:
        score_active = _as_float_array(df["assistant_score"])
    else:
        score_active = score_teacher.copy()

    if "s_scan_score" in df.columns:
        score_scan = _as_float_array(df["s_scan_score"])
    else:
        score_scan = score_active.copy()
    return score_teacher, score_active, score_scan


def _slot_novelty_against_active(
    *,
    codes: Sequence[str],
    active_codes: Sequence[str],
) -> np.ndarray:
    if len(codes) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(active_codes) == 0:
        return np.zeros((len(codes),), dtype=np.float64)
    x = codes_to_matrix(codes)
    a = codes_to_matrix(active_codes)
    n_slots = int(x.shape[1])
    freq = np.zeros((n_slots, 4), dtype=np.float64)
    for s in range(n_slots):
        vals = a[:, s]
        for v in range(4):
            freq[s, v] = float(np.mean(vals == v))
    novelty = np.zeros((x.shape[0],), dtype=np.float64)
    for s in range(n_slots):
        novelty += 1.0 - freq[s, x[:, s]]
    novelty /= float(n_slots)
    return novelty


def _hits_in_topk(
    *,
    scores: np.ndarray,
    codes: Sequence[str],
    active_codes: Sequence[str],
    top_k: int,
) -> int:
    if top_k <= 0:
        return 0
    if len(codes) == 0 or len(active_codes) == 0:
        return 0
    k = int(min(top_k, len(codes)))
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    top_codes = {str(codes[i]) for i in order[:k]}
    actives = set(str(x) for x in active_codes)
    return int(len(top_codes & actives))


def apply_two_stage_policy(
    df: pd.DataFrame,
    *,
    active_codes: Sequence[str] | None,
    cfg: TwoStagePolicyConfig,
) -> pd.DataFrame:
    out = df.copy()
    if "slot_code_11" not in out.columns:
        raise KeyError("Missing slot_code_11 column for two-stage policy")

    codes = out["slot_code_11"].astype(str).tolist()
    active_norm = ensure_active_codes(active_codes or [])
    score_teacher, score_active, score_scan = _resolve_scores(out)

    teacher_z = _robust_z(score_teacher)
    active_z = _robust_z(score_active)
    scan_z = _robust_z(score_scan)
    teacher_z = np.clip(teacher_z, -float(cfg.teacher_recall_max_abs_z), float(cfg.teacher_recall_max_abs_z))
    teacher_flipped = 0
    teacher_guardrail_fallback = 0

    if len(active_norm) > 0:
        sim_to_active = active_similarity_score(
            codes=codes,
            active_codes=active_norm,
            mode=str(cfg.active_similarity_mode),
            gamma=float(cfg.active_similarity_gamma),
            slot_weights=cfg.active_similarity_slot_weights,
        ).astype(np.float64)
        dist_to_active = min_hamming_distance(codes, active_norm).astype(np.float64)
        novelty_bonus = _slot_novelty_against_active(codes=codes, active_codes=active_norm)
    else:
        sim_to_active = np.zeros((len(out),), dtype=np.float64)
        dist_to_active = np.full((len(out),), 11.0, dtype=np.float64)
        novelty_bonus = np.zeros((len(out),), dtype=np.float64)

    recall_policy = str(cfg.recall_policy).strip().lower()
    if recall_policy == "scan_teacher_mix":
        recall_policy = "teacher_recall"
    if bool(cfg.teacher_auto_flip) and len(active_norm) > 0:
        probe_top_k = (
            int(cfg.recall_pool_size)
            if cfg.recall_pool_size is not None
            else int(min(len(codes), 100000))
        )
        probe_top_k = max(1, probe_top_k)
        pos_hits = _hits_in_topk(scores=teacher_z, codes=codes, active_codes=active_norm, top_k=probe_top_k)
        neg_hits = _hits_in_topk(scores=-teacher_z, codes=codes, active_codes=active_norm, top_k=probe_top_k)
        if neg_hits > pos_hits:
            teacher_z = -teacher_z
            score_teacher = -score_teacher
            teacher_flipped = 1

    if recall_policy == "scan_only":
        recall_score = scan_z
    elif recall_policy == "teacher_head":
        recall_score = teacher_z
    elif recall_policy == "teacher_active_mix":
        recall_score = (
            float(cfg.recall_teacher_weight) * teacher_z
            + float(cfg.recall_active_weight) * active_z
        )
    elif recall_policy == "teacher_recall":
        recall_score = (
            float(cfg.recall_teacher_weight) * teacher_z
            + float(cfg.recall_scan_weight) * scan_z
            + float(cfg.recall_active_weight) * active_z
        )
    elif recall_policy == "teacher_plausibility_filter":
        floor = float(np.quantile(score_teacher, float(cfg.teacher_plausibility_quantile)))
        penalty = np.clip(floor - score_teacher, a_min=0.0, a_max=None) * float(cfg.teacher_plausibility_penalty)
        recall_score = scan_z - penalty
    elif recall_policy == "teacher_recall_plus_diversity":
        recall_score = (
            float(cfg.recall_teacher_weight) * teacher_z
            + float(cfg.recall_scan_weight) * scan_z
            + float(cfg.recall_active_weight) * active_z
            + float(cfg.recall_diversity_weight) * novelty_bonus
        )
    elif recall_policy == "active_only":
        recall_score = active_z
    else:
        raise ValueError(f"Unknown recall_policy: {cfg.recall_policy}")
    if (
        bool(cfg.teacher_guardrail)
        and len(active_norm) > 0
        and recall_policy in {"teacher_recall", "teacher_recall_plus_diversity", "teacher_head", "teacher_active_mix"}
    ):
        probe_top_k = (
            int(cfg.recall_pool_size)
            if cfg.recall_pool_size is not None
            else int(min(len(codes), 100000))
        )
        probe_top_k = max(1, probe_top_k)
        scan_hits = _hits_in_topk(scores=scan_z, codes=codes, active_codes=active_norm, top_k=probe_top_k)
        mixed_hits = _hits_in_topk(scores=recall_score, codes=codes, active_codes=active_norm, top_k=probe_top_k)
        min_allowed = int(np.floor(float(cfg.teacher_guardrail_min_frac) * float(scan_hits)))
        if mixed_hits < min_allowed:
            recall_score = scan_z.copy()
            teacher_guardrail_fallback = 1

    rerank_policy = str(cfg.final_rerank_policy).strip().lower()
    if rerank_policy == "active_only":
        rerank_score = score_active + float(cfg.active_similarity_beta) * sim_to_active
        floor_penalty = np.zeros((len(out),), dtype=np.float64)
    elif rerank_policy == "active_heavy":
        base = score_active + float(cfg.active_similarity_beta) * sim_to_active
        floor = float(np.quantile(score_teacher, float(cfg.teacher_plausibility_quantile)))
        floor_penalty = np.clip(floor - score_teacher, a_min=0.0, a_max=None) * float(cfg.teacher_plausibility_penalty)
        rerank_score = base - floor_penalty
    else:
        raise ValueError(f"Unknown final_rerank_policy: {cfg.final_rerank_policy}")

    out["score_teacher"] = score_teacher
    out["score_active"] = score_active
    out["score_scan"] = score_scan
    out["sim_to_active"] = sim_to_active
    out["dist_to_active"] = dist_to_active
    out["active_similarity"] = sim_to_active
    out["min_hamming_to_active"] = dist_to_active
    out["recall_stage_score"] = recall_score
    out["rerank_stage_score"] = rerank_score
    out["final_score"] = rerank_score
    out["final_gated_score"] = rerank_score
    out["teacher_floor_penalty"] = floor_penalty
    out["recall_policy"] = recall_policy
    out["final_rerank_policy"] = rerank_policy
    out["teacher_recall_flipped"] = int(teacher_flipped)
    out["teacher_recall_guardrail_fallback"] = int(teacher_guardrail_fallback)
    return out
