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
class GatedPolicyConfig:
    policy_mode: str = "global_fixed"
    gate_signal: str = "kernel_similarity_to_actives"
    alpha_far: float = 0.60
    alpha_near: float = 0.15
    similarity_beta: float = 0.15
    kernel_gamma: float = 0.70
    density_radius: int = 3
    density_gamma: float = 1.0
    hard_distance_threshold: int = 3
    hard_similarity_threshold: float = 0.55
    soft_center: float | None = None
    soft_scale: float = 8.0
    recall_policy: str = "scan_teacher_mix"
    recall_teacher_weight: float = 0.70
    recall_active_weight: float = 0.10
    recall_scan_weight: float = 0.20
    recall_pool_size: int | None = 100000
    teacher_usage_mode: str = "none"
    teacher_plausibility_quantile: float = 0.05
    teacher_plausibility_penalty: float = 0.50
    slot_weights: tuple[float, ...] | None = None


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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _active_density_score(
    *,
    codes: Sequence[str],
    active_codes: Sequence[str],
    radius: int,
    gamma: float,
) -> np.ndarray:
    if len(codes) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(active_codes) == 0:
        return np.zeros((len(codes),), dtype=np.float64)
    x = codes_to_matrix(codes)
    a = codes_to_matrix(active_codes)
    ham = (x[:, None, :] != a[None, :, :]).sum(axis=2).astype(np.float64)
    within = (ham <= float(max(0, int(radius)))).astype(np.float64).mean(axis=1)
    kernel = np.exp(-float(gamma) * ham).mean(axis=1)
    return 0.5 * within + 0.5 * kernel


def _resolve_head_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "assistant_teacher_head_score" in df.columns:
        score_teacher = _as_float_array(df["assistant_teacher_head_score"])
    elif "teacher_global_score" in df.columns:
        score_teacher = _as_float_array(df["teacher_global_score"])
    elif "assistant_score" in df.columns:
        score_teacher = _as_float_array(df["assistant_score"])
    elif "s_scan_score" in df.columns:
        score_teacher = _as_float_array(df["s_scan_score"])
    else:
        score_teacher = np.zeros((len(df),), dtype=np.float64)

    if "assistant_active_head_score" in df.columns:
        score_active = _as_float_array(df["assistant_active_head_score"])
    elif "assistant_score" in df.columns:
        score_active = _as_float_array(df["assistant_score"])
    elif "s_scan_score" in df.columns:
        score_active = _as_float_array(df["s_scan_score"])
    else:
        score_active = score_teacher.copy()

    return score_teacher, score_active


def _recall_score(
    *,
    score_teacher: np.ndarray,
    score_active: np.ndarray,
    score_scan: np.ndarray,
    cfg: GatedPolicyConfig,
) -> np.ndarray:
    teacher_z = _robust_z(score_teacher)
    active_z = _robust_z(score_active)
    scan_z = _robust_z(score_scan)
    mode = str(cfg.recall_policy).strip().lower()
    if mode == "teacher_head":
        return teacher_z
    if mode == "scan_only":
        return scan_z
    if mode == "teacher_active_mix":
        return float(cfg.recall_teacher_weight) * teacher_z + (1.0 - float(cfg.recall_teacher_weight)) * active_z
    if mode == "active_only":
        return active_z
    return (
        float(cfg.recall_teacher_weight) * teacher_z
        + float(cfg.recall_active_weight) * active_z
        + float(cfg.recall_scan_weight) * scan_z
    )


def apply_gated_policy(
    df: pd.DataFrame,
    *,
    active_codes: Sequence[str] | None,
    cfg: GatedPolicyConfig,
) -> pd.DataFrame:
    out = df.copy()
    if "slot_code_11" not in out.columns:
        raise KeyError("Missing slot_code_11 column for gated policy")
    codes = out["slot_code_11"].astype(str).tolist()
    active_norm = ensure_active_codes(active_codes or [])

    score_teacher, score_active = _resolve_head_scores(out)
    score_scan = _as_float_array(out["s_scan_score"]) if "s_scan_score" in out.columns else score_active.copy()

    if len(active_norm) > 0:
        dist = min_hamming_distance(codes, active_norm).astype(np.float64)
        gate_signal = str(cfg.gate_signal).strip().lower()
        if gate_signal == "min_hamming_to_active":
            sim = 1.0 - (dist / 11.0)
        elif gate_signal == "active_density_score":
            sim = _active_density_score(
                codes=codes,
                active_codes=active_norm,
                radius=int(cfg.density_radius),
                gamma=float(cfg.density_gamma),
            )
        else:
            sim = active_similarity_score(
                codes=codes,
                active_codes=active_norm,
                mode="kernel_density_over_actives",
                gamma=float(cfg.kernel_gamma),
                slot_weights=cfg.slot_weights,
            ).astype(np.float64)
    else:
        dist = np.full((len(out),), 11.0, dtype=np.float64)
        sim = np.zeros((len(out),), dtype=np.float64)

    mode = str(cfg.policy_mode).strip().lower()
    gate_signal = str(cfg.gate_signal).strip().lower()
    if mode == "hard_gated":
        if gate_signal == "min_hamming_to_active":
            near = dist <= float(cfg.hard_distance_threshold)
        else:
            near = sim >= float(cfg.hard_similarity_threshold)
        near_w = near.astype(np.float64)
    elif mode == "soft_gated":
        if gate_signal == "min_hamming_to_active":
            center = float(cfg.soft_center) if cfg.soft_center is not None else float(cfg.hard_distance_threshold)
            near_w = _sigmoid((center - dist) * float(cfg.soft_scale))
        else:
            center = float(cfg.soft_center) if cfg.soft_center is not None else float(cfg.hard_similarity_threshold)
            near_w = _sigmoid((sim - center) * float(cfg.soft_scale))
    else:
        near_w = np.ones((len(out),), dtype=np.float64)

    teacher_w = float(cfg.alpha_far) * (1.0 - near_w) + float(cfg.alpha_near) * near_w
    teacher_w = np.clip(teacher_w, 0.0, 1.0)
    recall_stage = _recall_score(
        score_teacher=score_teacher,
        score_active=score_active,
        score_scan=score_scan,
        cfg=cfg,
    )
    rerank_stage = teacher_w * score_teacher + (1.0 - teacher_w) * score_active
    prior_gate = near_w if mode in {"hard_gated", "soft_gated"} else np.ones((len(out),), dtype=np.float64)
    final = rerank_stage + float(cfg.similarity_beta) * prior_gate * sim

    teacher_floor = np.nan
    floor_penalty = np.zeros((len(out),), dtype=np.float64)
    teacher_usage_mode = str(cfg.teacher_usage_mode).strip().lower()
    if teacher_usage_mode == "plausibility_filter":
        teacher_floor = float(np.quantile(score_teacher, float(cfg.teacher_plausibility_quantile)))
        deficit = np.clip(teacher_floor - score_teacher, a_min=0.0, a_max=None)
        floor_penalty = float(cfg.teacher_plausibility_penalty) * deficit
        final = final - floor_penalty

    out["score_teacher"] = score_teacher
    out["score_active"] = score_active
    out["sim_to_active"] = sim
    out["dist_to_active"] = dist
    out["active_similarity"] = sim
    out["min_hamming_to_active"] = dist
    out["gate_near_weight"] = near_w
    out["teacher_weight_dynamic"] = teacher_w
    out["recall_stage_score"] = recall_stage
    out["rerank_stage_score"] = rerank_stage
    out["teacher_floor_penalty"] = floor_penalty
    out["teacher_plausibility_floor"] = teacher_floor
    out["final_gated_score"] = final
    out["final_score"] = final
    out["policy_mode"] = mode
    out["gate_signal"] = gate_signal
    out["teacher_usage_mode"] = teacher_usage_mode
    return out
