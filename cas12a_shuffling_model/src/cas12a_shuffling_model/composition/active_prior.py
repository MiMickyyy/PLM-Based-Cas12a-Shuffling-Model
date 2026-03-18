from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.chimera_repr import (
    SLOT_COUNT,
    load_active_code_counts,
    slot_code_to_int_array,
    validate_slot_code_11,
)


ACTIVE_PRIOR_MODES: tuple[str, ...] = (
    "none",
    "min_hamming_similarity",
    "weighted_slot_similarity",
    "kernel_density_over_actives",
)


@dataclass(frozen=True)
class ActivePriorConfig:
    mode: str = "none"
    beta: float = 0.0
    gamma: float = 0.7
    slot_weights: tuple[float, ...] | None = None


def load_active_codes(active_table: str | Path) -> list[str]:
    counts = load_active_code_counts(active_table)
    return sorted(counts.keys())


def codes_to_matrix(codes: Sequence[str]) -> np.ndarray:
    if len(codes) == 0:
        return np.zeros((0, SLOT_COUNT), dtype=np.int64)
    return np.asarray([slot_code_to_int_array(validate_slot_code_11(c)) for c in codes], dtype=np.int64)


def min_hamming_distance(codes: Sequence[str], active_codes: Sequence[str]) -> np.ndarray:
    x = codes_to_matrix(codes)
    a = codes_to_matrix(active_codes)
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if a.shape[0] == 0:
        return np.full((x.shape[0],), SLOT_COUNT, dtype=np.int64)
    dist = (x[:, None, :] != a[None, :, :]).sum(axis=2)
    return dist.min(axis=1).astype(np.int64)


def _resolve_slot_weights(slot_weights: Sequence[float] | None) -> np.ndarray:
    if slot_weights is None or len(slot_weights) == 0:
        w = np.ones((SLOT_COUNT,), dtype=np.float64)
    else:
        if len(slot_weights) != SLOT_COUNT:
            raise ValueError(f"slot_weights length must be {SLOT_COUNT}")
        w = np.asarray(slot_weights, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        w = np.ones((SLOT_COUNT,), dtype=np.float64)
    return w / np.sum(w)


def active_similarity_score(
    *,
    codes: Sequence[str],
    active_codes: Sequence[str],
    mode: str,
    gamma: float = 0.7,
    slot_weights: Sequence[float] | None = None,
) -> np.ndarray:
    if len(codes) == 0:
        return np.zeros((0,), dtype=np.float64)
    m = str(mode).strip().lower()
    if m == "none" or len(active_codes) == 0:
        return np.zeros((len(codes),), dtype=np.float64)

    x = codes_to_matrix(codes)
    a = codes_to_matrix(active_codes)
    diff = (x[:, None, :] != a[None, :, :]).astype(np.float64)
    ham = diff.sum(axis=2)

    if m == "min_hamming_similarity":
        min_d = ham.min(axis=1)
        return 1.0 - (min_d / float(SLOT_COUNT))

    if m == "weighted_slot_similarity":
        w = _resolve_slot_weights(slot_weights)
        eq = 1.0 - diff
        sim = (eq * w[None, None, :]).sum(axis=2)
        return sim.max(axis=1)

    if m == "kernel_density_over_actives":
        k = np.exp(-float(gamma) * ham)
        return k.mean(axis=1)

    raise ValueError(f"Unknown active similarity mode: {mode}")


def apply_active_prior(
    df: pd.DataFrame,
    *,
    score_col: str,
    active_codes: Sequence[str],
    cfg: ActivePriorConfig,
) -> pd.DataFrame:
    out = df.copy()
    if score_col not in out.columns:
        raise KeyError(f"Missing score column for active prior: {score_col}")
    codes = out["slot_code_11"].astype(str).tolist()
    sim = active_similarity_score(
        codes=codes,
        active_codes=active_codes,
        mode=cfg.mode,
        gamma=cfg.gamma,
        slot_weights=cfg.slot_weights,
    )
    out["active_similarity"] = sim.astype(np.float64)
    out["active_prior_mode"] = cfg.mode
    out["active_prior_beta"] = float(cfg.beta)
    out["final_score"] = pd.to_numeric(out[score_col], errors="coerce").to_numpy(dtype=np.float64) + float(cfg.beta) * sim
    out["min_hamming_to_active"] = min_hamming_distance(codes, active_codes).astype(np.int64)
    return out


def active_ranking_summary(
    *,
    df: pd.DataFrame,
    score_col: str,
    active_codes: Sequence[str],
    top_ks: Sequence[int] = (50, 100),
    distance_ks: Sequence[int] = (50,),
) -> dict[str, float | int | list]:
    scored = df.copy()
    scored = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1, dtype=np.int64)
    active_set = set(active_codes)

    codes = scored["slot_code_11"].astype(str).tolist()
    code_to_rank = {c: int(i + 1) for i, c in enumerate(codes)}
    active_ranks = {c: code_to_rank.get(c, None) for c in sorted(active_set)}
    present_ranks = [r for r in active_ranks.values() if r is not None]
    out: dict[str, float | int | list] = {
        "n_rows": int(len(scored)),
        "n_active_total": int(len(active_set)),
        "n_active_present": int(len(present_ranks)),
        "best_rank_active": int(min(present_ranks)) if present_ranks else None,
        "median_rank_active": float(np.median(present_ranks)) if present_ranks else None,
        "active_ranks": active_ranks,
    }
    for k in top_ks:
        kk = max(1, min(int(k), len(scored)))
        top_codes = set(scored.head(kk)["slot_code_11"].astype(str).tolist())
        hits = len(top_codes & active_set)
        out[f"hits_at_{kk}"] = int(hits)
        out[f"recall_at_{kk}"] = float(hits / len(active_set)) if len(active_set) > 0 else float("nan")

    if len(active_set) > 0 and len(scored) > 0:
        for k in distance_ks:
            dcap = max(1, min(int(k), len(scored)))
            top = scored.head(dcap)
            d = min_hamming_distance(top["slot_code_11"].astype(str).tolist(), list(active_set)).astype(np.int64)
            out[f"top{dcap}_min_hamming_min"] = int(np.min(d))
            out[f"top{dcap}_min_hamming_median"] = float(np.median(d))
            out[f"top{dcap}_within_hamming_1"] = int(np.sum(d <= 1))
            out[f"top{dcap}_within_hamming_2"] = int(np.sum(d <= 2))
            out[f"top{dcap}_within_hamming_4"] = int(np.sum(d <= 4))
    return out


def active_prior_beta_sweep(
    *,
    df: pd.DataFrame,
    base_score_col: str,
    active_codes: Sequence[str],
    mode: str,
    betas: Sequence[float],
    gamma: float = 0.7,
    slot_weights: Sequence[float] | None = None,
    top_ks: Sequence[int] = (50, 100),
) -> pd.DataFrame:
    work = df.copy()
    if base_score_col not in work.columns:
        raise KeyError(f"Missing base score column: {base_score_col}")
    sim = active_similarity_score(
        codes=work["slot_code_11"].astype(str).tolist(),
        active_codes=active_codes,
        mode=mode,
        gamma=gamma,
        slot_weights=slot_weights,
    )
    base = pd.to_numeric(work[base_score_col], errors="coerce").to_numpy(dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for b in betas:
        score = base + float(b) * sim
        tmp = work.copy()
        tmp["_score"] = score
        m = active_ranking_summary(
            df=tmp,
            score_col="_score",
            active_codes=active_codes,
            top_ks=top_ks,
            distance_ks=top_ks,
        )
        rec: dict[str, float | int] = {"beta": float(b)}
        for k, v in m.items():
            if isinstance(v, (int, float)) or v is None:
                rec[k] = v
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)
    return out


def parse_slot_weights(text: str | Sequence[float] | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return tuple(float(x) for x in text)
    raw = str(text).strip()
    if not raw:
        return None
    if raw.startswith("["):
        arr = json.loads(raw.replace("'", "\""))
        return tuple(float(x) for x in arr)
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return tuple(float(x) for x in vals)


def parse_beta_list(text: str | Sequence[float] | None) -> list[float]:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    raw = str(text).strip()
    if not raw:
        return []
    if raw.startswith("["):
        arr = json.loads(raw.replace("'", "\""))
        return [float(x) for x in arr]
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(v) for v in vals]


def write_summary_json(summary: dict, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(p)


def ensure_active_codes(active_codes: Iterable[str]) -> list[str]:
    out = []
    for code in active_codes:
        try:
            out.append(validate_slot_code_11(str(code)))
        except Exception:
            continue
    return sorted(set(out))
