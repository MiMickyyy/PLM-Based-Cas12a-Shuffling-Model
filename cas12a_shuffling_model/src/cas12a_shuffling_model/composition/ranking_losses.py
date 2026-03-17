from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def pearson_corr_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() < 2 or b.numel() < 2:
        return torch.tensor(0.0, device=a.device)
    aa = a - a.mean()
    bb = b - b.mean()
    denom = torch.sqrt(torch.sum(aa * aa) * torch.sum(bb * bb))
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-8:
        return torch.tensor(0.0, device=a.device)
    return torch.clamp(torch.sum(aa * bb) / denom, min=-1.0, max=1.0)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if int(mask.sum().item()) < 2:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - pearson_corr_torch(pred[mask], target[mask])


def _sample_gap_bin_pairs(
    *,
    diffs: torch.Tensor,
    easy_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
    max_pairs: int,
) -> torch.Tensor:
    if diffs.numel() == 0 or max_pairs <= 0:
        return torch.empty(0, dtype=torch.long, device=diffs.device)
    q33 = torch.quantile(diffs, 0.33)
    q66 = torch.quantile(diffs, 0.66)
    hard = torch.nonzero(diffs <= q33, as_tuple=False).squeeze(-1)
    medium = torch.nonzero((diffs > q33) & (diffs <= q66), as_tuple=False).squeeze(-1)
    easy = torch.nonzero(diffs > q66, as_tuple=False).squeeze(-1)

    def _take(src: torch.Tensor, n: int) -> torch.Tensor:
        if src.numel() == 0 or n <= 0:
            return torch.empty(0, dtype=torch.long, device=diffs.device)
        if src.numel() <= n:
            return src
        sel = torch.randperm(src.numel(), device=diffs.device)[:n]
        return src[sel]

    easy_n = max(1, int(round(max_pairs * max(0.0, float(easy_ratio)))))
    medium_n = max(1, int(round(max_pairs * max(0.0, float(medium_ratio)))))
    hard_n = max(1, int(round(max_pairs * max(0.0, float(hard_ratio)))))
    picked = torch.cat([_take(easy, easy_n), _take(medium, medium_n), _take(hard, hard_n)], dim=0)
    if picked.numel() == 0:
        picked = torch.randperm(diffs.numel(), device=diffs.device)[: min(max_pairs, diffs.numel())]
    elif picked.numel() > max_pairs:
        sel = torch.randperm(picked.numel(), device=diffs.device)[:max_pairs]
        picked = picked[sel]
    return picked


def filtered_pairwise_rank_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    pairs_per_batch: int,
    min_gap: float,
    near_tie_gap: float,
    easy_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
    base_margin: float,
    margin_alpha: float,
    margin_min: float,
    margin_max: float,
) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
    t = target[idx]
    s = pred[idx]

    candidate = max(4 * int(pairs_per_batch), 256)
    ii = torch.randint(0, t.numel(), (candidate,), device=t.device)
    jj = torch.randint(0, t.numel(), (candidate,), device=t.device)
    valid = ii != jj
    ii = ii[valid]
    jj = jj[valid]
    if ii.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    diff = torch.abs(t[ii] - t[jj])
    keep = diff >= float(max(min_gap, near_tie_gap))
    ii = ii[keep]
    jj = jj[keep]
    diff = diff[keep]
    if ii.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    picked = _sample_gap_bin_pairs(
        diffs=diff,
        easy_ratio=easy_ratio,
        medium_ratio=medium_ratio,
        hard_ratio=hard_ratio,
        max_pairs=int(pairs_per_batch),
    )
    ii = ii[picked]
    jj = jj[picked]
    diff = diff[picked]
    if ii.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    sign = torch.sign(t[ii] - t[jj])
    nz = sign != 0
    ii = ii[nz]
    jj = jj[nz]
    sign = sign[nz]
    diff = diff[nz]
    if ii.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    margin = torch.clamp(
        float(base_margin) + float(margin_alpha) * diff,
        min=float(margin_min),
        max=float(margin_max),
    )
    margin_term = sign * (s[ii] - s[jj]) - margin
    pair_loss = F.softplus(-margin_term)
    weight = 1.0 + torch.clamp(diff, min=0.0)
    return torch.mean(pair_loss * weight)


def top_focus_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    top_fraction: float,
    pairs_per_batch: int,
    margin: float,
) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() < 4:
        return torch.tensor(0.0, device=pred.device)

    p = pred[idx]
    t = target[idx]
    n = int(t.shape[0])
    k = max(1, min(n - 1, int(round(n * float(top_fraction)))))
    order = torch.argsort(t, descending=True)
    top_idx = order[:k]
    rest_idx = order[k:]
    if rest_idx.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    n_pairs = int(max(1, pairs_per_batch))
    ti = top_idx[torch.randint(0, top_idx.numel(), (n_pairs,), device=t.device)]
    rj = rest_idx[torch.randint(0, rest_idx.numel(), (n_pairs,), device=t.device)]
    margin_term = (p[ti] - p[rj]) - float(margin)
    return F.softplus(-margin_term).mean()


def hard_negative_rank_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    top_fraction: float,
    pairs_per_batch: int,
    margin: float,
) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() < 4:
        return torch.tensor(0.0, device=pred.device)

    p = pred[idx]
    t = target[idx]
    n = int(t.shape[0])
    k = max(1, min(n - 1, int(round(n * float(top_fraction)))))

    order_t = torch.argsort(t, descending=True)
    top_idx = order_t[:k]
    rest_idx = order_t[k:]
    if rest_idx.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    m = max(1, min(int(pairs_per_batch), int(top_idx.numel()), int(rest_idx.numel())))
    hard_false_pos = rest_idx[torch.argsort(p[rest_idx], descending=True)[:m]]
    hard_false_neg = top_idx[torch.argsort(p[top_idx], descending=False)[:m]]

    margin_term = (p[hard_false_neg] - p[hard_false_pos]) - float(margin)
    return F.softplus(-margin_term).mean()


def _pearson_np(true: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    x = true[mask]
    y = pred[mask]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_np(true: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    x = pd.Series(true[mask])
    y = pd.Series(pred[mask])
    return float(x.corr(y, method="spearman"))


def topk_overlap(true: np.ndarray, pred: np.ndarray, frac: float) -> float:
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    t = true[mask]
    p = pred[mask]
    n = int(t.shape[0])
    k = max(1, min(n, int(round(n * float(frac)))))
    top_t = np.argsort(t)[::-1][:k]
    top_p = np.argsort(p)[::-1][:k]
    return float(len(set(top_t.tolist()) & set(top_p.tolist())) / max(1, k))


def pair_acc_by_gap_bin(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    min_gap: float = 0.01,
    max_pairs: int = 50000,
) -> dict[str, float]:
    mask = np.isfinite(true) & np.isfinite(pred)
    t = true[mask]
    p = pred[mask]
    n = int(t.shape[0])
    if n < 2:
        return {"overall": float("nan"), "easy": float("nan"), "medium": float("nan"), "hard": float("nan")}
    rng = np.random.default_rng(13)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    valid = i != j
    i = i[valid]
    j = j[valid]
    td = t[i] - t[j]
    pdiff = p[i] - p[j]
    ad = np.abs(td)
    keep = ad >= float(min_gap)
    td = td[keep]
    pdiff = pdiff[keep]
    ad = ad[keep]
    if ad.size == 0:
        return {"overall": float("nan"), "easy": float("nan"), "medium": float("nan"), "hard": float("nan")}
    q33 = np.quantile(ad, 0.33)
    q66 = np.quantile(ad, 0.66)
    acc = np.sign(td) == np.sign(pdiff)
    out = {"overall": float(np.mean(acc))}
    bins = {
        "hard": ad <= q33,
        "medium": (ad > q33) & (ad <= q66),
        "easy": ad > q66,
    }
    for name, m in bins.items():
        out[name] = float(np.mean(acc[m])) if np.any(m) else float("nan")
    return out


def ranking_metrics(true: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(true, dtype=np.float64)
    y_pred = np.asarray(pred, dtype=np.float64)
    metrics = {
        "global_corr_chimera": _pearson_np(y_true, y_pred),
        "spearman_corr_chimera": _spearman_np(y_true, y_pred),
        "top1_overlap": topk_overlap(y_true, y_pred, 0.01),
        "top5_overlap": topk_overlap(y_true, y_pred, 0.05),
        "top10_overlap": topk_overlap(y_true, y_pred, 0.10),
    }
    acc = pair_acc_by_gap_bin(y_true, y_pred)
    metrics["pair_acc_overall"] = acc["overall"]
    metrics["pair_acc_easy"] = acc["easy"]
    metrics["pair_acc_medium"] = acc["medium"]
    metrics["pair_acc_hard"] = acc["hard"]
    return metrics


def is_better_metric(
    *,
    new_metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
    primary_key: str = "global_corr_chimera",
    tie_breakers: Sequence[str] = ("top5_overlap", "top10_overlap"),
) -> bool:
    if best_metrics is None:
        return True
    def _val(d: dict[str, float], k: str) -> float:
        v = d.get(k, float("nan"))
        return float(v) if math.isfinite(float(v)) else -float("inf")
    if _val(new_metrics, primary_key) > _val(best_metrics, primary_key):
        return True
    if _val(new_metrics, primary_key) < _val(best_metrics, primary_key):
        return False
    for key in tie_breakers:
        if _val(new_metrics, key) > _val(best_metrics, key):
            return True
        if _val(new_metrics, key) < _val(best_metrics, key):
            return False
    return False
