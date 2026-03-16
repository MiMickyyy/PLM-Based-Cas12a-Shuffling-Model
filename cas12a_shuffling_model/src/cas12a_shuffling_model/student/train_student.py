from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from functools import partial
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from cas12a_shuffling_model.student.distill_dataset import (
    DistillDataset,
    collate_distill_batch,
    load_distill_records_from_csv,
    split_indices,
)
from cas12a_shuffling_model.student.gru_model import GRUAutoregressiveLM
from cas12a_shuffling_model.student.transformer_model import TransformerAutoregressiveLM
from cas12a_shuffling_model.student.vocab import AminoAcidVocab, build_default_vocab
from cas12a_shuffling_model.teacher.junction_scoring import (
    JunctionWindowConfig,
    compute_boundary_positions,
    score_windows_from_per_residue_ll,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudentModelConfig:
    model_type: str = "gru"
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    num_heads: int = 4
    ff_dim: int = 512
    max_positions: int = 4096


@dataclass(frozen=True)
class StudentTrainConfig:
    seed: int = 13
    batch_size: int = 8
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    nll_weight: float = 1.0
    global_weight: float = 1.0
    junction_weight: float = 1.0
    natural_global_weight: float = 1.0
    chimera_global_weight: float = 1.0
    natural_junction_weight: float = 1.0
    chimera_junction_weight: float = 1.0
    correlation_weight: float = 0.0
    correlation_on_chimera_only: bool = True
    pairwise_weight: float = 0.0
    pairwise_margin: float = 0.0
    pairwise_pairs_per_batch: int = 64
    pairwise_min_teacher_diff: float = 0.01
    pairwise_ignore_close_diff: float = 0.0
    pairwise_hard_ratio: float = 0.5
    pairwise_medium_ratio: float = 0.3
    pairwise_easy_ratio: float = 0.2
    pairwise_length_bin_size: int = 64
    pairwise_on_chimera_only: bool = True
    pairwise_warmup_epochs: int = 0
    stage_a_natural_epochs: int = 0
    stage_a_batch_size: int | None = None
    stage_b_chimera_only: bool = False
    normalize_teacher_global: bool = False
    normalize_length_bin_size: int = 64
    normalize_min_group_size: int = 32
    topk_fracs: tuple[float, ...] = (0.01, 0.05, 0.10)
    best_metric: str = "val_loss"
    best_metric_mode: str = "auto"
    balance_source_types: bool = False
    num_workers: int = 0
    device: str | None = None
    cpu_threads: int | None = None
    interop_threads: int | None = None


def _normalize_model_type(model_type: str | None) -> str:
    mt = str(model_type or "gru").strip().lower()
    if mt in {"gru", "rnn"}:
        return "gru"
    if mt in {"transformer", "tx", "small_transformer"}:
        return "transformer"
    raise ValueError(f"Unsupported student model_type: {model_type}")


def build_student_model(
    *,
    cfg: StudentModelConfig,
    vocab_size: int,
    pad_idx: int,
) -> nn.Module:
    model_type = _normalize_model_type(cfg.model_type)
    if model_type == "gru":
        return GRUAutoregressiveLM(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_idx=pad_idx,
        )
    return TransformerAutoregressiveLM(
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        pad_idx=pad_idx,
        max_positions=cfg.max_positions,
    )


def detect_torch_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(cfg: StudentTrainConfig) -> None:
    if cfg.cpu_threads is not None and int(cfg.cpu_threads) > 0:
        torch.set_num_threads(int(cfg.cpu_threads))
    if cfg.interop_threads is not None and int(cfg.interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(cfg.interop_threads))
        except RuntimeError:
            pass


def _nanmean_np(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mask = np.isfinite(arr)
    if not mask.any():
        return float("nan")
    return float(np.mean(arr[mask]))


def _pearson_np(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 2:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask], target[mask])


def _weighted_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = torch.isfinite(target)
    if sample_weights is None:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return F.mse_loss(pred[mask], target[mask])

    w = sample_weights
    if w.ndim == 1 and pred.ndim == 2:
        w = w.unsqueeze(-1).expand_as(pred)
    if w.ndim != pred.ndim:
        raise ValueError("sample_weights shape mismatch")
    if w.device != pred.device:
        w = w.to(pred.device)

    finite_w = torch.isfinite(w)
    mask = mask & finite_w
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    wv = torch.clamp(w[mask], min=0.0)
    denom = wv.sum()
    if float(denom.item()) <= 0.0:
        return torch.tensor(0.0, device=pred.device)
    diff = (pred - target) ** 2
    return (diff[mask] * wv).sum() / denom


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    q1 = float(np.quantile(vals, 0.25))
    q3 = float(np.quantile(vals, 0.75))
    iqr = q3 - q1
    robust_std = iqr / 1.349 if iqr > 1e-8 else float(np.std(vals))
    if not math.isfinite(robust_std) or robust_std <= 1e-8:
        robust_std = float(np.std(vals))
    if not math.isfinite(robust_std) or robust_std <= 1e-8:
        robust_std = 1.0
    return med, robust_std


def _length_bin(length: int, bin_size: int) -> int:
    bsz = max(1, int(bin_size))
    return int(length // bsz) * bsz


def _normalize_teacher_global_scores(
    records: Sequence[Any], cfg: StudentTrainConfig
) -> list[Any]:
    if not bool(cfg.normalize_teacher_global):
        return list(records)
    if len(records) == 0:
        return []

    teacher_vals = np.asarray([float(r.teacher_global) for r in records], dtype=np.float64)
    source_only_groups: dict[str, list[int]] = {}
    source_len_groups: dict[tuple[str, int], list[int]] = {}
    for idx, rec in enumerate(records):
        st = str(rec.source_type).strip().lower() or "unknown"
        lb = _length_bin(len(str(rec.sequence_aa)), int(cfg.normalize_length_bin_size))
        source_only_groups.setdefault(st, []).append(idx)
        source_len_groups.setdefault((st, lb), []).append(idx)

    global_med, global_std = _safe_mean_std(teacher_vals)
    source_stats: dict[str, tuple[float, float]] = {
        st: _safe_mean_std(teacher_vals[idxs]) for st, idxs in source_only_groups.items()
    }
    source_len_stats: dict[tuple[str, int], tuple[float, float]] = {}
    for key, idxs in source_len_groups.items():
        if len(idxs) >= int(cfg.normalize_min_group_size):
            source_len_stats[key] = _safe_mean_std(teacher_vals[idxs])

    normalized: list[Any] = []
    for idx, rec in enumerate(records):
        st = str(rec.source_type).strip().lower() or "unknown"
        lb = _length_bin(len(str(rec.sequence_aa)), int(cfg.normalize_length_bin_size))
        med, std = source_len_stats.get(
            (st, lb),
            source_stats.get(st, (global_med, global_std)),
        )
        z = (float(rec.teacher_global) - med) / std
        normalized.append(replace(rec, teacher_global=float(z)))
    return normalized


def _pearson_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() < 2 or b.numel() < 2:
        return torch.tensor(0.0, device=a.device)
    aa = a - a.mean()
    bb = b - b.mean()
    denom = torch.sqrt(torch.sum(aa * aa) * torch.sum(bb * bb))
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-8:
        return torch.tensor(0.0, device=a.device)
    corr = torch.sum(aa * bb) / denom
    return torch.clamp(corr, min=-1.0, max=1.0)


def _correlation_alignment_loss(
    *,
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
    source_is_chimera: torch.Tensor,
    cfg: StudentTrainConfig,
) -> torch.Tensor:
    if float(cfg.correlation_weight) <= 0.0:
        return torch.tensor(0.0, device=student_global.device)
    mask = torch.isfinite(student_global) & torch.isfinite(teacher_global)
    if bool(cfg.correlation_on_chimera_only):
        mask = mask & (source_is_chimera > 0.5)
    if int(mask.sum().item()) < 2:
        return torch.tensor(0.0, device=student_global.device)
    corr = _pearson_torch(student_global[mask], teacher_global[mask])
    return 1.0 - corr


def _sample_pair_indices(
    *,
    teacher_values: torch.Tensor,
    seq_lengths: torch.Tensor,
    max_pairs: int,
    min_teacher_diff: float,
    ignore_close_diff: float,
    hard_ratio: float,
    medium_ratio: float,
    easy_ratio: float,
    length_bin_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(teacher_values.shape[0])
    if n < 2 or max_pairs <= 0:
        empty = torch.empty(0, dtype=torch.long, device=teacher_values.device)
        return empty, empty, torch.empty(0, dtype=teacher_values.dtype, device=teacher_values.device)

    max_pairs = int(max_pairs)
    candidate_pairs = max(4 * max_pairs, 256)
    ii = torch.randint(0, n, (candidate_pairs,), device=teacher_values.device)
    jj = torch.randint(0, n, (candidate_pairs,), device=teacher_values.device)
    mask_diff_idx = ii != jj
    if mask_diff_idx.sum() == 0:
        empty = torch.empty(0, dtype=torch.long, device=teacher_values.device)
        return empty, empty, torch.empty(0, dtype=teacher_values.dtype, device=teacher_values.device)
    ii = ii[mask_diff_idx]
    jj = jj[mask_diff_idx]

    if int(length_bin_size) > 0:
        bsz = int(length_bin_size)
        len_bin_i = torch.div(seq_lengths[ii], bsz, rounding_mode="floor")
        len_bin_j = torch.div(seq_lengths[jj], bsz, rounding_mode="floor")
        same_bin = len_bin_i == len_bin_j
        if same_bin.sum() > 0:
            ii = ii[same_bin]
            jj = jj[same_bin]

    if ii.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=teacher_values.device)
        return empty, empty, torch.empty(0, dtype=teacher_values.dtype, device=teacher_values.device)

    diff = torch.abs(teacher_values[ii] - teacher_values[jj])
    valid = diff >= float(max(min_teacher_diff, ignore_close_diff))
    ii = ii[valid]
    jj = jj[valid]
    diff = diff[valid]
    if ii.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=teacher_values.device)
        return empty, empty, torch.empty(0, dtype=teacher_values.dtype, device=teacher_values.device)

    q33 = torch.quantile(diff, 0.33)
    q66 = torch.quantile(diff, 0.66)
    hard_idx = torch.nonzero(diff <= q33, as_tuple=False).squeeze(-1)
    med_idx = torch.nonzero((diff > q33) & (diff <= q66), as_tuple=False).squeeze(-1)
    easy_idx = torch.nonzero(diff > q66, as_tuple=False).squeeze(-1)

    hard_n = max(1, int(round(max_pairs * max(0.0, float(hard_ratio)))))
    med_n = max(1, int(round(max_pairs * max(0.0, float(medium_ratio)))))
    easy_n = max(1, int(round(max_pairs * max(0.0, float(easy_ratio)))))

    def _take(indices: torch.Tensor, k: int) -> torch.Tensor:
        if indices.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long, device=teacher_values.device)
        if indices.numel() <= k:
            return indices
        sel = torch.randperm(indices.numel(), device=teacher_values.device)[:k]
        return indices[sel]

    picked = torch.cat(
        [
            _take(hard_idx, hard_n),
            _take(med_idx, med_n),
            _take(easy_idx, easy_n),
        ],
        dim=0,
    )
    if picked.numel() == 0:
        picked = torch.randperm(ii.numel(), device=teacher_values.device)[: min(max_pairs, ii.numel())]
    else:
        if picked.numel() > max_pairs:
            sel = torch.randperm(picked.numel(), device=teacher_values.device)[:max_pairs]
            picked = picked[sel]
    return ii[picked], jj[picked], diff[picked]


def _pairwise_ranking_loss(
    *,
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
    source_is_chimera: torch.Tensor,
    seq_lengths: torch.Tensor | None = None,
    cfg: StudentTrainConfig,
) -> torch.Tensor:
    if float(cfg.pairwise_weight) <= 0.0:
        return torch.tensor(0.0, device=student_global.device)

    mask = torch.isfinite(student_global) & torch.isfinite(teacher_global)
    if bool(cfg.pairwise_on_chimera_only):
        mask = mask & (source_is_chimera > 0.5)

    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() < 2:
        return torch.tensor(0.0, device=student_global.device)

    t = teacher_global[valid_idx]
    s = student_global[valid_idx]
    if seq_lengths is None:
        l = torch.full_like(valid_idx, fill_value=1, dtype=torch.long)
    else:
        l = seq_lengths[valid_idx]
    ii, jj, teacher_diff = _sample_pair_indices(
        teacher_values=t,
        seq_lengths=l,
        max_pairs=int(cfg.pairwise_pairs_per_batch),
        min_teacher_diff=float(cfg.pairwise_min_teacher_diff),
        ignore_close_diff=float(cfg.pairwise_ignore_close_diff),
        hard_ratio=float(cfg.pairwise_hard_ratio),
        medium_ratio=float(cfg.pairwise_medium_ratio),
        easy_ratio=float(cfg.pairwise_easy_ratio),
        length_bin_size=int(cfg.pairwise_length_bin_size),
    )
    if ii.numel() == 0:
        return torch.tensor(0.0, device=student_global.device)

    teacher_sign = torch.sign(t[ii] - t[jj])
    nonzero = teacher_sign != 0
    if nonzero.sum() == 0:
        return torch.tensor(0.0, device=student_global.device)
    ii = ii[nonzero]
    jj = jj[nonzero]
    teacher_sign = teacher_sign[nonzero]
    teacher_diff = teacher_diff[nonzero]
    margin_term = teacher_sign * (s[ii] - s[jj]) - float(cfg.pairwise_margin)
    pair_loss = F.softplus(-margin_term)
    weight = 1.0 + torch.clamp(teacher_diff, min=0.0)
    return torch.mean(pair_loss * weight)


def _student_scores_from_token_ll(
    *,
    token_ll: torch.Tensor,  # [B, L]
    mask: torch.Tensor,  # [B, L]
    domain_lengths: list[list[int]],
    window: JunctionWindowConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = token_ll.shape[0]
    global_scores = []
    junction_scores = []

    token_ll_cpu = token_ll.detach().cpu()
    mask_cpu = mask.detach().cpu()
    for i in range(bsz):
        n = int(mask_cpu[i].sum().item())
        vals = token_ll_cpu[i, :n].tolist()
        if len(vals) == 0:
            global_scores.append(float("nan"))
            junction_scores.append([float("nan")] * 10)
            continue
        g = float(sum(vals) / len(vals))
        global_scores.append(g)

        per_residue = [None]
        if n > 1:
            per_residue.extend(float(x) for x in vals[1:])
        if n == 1:
            per_residue = [None]

        dlen = domain_lengths[i]
        if len(dlen) != 11:
            approx = [n // 11] * 11
            approx[-1] += max(0, n - sum(approx))
            dlen = approx
        boundaries = compute_boundary_positions(dlen)
        j = score_windows_from_per_residue_ll(per_residue, boundaries, window)
        if len(j) != 10:
            j = (j + [float("nan")] * 10)[:10]
        junction_scores.append(j)

    gs = torch.tensor(global_scores, dtype=torch.float32, device=token_ll.device)
    js = torch.tensor(junction_scores, dtype=torch.float32, device=token_ll.device)
    return gs, js


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    train_cfg: StudentTrainConfig,
    window: JunctionWindowConfig,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats = {
        "loss": 0.0,
        "nll": 0.0,
        "global_mse": 0.0,
        "junction_mse": 0.0,
        "corr_loss": 0.0,
        "pairwise_loss": 0.0,
        "batches": 0,
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask = batch["mask"].to(device)
        teacher_global = batch["teacher_global"].to(device)
        teacher_junctions = batch["teacher_junctions"].to(device)
        seq_lengths = torch.tensor(batch["lengths"], dtype=torch.long, device=device)
        source_is_chimera = batch.get("source_is_chimera")
        if source_is_chimera is None:
            source_is_chimera = torch.ones_like(teacher_global, dtype=torch.float32)
        source_is_chimera = source_is_chimera.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            token_ll = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            nll = -(token_ll[mask]).mean()

            student_global, student_junctions = _student_scores_from_token_ll(
                token_ll=token_ll,
                mask=mask,
                domain_lengths=batch["domain_lengths"],
                window=window,
            )
            global_weights = torch.where(
                source_is_chimera > 0.5,
                torch.tensor(float(train_cfg.chimera_global_weight), device=device),
                torch.tensor(float(train_cfg.natural_global_weight), device=device),
            )
            junction_weights = torch.where(
                source_is_chimera > 0.5,
                torch.tensor(float(train_cfg.chimera_junction_weight), device=device),
                torch.tensor(float(train_cfg.natural_junction_weight), device=device),
            )
            global_mse = _weighted_masked_mse(student_global, teacher_global, global_weights)
            junction_mse = _weighted_masked_mse(
                student_junctions, teacher_junctions, junction_weights
            )
            corr_loss = _correlation_alignment_loss(
                student_global=student_global,
                teacher_global=teacher_global,
                source_is_chimera=source_is_chimera,
                cfg=train_cfg,
            )
            pairwise_loss = _pairwise_ranking_loss(
                student_global=student_global,
                teacher_global=teacher_global,
                source_is_chimera=source_is_chimera,
                seq_lengths=seq_lengths,
                cfg=train_cfg,
            )

            loss = (
                train_cfg.nll_weight * nll
                + train_cfg.global_weight * global_mse
                + train_cfg.junction_weight * junction_mse
                + train_cfg.correlation_weight * corr_loss
                + train_cfg.pairwise_weight * pairwise_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

        stats["loss"] += float(loss.item())
        stats["nll"] += float(nll.item())
        stats["global_mse"] += float(global_mse.item())
        stats["junction_mse"] += float(junction_mse.item())
        stats["corr_loss"] += float(corr_loss.item())
        stats["pairwise_loss"] += float(pairwise_loss.item())
        stats["batches"] += 1

    if stats["batches"] > 0:
        for k in ("loss", "nll", "global_mse", "junction_mse", "corr_loss", "pairwise_loss"):
            stats[k] /= stats["batches"]
    return stats


def _topk_overlap_fraction(
    true_score: np.ndarray,
    pred_score: np.ndarray,
    frac: float,
) -> float:
    mask = np.isfinite(true_score) & np.isfinite(pred_score)
    if mask.sum() < 2:
        return float("nan")
    t = true_score[mask]
    p = pred_score[mask]
    n = int(t.shape[0])
    k = max(1, int(round(n * float(frac))))
    k = min(k, n)
    top_true = np.argsort(t)[::-1][:k]
    top_pred = np.argsort(p)[::-1][:k]
    overlap = len(set(top_true.tolist()) & set(top_pred.tolist()))
    return float(overlap / max(1, k))


def _pairwise_order_accuracy_np(
    true_score: np.ndarray,
    pred_score: np.ndarray,
    min_diff: float,
) -> float:
    mask = np.isfinite(true_score) & np.isfinite(pred_score)
    t = true_score[mask]
    p = pred_score[mask]
    n = int(t.shape[0])
    if n < 2:
        return float("nan")
    max_pairs = min(20000, n * (n - 1) // 2)
    if max_pairs <= 0:
        return float("nan")
    rng = np.random.default_rng(13)
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    valid = i != j
    i = i[valid]
    j = j[valid]
    td = t[i] - t[j]
    keep = np.abs(td) >= float(min_diff)
    if not keep.any():
        return float("nan")
    i = i[keep]
    j = j[keep]
    td = td[keep]
    pd = p[i] - p[j]
    acc = np.mean(np.sign(td) == np.sign(pd))
    return float(acc)


@torch.no_grad()
def _evaluate_regression_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: str,
    window: JunctionWindowConfig,
    topk_fracs: Iterable[float],
    pairwise_min_teacher_diff: float,
) -> dict[str, float]:
    model.eval()
    global_pred = []
    global_true = []
    jmean_pred = []
    jmean_true = []
    source_type = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask = batch["mask"].to(device)
        teacher_global = batch["teacher_global"].cpu().tolist()
        teacher_junctions = batch["teacher_junctions"].cpu().numpy()

        logits = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        s_global, s_junction = _student_scores_from_token_ll(
            token_ll=token_ll, mask=mask, domain_lengths=batch["domain_lengths"], window=window
        )
        s_global = s_global.cpu().tolist()
        s_junction = s_junction.cpu().numpy()
        source_batch = [str(x).strip().lower() for x in batch.get("source_type", [])]
        if len(source_batch) != len(s_global):
            source_batch = ["unknown"] * len(s_global)

        for i in range(len(s_global)):
            global_pred.append(float(s_global[i]))
            global_true.append(float(teacher_global[i]))
            jmean_pred.append(_nanmean_np(s_junction[i].tolist()))
            jmean_true.append(_nanmean_np(teacher_junctions[i].tolist()))
            source_type.append(source_batch[i])

    g_true = np.asarray(global_true, dtype=np.float64)
    g_pred = np.asarray(global_pred, dtype=np.float64)
    g_mask = np.isfinite(g_true) & np.isfinite(g_pred)

    jm_true = np.asarray(jmean_true, dtype=np.float64)
    jm_pred = np.asarray(jmean_pred, dtype=np.float64)
    jm_mask = np.isfinite(jm_true) & np.isfinite(jm_pred)

    metrics = {
        "global_corr": _pearson_np(g_true[g_mask], g_pred[g_mask]) if g_mask.any() else float("nan"),
        "global_mae": float(np.mean(np.abs(g_true[g_mask] - g_pred[g_mask]))) if g_mask.any() else float("nan"),
        "global_mse": float(np.mean((g_true[g_mask] - g_pred[g_mask]) ** 2)) if g_mask.any() else float("nan"),
        "junction_mean_corr": _pearson_np(jm_true[jm_mask], jm_pred[jm_mask])
        if jm_mask.any()
        else float("nan"),
        "junction_mean_mae": float(np.mean(np.abs(jm_true[jm_mask] - jm_pred[jm_mask])))
        if jm_mask.any()
        else float("nan"),
        "junction_mean_mse": float(np.mean((jm_true[jm_mask] - jm_pred[jm_mask]) ** 2))
        if jm_mask.any()
        else float("nan"),
    }

    source_arr = np.asarray(source_type, dtype=object)
    for st in ("natural", "chimera"):
        st_mask = source_arr == st
        st_global_mask = st_mask & g_mask
        st_jm_mask = st_mask & jm_mask
        metrics[f"n_{st}"] = int(st_mask.sum())
        metrics[f"global_corr_{st}"] = (
            _pearson_np(g_true[st_global_mask], g_pred[st_global_mask])
            if st_global_mask.any()
            else float("nan")
        )
        metrics[f"global_mae_{st}"] = (
            float(np.mean(np.abs(g_true[st_global_mask] - g_pred[st_global_mask])))
            if st_global_mask.any()
            else float("nan")
        )
        metrics[f"global_mse_{st}"] = (
            float(np.mean((g_true[st_global_mask] - g_pred[st_global_mask]) ** 2))
            if st_global_mask.any()
            else float("nan")
        )
        metrics[f"junction_mean_corr_{st}"] = (
            _pearson_np(jm_true[st_jm_mask], jm_pred[st_jm_mask])
            if st_jm_mask.any()
            else float("nan")
        )

    chimera_mask = source_arr == "chimera"
    chimera_true = g_true[chimera_mask]
    chimera_pred = g_pred[chimera_mask]
    for frac in topk_fracs:
        frac = float(frac)
        if frac <= 0:
            continue
        pct = int(round(frac * 100))
        metrics[f"topk_overlap_chimera_{pct}pct"] = _topk_overlap_fraction(
            chimera_true, chimera_pred, frac
        )
    metrics["hard_pair_acc_chimera"] = _pairwise_order_accuracy_np(
        chimera_true, chimera_pred, min_diff=float(pairwise_min_teacher_diff)
    )
    return metrics


def train_student_from_distill_csv(
    *,
    distill_csv: str,
    validated_domains: dict[tuple[str, int], str] | None,
    model_cfg: StudentModelConfig,
    train_cfg: StudentTrainConfig,
    window: JunctionWindowConfig,
    out_dir: str,
) -> dict[str, Any]:
    _set_seed(train_cfg.seed)
    _set_cpu_threads(train_cfg)
    device = detect_torch_device(train_cfg.device)
    logger.info("Student device: %s", device)

    all_records = load_distill_records_from_csv(
        csv_path=distill_csv,
        validated_domains=validated_domains,
    )
    all_records = _normalize_teacher_global_scores(all_records, train_cfg)
    vocab: AminoAcidVocab = build_default_vocab()

    def _build_loaders(
        recs: Sequence[Any],
        *,
        batch_size: int,
        balance_source_types: bool,
    ) -> tuple[DistillDataset, list[int], list[int], DataLoader, DataLoader, Counter, Counter]:
        if len(recs) < 2:
            raise ValueError("Need at least 2 distill examples in selected stage")
        dataset_local = DistillDataset(recs, vocab=vocab)
        source_labels_local = [str(r.source_type).strip().lower() for r in recs]
        train_idx_local, val_idx_local = split_indices(
            len(dataset_local),
            val_fraction=train_cfg.val_fraction,
            seed=train_cfg.seed,
            labels=source_labels_local,
        )
        train_counts = Counter(source_labels_local[i] for i in train_idx_local)
        val_counts = Counter(source_labels_local[i] for i in val_idx_local)
        train_subset_local = Subset(dataset_local, train_idx_local)
        train_sampler_local = None
        train_shuffle_local = True
        if bool(balance_source_types):
            train_labels = [source_labels_local[i] for i in train_idx_local]
            label_counts = Counter(train_labels)
            sample_weights = [1.0 / max(1, label_counts[lb]) for lb in train_labels]
            train_sampler_local = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_subset_local),
                replacement=True,
            )
            train_shuffle_local = False
            logger.info("Enabled balanced source sampler: counts=%s", dict(label_counts))
        train_loader_local = DataLoader(
            train_subset_local,
            batch_size=int(batch_size),
            shuffle=train_shuffle_local,
            sampler=train_sampler_local,
            num_workers=train_cfg.num_workers,
            collate_fn=partial(collate_distill_batch, pad_id=vocab.pad_id),
        )
        val_loader_local = DataLoader(
            Subset(dataset_local, val_idx_local),
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=train_cfg.num_workers,
            collate_fn=partial(collate_distill_batch, pad_id=vocab.pad_id),
        )
        return (
            dataset_local,
            train_idx_local,
            val_idx_local,
            train_loader_local,
            val_loader_local,
            train_counts,
            val_counts,
        )

    natural_records = [r for r in all_records if str(r.source_type).strip().lower() == "natural"]
    chimera_records = [r for r in all_records if str(r.source_type).strip().lower() == "chimera"]
    if bool(train_cfg.stage_b_chimera_only):
        stage_b_records = chimera_records
    else:
        stage_b_records = all_records

    model_type = _normalize_model_type(model_cfg.model_type)
    model = build_student_model(
        cfg=model_cfg,
        vocab_size=vocab.size,
        pad_idx=vocab.pad_id,
    ).to(device)
    logger.info("Student model_type=%s", model_type)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    history_rows = []
    best_val_loss = float("inf")
    best_metric_name = str(train_cfg.best_metric).strip() or "val_loss"
    best_metric_mode = str(train_cfg.best_metric_mode).strip().lower() or "auto"
    maximize_metric = (
        any(
            key in best_metric_name.lower()
            for key in ("corr", "overlap", "acc", "auc", "spearman", "pearson")
        )
        if best_metric_mode == "auto"
        else best_metric_mode == "max"
    )
    best_metric_value = -float("inf") if maximize_metric else float("inf")
    best_epoch = -1
    total_epochs = int(train_cfg.epochs)

    if int(train_cfg.stage_a_natural_epochs) > 0 and len(natural_records) >= 2:
        stage_a_batch_size = int(train_cfg.stage_a_batch_size or train_cfg.batch_size)
        (
            stage_a_dataset,
            stage_a_train_idx,
            stage_a_val_idx,
            stage_a_train_loader,
            stage_a_val_loader,
            stage_a_train_counts,
            stage_a_val_counts,
        ) = _build_loaders(
            natural_records,
            batch_size=stage_a_batch_size,
            balance_source_types=False,
        )
        logger.info(
            "Stage A natural-only split: train=%d val=%d train_sources=%s val_sources=%s",
            len(stage_a_train_idx),
            len(stage_a_val_idx),
            dict(stage_a_train_counts),
            dict(stage_a_val_counts),
        )
        stage_a_cfg = StudentTrainConfig(
            **{
                **train_cfg.__dict__,
                "nll_weight": 1.0,
                "global_weight": 0.0,
                "junction_weight": 0.0,
                "correlation_weight": 0.0,
                "pairwise_weight": 0.0,
                "balance_source_types": False,
            }
        )
        for stage_a_epoch in range(1, int(train_cfg.stage_a_natural_epochs) + 1):
            train_stats = _run_epoch(
                model=model,
                loader=stage_a_train_loader,
                optimizer=optimizer,
                device=device,
                train_cfg=stage_a_cfg,
                window=window,
            )
            val_stats = _run_epoch(
                model=model,
                loader=stage_a_val_loader,
                optimizer=None,
                device=device,
                train_cfg=stage_a_cfg,
                window=window,
            )
            row = {
                "stage": "A",
                "stage_epoch": stage_a_epoch,
                "epoch": stage_a_epoch,
                "train_loss": train_stats["loss"],
                "train_nll": train_stats["nll"],
                "train_global_mse": train_stats["global_mse"],
                "train_junction_mse": train_stats["junction_mse"],
                "train_corr_loss": train_stats["corr_loss"],
                "train_pairwise_loss": train_stats["pairwise_loss"],
                "val_loss": val_stats["loss"],
                "val_nll": val_stats["nll"],
                "val_global_mse": val_stats["global_mse"],
                "val_junction_mse": val_stats["junction_mse"],
                "val_corr_loss": val_stats["corr_loss"],
                "val_pairwise_loss": val_stats["pairwise_loss"],
            }
            history_rows.append(row)
            logger.info(
                "Stage A epoch %d/%d train_loss=%.4f val_loss=%.4f",
                stage_a_epoch,
                int(train_cfg.stage_a_natural_epochs),
                row["train_loss"],
                row["val_loss"],
            )

    (
        stage_b_dataset,
        train_idx,
        val_idx,
        train_loader,
        val_loader,
        train_source_counts,
        val_source_counts,
    ) = _build_loaders(
        stage_b_records,
        batch_size=train_cfg.batch_size,
        balance_source_types=bool(train_cfg.balance_source_types),
    )
    logger.info(
        "Stage B distill split: train=%d val=%d train_sources=%s val_sources=%s",
        len(train_idx),
        len(val_idx),
        dict(train_source_counts),
        dict(val_source_counts),
    )

    for epoch in range(1, total_epochs + 1):
        warmup_pairwise_off = epoch <= int(train_cfg.pairwise_warmup_epochs)
        effective_cfg = train_cfg
        if warmup_pairwise_off and float(train_cfg.pairwise_weight) > 0.0:
            effective_cfg = StudentTrainConfig(
                **{**train_cfg.__dict__, "pairwise_weight": 0.0}
            )
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_cfg=effective_cfg,
            window=window,
        )
        val_stats = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            train_cfg=effective_cfg,
            window=window,
        )
        reg_metrics = _evaluate_regression_metrics(
            model=model,
            loader=val_loader,
            device=device,
            window=window,
            topk_fracs=train_cfg.topk_fracs,
            pairwise_min_teacher_diff=train_cfg.pairwise_min_teacher_diff,
        )

        row = {
            "stage": "B",
            "stage_epoch": epoch,
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_nll": train_stats["nll"],
            "train_global_mse": train_stats["global_mse"],
            "train_junction_mse": train_stats["junction_mse"],
            "train_corr_loss": train_stats["corr_loss"],
            "train_pairwise_loss": train_stats["pairwise_loss"],
            "val_loss": val_stats["loss"],
            "val_nll": val_stats["nll"],
            "val_global_mse": val_stats["global_mse"],
            "val_junction_mse": val_stats["junction_mse"],
            "val_corr_loss": val_stats["corr_loss"],
            "val_pairwise_loss": val_stats["pairwise_loss"],
            "pairwise_weight_effective": float(effective_cfg.pairwise_weight),
            **reg_metrics,
        }
        history_rows.append(row)
        logger.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f global_corr=%.4f",
            epoch,
            total_epochs,
            row["train_loss"],
            row["val_loss"],
            row["global_corr"] if math.isfinite(row["global_corr"]) else float("nan"),
        )

        last_ckpt = out_path / "student_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "model_type": model_type,
                "vocab_stoi": vocab.stoi,
                "epoch": epoch,
            },
            last_ckpt,
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
        metric_value = row.get(best_metric_name, float("nan"))
        metric_valid = isinstance(metric_value, (float, int)) and math.isfinite(float(metric_value))
        improved = False
        if metric_valid:
            metric_value = float(metric_value)
            improved = metric_value > best_metric_value if maximize_metric else metric_value < best_metric_value
        elif best_epoch < 0:
            metric_value = float(row["val_loss"])
            improved = True
        if improved:
            best_metric_value = float(metric_value)
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "model_type": model_type,
                    "vocab_stoi": vocab.stoi,
                    "epoch": epoch,
                },
                out_path / "student_best.pt",
            )

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(out_path / "train_history.csv", index=False)
    final_metrics = history_rows[-1].copy() if len(history_rows) > 0 else {}
    summary = {
        "distill_csv": distill_csv,
        "n_records": len(all_records),
        "n_records_stage_b": len(stage_b_dataset),
        "n_records_chimera": len(chimera_records),
        "n_records_natural": len(natural_records),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metric": best_metric_name,
        "best_metric_mode": "max" if maximize_metric else "min",
        "best_metric_value": best_metric_value,
        "device": device,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "window": asdict(window),
        "final_metrics": final_metrics,
    }
    (out_path / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary
