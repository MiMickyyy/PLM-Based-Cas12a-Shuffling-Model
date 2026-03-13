from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

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
    stats = {"loss": 0.0, "nll": 0.0, "global_mse": 0.0, "junction_mse": 0.0, "batches": 0}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask = batch["mask"].to(device)
        teacher_global = batch["teacher_global"].to(device)
        teacher_junctions = batch["teacher_junctions"].to(device)
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

            loss = (
                train_cfg.nll_weight * nll
                + train_cfg.global_weight * global_mse
                + train_cfg.junction_weight * junction_mse
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
        stats["batches"] += 1

    if stats["batches"] > 0:
        for k in ("loss", "nll", "global_mse", "junction_mse"):
            stats[k] /= stats["batches"]
    return stats


@torch.no_grad()
def _evaluate_regression_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: str,
    window: JunctionWindowConfig,
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

    records = load_distill_records_from_csv(
        csv_path=distill_csv,
        validated_domains=validated_domains,
    )
    vocab: AminoAcidVocab = build_default_vocab()
    dataset = DistillDataset(records, vocab=vocab)
    source_labels = [str(r.source_type).strip().lower() for r in records]
    train_idx, val_idx = split_indices(
        len(dataset),
        val_fraction=train_cfg.val_fraction,
        seed=train_cfg.seed,
        labels=source_labels,
    )

    train_source_counts = Counter(source_labels[i] for i in train_idx)
    val_source_counts = Counter(source_labels[i] for i in val_idx)
    logger.info(
        "Student split: train=%d val=%d train_sources=%s val_sources=%s",
        len(train_idx),
        len(val_idx),
        dict(train_source_counts),
        dict(val_source_counts),
    )

    train_subset = Subset(dataset, train_idx)
    train_sampler = None
    train_shuffle = True
    if bool(train_cfg.balance_source_types):
        train_labels = [source_labels[i] for i in train_idx]
        label_counts = Counter(train_labels)
        sample_weights = [1.0 / max(1, label_counts[lb]) for lb in train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_subset),
            replacement=True,
        )
        train_shuffle = False
        logger.info("Enabled balanced source sampler: counts=%s", dict(label_counts))

    train_loader = DataLoader(
        train_subset,
        batch_size=train_cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=partial(collate_distill_batch, pad_id=vocab.pad_id),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=partial(collate_distill_batch, pad_id=vocab.pad_id),
    )

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
    best_epoch = -1

    for epoch in range(1, train_cfg.epochs + 1):
        train_stats = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_cfg=train_cfg,
            window=window,
        )
        val_stats = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            train_cfg=train_cfg,
            window=window,
        )
        reg_metrics = _evaluate_regression_metrics(
            model=model, loader=val_loader, device=device, window=window
        )

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_nll": train_stats["nll"],
            "train_global_mse": train_stats["global_mse"],
            "train_junction_mse": train_stats["junction_mse"],
            "val_loss": val_stats["loss"],
            "val_nll": val_stats["nll"],
            "val_global_mse": val_stats["global_mse"],
            "val_junction_mse": val_stats["junction_mse"],
            **reg_metrics,
        }
        history_rows.append(row)
        logger.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f global_corr=%.4f",
            epoch,
            train_cfg.epochs,
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
    final_metrics = history_rows[-1].copy()
    summary = {
        "distill_csv": distill_csv,
        "n_records": len(dataset),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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
