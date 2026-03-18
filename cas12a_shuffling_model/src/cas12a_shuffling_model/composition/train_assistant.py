from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from cas12a_shuffling_model.composition.assistant_ranker import (
    AssistantModelConfig,
    AssistantRanker,
    build_assistant_checkpoint_payload,
    detect_torch_device,
)
from cas12a_shuffling_model.composition.chimera_repr import (
    SLOT_COUNT,
    canonicalize_chimera_table,
    load_active_code_counts,
    slot_columns,
)
from cas12a_shuffling_model.composition.ranking_losses import (
    active_vs_background_rank_loss,
    correlation_loss,
    filtered_pairwise_rank_loss,
    is_better_metric,
    active_local_margin_loss,
    top_tail_active_loss,
    ranking_metrics,
    top_focus_loss,
)
from cas12a_shuffling_model.composition.table_io import read_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssistantTrainConfig:
    seed: int = 13
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    top_weight: float = 1.0
    corr_weight: float = 0.3
    pair_weight: float = 0.2
    objective_mode: str = "legacy"
    local_target_col: str | None = "active_local_target"
    hard_negative_col: str = "is_hard_negative"
    local_distance_col: str = "local_active_distance"
    local_max_distance: int = 4
    active_local_weight: float = 1.0
    top_tail_weight: float = 0.5
    pair_local_weight: float = 0.2
    teacher_corr_weight: float = 0.1
    active_anchor_repeat: int = 24
    active_local_pairs_per_batch: int = 256
    top_tail_pairs_per_batch: int = 256
    active_local_margin: float = 0.2
    top_tail_margin: float = 0.1
    pair_local_pairs_per_batch: int = 512
    pair_local_min_gap: float = 0.0
    pair_local_near_tie_gap: float = 0.0
    pair_min_gap: float = 0.01
    pair_near_tie_gap: float = 0.01
    pair_easy_ratio: float = 0.2
    pair_medium_ratio: float = 0.5
    pair_hard_ratio: float = 0.3
    pair_pairs_per_batch: int = 512
    pair_margin: float = 0.0
    pair_margin_alpha: float = 0.5
    pair_margin_min: float = 0.0
    pair_margin_max: float = 0.3
    top_fraction: float = 0.10
    top_pairs_per_batch: int = 256
    top_margin: float = 0.0
    chimera_only: bool = True
    target_col: str = "teacher_seq_score_norm"
    experimental_label_col: str | None = None
    experimental_weight: float = 0.0
    oversample_top_fraction: float = 0.10
    oversample_weight: float = 2.0
    active_codes_path: str | None = None
    active_sample_weight: float = 1.0
    active_force_train: bool = True
    active_loss_weight: float = 0.0
    active_pairs_per_batch: int = 256
    active_margin: float = 0.10
    active_min_target_gap: float = 0.0
    best_metric: str = "global_corr_chimera"
    num_workers: int = 0
    device: str | None = None
    cpu_threads: int | None = None
    interop_threads: int | None = None


class _AssistantDataset(Dataset):
    def __init__(
        self,
        slots: np.ndarray,
        extra: np.ndarray,
        target: np.ndarray,
        local_target: np.ndarray,
        exp: np.ndarray,
        is_active: np.ndarray,
        is_hard_negative: np.ndarray,
        local_distance: np.ndarray,
    ):
        self.slots = slots
        self.extra = extra
        self.target = target
        self.local_target = local_target
        self.exp = exp
        self.is_active = is_active
        self.is_hard_negative = is_hard_negative
        self.local_distance = local_distance

    def __len__(self) -> int:
        return int(self.slots.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "slots": torch.tensor(self.slots[idx], dtype=torch.long),
            "extra": torch.tensor(self.extra[idx], dtype=torch.float32),
            "target": torch.tensor(self.target[idx], dtype=torch.float32),
            "local_target": torch.tensor(self.local_target[idx], dtype=torch.float32),
            "exp": torch.tensor(self.exp[idx], dtype=torch.float32),
            "is_active": torch.tensor(self.is_active[idx], dtype=torch.float32),
            "is_hard_negative": torch.tensor(self.is_hard_negative[idx], dtype=torch.float32),
            "local_distance": torch.tensor(self.local_distance[idx], dtype=torch.float32),
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(cfg: AssistantTrainConfig) -> None:
    if cfg.cpu_threads is not None and int(cfg.cpu_threads) > 0:
        torch.set_num_threads(int(cfg.cpu_threads))
    if cfg.interop_threads is not None and int(cfg.interop_threads) > 0:
        try:
            torch.set_num_interop_threads(int(cfg.interop_threads))
        except RuntimeError:
            pass


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    n_val = max(1, min(n - 1, int(round(n * float(val_fraction)))))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def _prepare_arrays(
    df: pd.DataFrame,
    *,
    target_col: str,
    local_target_col: str | None,
    feature_cols: Sequence[str],
    experimental_col: str | None,
    is_active: np.ndarray | None = None,
    is_hard_negative: np.ndarray | None = None,
    local_distance: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical = canonicalize_chimera_table(df, require_sequence=False)
    slots = canonical[slot_columns()].to_numpy(dtype=np.int64)
    y = pd.to_numeric(canonical[target_col], errors="coerce").to_numpy(dtype=np.float32)
    if len(feature_cols) > 0:
        ex = canonical.reindex(columns=list(feature_cols), fill_value=np.nan).to_numpy(dtype=np.float32)
        ex = np.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        ex = np.zeros((len(canonical), 0), dtype=np.float32)
    if experimental_col and experimental_col in canonical.columns:
        exp = pd.to_numeric(canonical[experimental_col], errors="coerce").to_numpy(dtype=np.float32)
    else:
        exp = np.full((len(canonical),), np.nan, dtype=np.float32)
    if local_target_col and local_target_col in canonical.columns:
        local_target = pd.to_numeric(canonical[local_target_col], errors="coerce").to_numpy(dtype=np.float32)
    else:
        local_target = y.copy()
    if is_active is None:
        active = np.zeros((len(canonical),), dtype=np.float32)
    else:
        active = np.asarray(is_active, dtype=np.float32).reshape(-1)
        if active.shape[0] != len(canonical):
            raise ValueError("is_active length mismatch")
    if is_hard_negative is None:
        hard_neg = np.zeros((len(canonical),), dtype=np.float32)
    else:
        hard_neg = np.asarray(is_hard_negative, dtype=np.float32).reshape(-1)
        if hard_neg.shape[0] != len(canonical):
            raise ValueError("is_hard_negative length mismatch")
    if local_distance is None:
        local_dist = np.full((len(canonical),), np.nan, dtype=np.float32)
    else:
        local_dist = np.asarray(local_distance, dtype=np.float32).reshape(-1)
        if local_dist.shape[0] != len(canonical):
            raise ValueError("local_distance length mismatch")

    if train_idx is None or ex.shape[1] == 0:
        mean = np.zeros((ex.shape[1],), dtype=np.float32)
        std = np.ones((ex.shape[1],), dtype=np.float32)
    else:
        train_extra = ex[train_idx]
        mean = np.mean(train_extra, axis=0).astype(np.float32)
        std = np.std(train_extra, axis=0).astype(np.float32)
        std = np.where(np.abs(std) < 1e-8, 1.0, std)
        ex = (ex - mean) / std
    return slots, ex, y, local_target, exp, active, hard_neg, local_dist, mean, std


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    cfg: AssistantTrainConfig,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats = {
        "loss": 0.0,
        "top": 0.0,
        "corr": 0.0,
        "pair": 0.0,
        "exp": 0.0,
        "active": 0.0,
        "local_active": 0.0,
        "top_tail": 0.0,
        "pair_local": 0.0,
        "teacher_corr": 0.0,
        "batches": 0,
    }
    mode = str(cfg.objective_mode).strip().lower()
    for batch in loader:
        slots = batch["slots"].to(device)
        extra = batch["extra"].to(device) if batch["extra"].shape[1] > 0 else None
        target = batch["target"].to(device)
        local_target = batch["local_target"].to(device)
        exp = batch["exp"].to(device)
        is_active = batch["is_active"].to(device)
        is_hard_negative = batch["is_hard_negative"].to(device)
        local_distance = batch["local_distance"].to(device)
        with torch.set_grad_enabled(is_train):
            if bool(getattr(model, "cfg", None) and getattr(model.cfg, "dual_head", False)):
                _, pred_teacher, pred_active = model(slots, extra, return_heads=True)
            else:
                pred_active = model(slots, extra)
                pred_teacher = pred_active
            pred = pred_active

            l_top = torch.tensor(0.0, device=device)
            l_corr = torch.tensor(0.0, device=device)
            l_pair = torch.tensor(0.0, device=device)
            l_local_active = torch.tensor(0.0, device=device)
            l_top_tail = torch.tensor(0.0, device=device)
            l_pair_local = torch.tensor(0.0, device=device)
            l_teacher_corr = torch.tensor(0.0, device=device)

            if mode == "active_first":
                l_local_active = active_local_margin_loss(
                    pred=pred_active,
                    is_active=is_active,
                    is_hard_negative=is_hard_negative,
                    local_distance=local_distance,
                    max_local_distance=int(cfg.local_max_distance),
                    pairs_per_batch=int(cfg.active_local_pairs_per_batch),
                    margin=float(cfg.active_local_margin),
                )
                l_top_tail = top_tail_active_loss(
                    pred=pred_active,
                    is_active=is_active,
                    is_hard_negative=is_hard_negative,
                    local_distance=local_distance,
                    max_local_distance=int(cfg.local_max_distance),
                    pairs_per_batch=int(cfg.top_tail_pairs_per_batch),
                    margin=float(cfg.top_tail_margin),
                )
                local_mask = (
                    (is_active > 0.5)
                    | (is_hard_negative > 0.5)
                    | (
                        torch.isfinite(local_distance)
                        & (local_distance >= 1.0)
                        & (local_distance <= float(cfg.local_max_distance))
                    )
                )
                if int(local_mask.sum().item()) >= 4:
                    l_pair_local = filtered_pairwise_rank_loss(
                        pred=pred_active[local_mask],
                        target=local_target[local_mask],
                        pairs_per_batch=int(cfg.pair_local_pairs_per_batch),
                        min_gap=float(cfg.pair_local_min_gap),
                        near_tie_gap=float(cfg.pair_local_near_tie_gap),
                        easy_ratio=cfg.pair_easy_ratio,
                        medium_ratio=cfg.pair_medium_ratio,
                        hard_ratio=cfg.pair_hard_ratio,
                        base_margin=cfg.pair_margin,
                        margin_alpha=cfg.pair_margin_alpha,
                        margin_min=cfg.pair_margin_min,
                        margin_max=cfg.pair_margin_max,
                    )
                l_teacher_corr = correlation_loss(pred_teacher, target)
                loss = (
                    float(cfg.active_local_weight) * l_local_active
                    + float(cfg.top_tail_weight) * l_top_tail
                    + float(cfg.pair_local_weight) * l_pair_local
                    + float(cfg.teacher_corr_weight) * l_teacher_corr
                )
            else:
                l_top = top_focus_loss(
                    pred=pred,
                    target=target,
                    top_fraction=cfg.top_fraction,
                    pairs_per_batch=cfg.top_pairs_per_batch,
                    margin=cfg.top_margin,
                )
                l_corr = correlation_loss(pred, target)
                l_pair = filtered_pairwise_rank_loss(
                    pred=pred,
                    target=target,
                    pairs_per_batch=cfg.pair_pairs_per_batch,
                    min_gap=cfg.pair_min_gap,
                    near_tie_gap=cfg.pair_near_tie_gap,
                    easy_ratio=cfg.pair_easy_ratio,
                    medium_ratio=cfg.pair_medium_ratio,
                    hard_ratio=cfg.pair_hard_ratio,
                    base_margin=cfg.pair_margin,
                    margin_alpha=cfg.pair_margin_alpha,
                    margin_min=cfg.pair_margin_min,
                    margin_max=cfg.pair_margin_max,
                )
                loss = (
                    cfg.top_weight * l_top
                    + cfg.corr_weight * l_corr
                    + cfg.pair_weight * l_pair
                )
            l_exp = torch.tensor(0.0, device=device)
            if float(cfg.experimental_weight) > 0.0:
                mask = torch.isfinite(exp)
                if int(mask.sum().item()) > 0:
                    l_exp = torch.mean((pred_active[mask] - exp[mask]) ** 2)
            l_active = torch.tensor(0.0, device=device)
            if mode != "active_first" and float(cfg.active_loss_weight) > 0.0:
                l_active = active_vs_background_rank_loss(
                    pred=pred,
                    target=target,
                    is_active=is_active,
                    pairs_per_batch=int(cfg.active_pairs_per_batch),
                    margin=float(cfg.active_margin),
                    min_target_gap=float(cfg.active_min_target_gap),
                )
            if mode == "active_first":
                loss = loss + cfg.experimental_weight * l_exp
            else:
                loss = loss + cfg.experimental_weight * l_exp + cfg.active_loss_weight * l_active
            if is_train and (not loss.requires_grad):
                loss = pred_active.sum() * 0.0
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
        stats["loss"] += float(loss.item())
        stats["top"] += float(l_top.item())
        stats["corr"] += float(l_corr.item())
        stats["pair"] += float(l_pair.item())
        stats["exp"] += float(l_exp.item())
        stats["active"] += float(l_active.item())
        stats["local_active"] += float(l_local_active.item())
        stats["top_tail"] += float(l_top_tail.item())
        stats["pair_local"] += float(l_pair_local.item())
        stats["teacher_corr"] += float(l_teacher_corr.item())
        stats["batches"] += 1
    if stats["batches"] > 0:
        for k in ("loss", "top", "corr", "pair", "exp", "active", "local_active", "top_tail", "pair_local", "teacher_corr"):
            stats[k] /= stats["batches"]
    return stats


@torch.no_grad()
def _predict_loader(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        slots = batch["slots"].to(device)
        extra = batch["extra"].to(device) if batch["extra"].shape[1] > 0 else None
        target = batch["target"].numpy()
        pred = model(slots, extra).detach().cpu().numpy()
        y_true.append(target)
        y_pred.append(pred)
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


def train_assistant_ranker(
    *,
    data_table: str,
    out_dir: str,
    model_cfg: AssistantModelConfig,
    train_cfg: AssistantTrainConfig,
    feature_cols: Sequence[str],
) -> dict[str, Any]:
    _set_seed(train_cfg.seed)
    _set_cpu_threads(train_cfg)
    device = detect_torch_device(train_cfg.device)
    logger.info("Assistant device: %s", device)

    df = read_table(data_table)
    if bool(train_cfg.chimera_only) and "source_type" in df.columns:
        src = df["source_type"]
        src_str = src.astype(str).str.strip().str.lower()
        keep = src.isna() | src_str.isin({"", "nan", "chimera", "active"})
        df = df[keep].copy()
    if len(df) < 10:
        raise ValueError("Not enough rows for assistant training")
    df = canonicalize_chimera_table(df, require_sequence=False)
    if train_cfg.target_col not in df.columns:
        raise KeyError(f"Missing target column: {train_cfg.target_col}")
    target_numeric = pd.to_numeric(df[train_cfg.target_col], errors="coerce")
    mode = str(train_cfg.objective_mode).strip().lower()
    if mode == "active_first":
        has_any_teacher = int(np.isfinite(target_numeric.to_numpy()).sum()) >= 2
        if not has_any_teacher:
            logger.warning(
                "Active-first mode: no finite teacher targets found in %s, teacher regularizer will be skipped",
                train_cfg.target_col,
            )
    else:
        df = df[target_numeric.notna()].reset_index(drop=True)
        if len(df) < 10:
            raise ValueError("No finite targets after filtering")

    active_strength = np.zeros((len(df),), dtype=np.float32)
    if "is_active" in df.columns:
        col_active = pd.to_numeric(df["is_active"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        active_strength = np.maximum(active_strength, (col_active > 0.0).astype(np.float32))
    hard_negative = np.zeros((len(df),), dtype=np.float32)
    if train_cfg.hard_negative_col in df.columns:
        col_hn = pd.to_numeric(df[train_cfg.hard_negative_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        hard_negative = np.maximum(hard_negative, (col_hn > 0.0).astype(np.float32))
    local_distance = np.full((len(df),), np.nan, dtype=np.float32)
    if train_cfg.local_distance_col in df.columns:
        local_distance = pd.to_numeric(df[train_cfg.local_distance_col], errors="coerce").to_numpy(dtype=np.float32)
    active_counts: dict[str, int] = {}
    if train_cfg.active_codes_path:
        try:
            active_counts = load_active_code_counts(train_cfg.active_codes_path)
        except FileNotFoundError:
            logger.warning("Active weighting table not found: %s", train_cfg.active_codes_path)
            active_counts = {}
        if len(active_counts) > 0:
            combo = df["combo_compact"].astype(str).to_numpy()
            for i, code in enumerate(combo.tolist()):
                cnt = int(active_counts.get(code, 0))
                if cnt > 0:
                    active_strength[i] = max(active_strength[i], float(cnt))
            matched = int(np.sum(active_strength > 0))
            logger.info(
                "Active weighting table=%s unique_codes=%d matched_rows=%d",
                train_cfg.active_codes_path,
                len(active_counts),
                matched,
            )
        else:
            logger.warning("Active weighting table has no parsable combos: %s", train_cfg.active_codes_path)
    df["is_active"] = (active_strength > 0).astype(np.int64)

    train_idx, val_idx = _split_indices(len(df), train_cfg.val_fraction, train_cfg.seed)
    if bool(train_cfg.active_force_train):
        active_in_val = active_strength[val_idx] > 0
        if int(np.sum(active_in_val)) > 0:
            moved = val_idx[active_in_val]
            val_idx = val_idx[~active_in_val]
            train_idx = np.sort(np.concatenate([train_idx, moved], axis=0))
            if len(val_idx) == 0 and len(train_idx) > 1:
                val_idx = np.asarray([train_idx[-1]], dtype=np.int64)
                train_idx = train_idx[:-1]
            logger.info("Moved %d active rows from val to train", int(len(moved)))

    slots, extra, y, local_target, exp, is_active, is_hard_negative, local_distance, fmean, fstd = _prepare_arrays(
        df,
        target_col=train_cfg.target_col,
        local_target_col=train_cfg.local_target_col,
        feature_cols=feature_cols,
        experimental_col=train_cfg.experimental_label_col,
        is_active=active_strength,
        is_hard_negative=hard_negative,
        local_distance=local_distance,
        train_idx=train_idx,
    )
    if mode == "active_first":
        # local target fallback: for rows without local target, use teacher target
        local_target = np.where(np.isfinite(local_target), local_target, y).astype(np.float32)
        no_local = int(np.sum(~np.isfinite(local_target)))
        if no_local > 0:
            logger.warning("Active-first mode: %d rows have non-finite local target after fallback", no_local)

    ds = _AssistantDataset(slots, extra, y, local_target, exp, is_active, is_hard_negative, local_distance)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())

    sampler = None
    sampler_num_samples = len(train_idx)
    if float(train_cfg.oversample_top_fraction) > 0.0 or float(train_cfg.active_sample_weight) > 1.0:
        yt = y[train_idx]
        w = np.ones((len(train_idx),), dtype=np.float32)
        if float(train_cfg.oversample_top_fraction) > 0.0:
            thr = float(np.quantile(yt, 1.0 - float(train_cfg.oversample_top_fraction)))
            w[yt >= thr] = float(train_cfg.oversample_weight)
        if float(train_cfg.active_sample_weight) > 1.0:
            a = is_active[train_idx]
            active_mult = float(train_cfg.active_sample_weight) * np.maximum(1.0, a)
            w = np.where(a > 0, w * active_mult, w)
        if int(train_cfg.active_anchor_repeat) > 1:
            a = is_active[train_idx] > 0.0
            w = np.where(a, w * float(train_cfg.active_anchor_repeat), w)
            sampler_num_samples = int(
                len(train_idx) + max(0, int(train_cfg.active_anchor_repeat) - 1) * int(np.sum(a))
            )
        w = np.clip(w, 1.0, 1024.0)
        sampler = WeightedRandomSampler(w.tolist(), num_samples=max(1, sampler_num_samples), replacement=True)

    train_active = int(np.sum(is_active[train_idx] > 0))
    val_active = int(np.sum(is_active[val_idx] > 0))
    train_hard_neg = int(np.sum(is_hard_negative[train_idx] > 0))
    val_hard_neg = int(np.sum(is_hard_negative[val_idx] > 0))
    logger.info(
        "Assistant split: train=%d val=%d train_active=%d val_active=%d train_hard_neg=%d val_hard_neg=%d mode=%s",
        len(train_idx),
        len(val_idx),
        train_active,
        val_active,
        train_hard_neg,
        val_hard_neg,
        mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )

    model = AssistantRanker(cfg=model_cfg, extra_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, Any]] = []
    best_epoch = -1
    best_metrics: dict[str, float] | None = None
    best_val_loss = float("inf")

    for epoch in range(1, int(train_cfg.epochs) + 1):
        tr = _run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device, cfg=train_cfg)
        va = _run_epoch(model=model, loader=val_loader, optimizer=None, device=device, cfg=train_cfg)
        y_true, y_pred = _predict_loader(model, val_loader, device)
        metrics = ranking_metrics(y_true, y_pred, is_active=is_active[val_idx])
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_top_loss": tr["top"],
            "train_corr_loss": tr["corr"],
            "train_pair_loss": tr["pair"],
            "train_exp_loss": tr["exp"],
            "train_active_loss": tr["active"],
            "train_active_local_loss": tr["local_active"],
            "train_top_tail_loss": tr["top_tail"],
            "train_pair_local_loss": tr["pair_local"],
            "train_teacher_corr_loss": tr["teacher_corr"],
            "val_loss": va["loss"],
            "val_top_loss": va["top"],
            "val_corr_loss": va["corr"],
            "val_pair_loss": va["pair"],
            "val_exp_loss": va["exp"],
            "val_active_loss": va["active"],
            "val_active_local_loss": va["local_active"],
            "val_top_tail_loss": va["top_tail"],
            "val_pair_local_loss": va["pair_local"],
            "val_teacher_corr_loss": va["teacher_corr"],
            **metrics,
        }
        history_rows.append(row)
        logger.info(
            "Assistant epoch %d/%d train_loss=%.4f val_loss=%.4f global_corr=%.4f top5=%.4f active_top50=%d",
            epoch,
            train_cfg.epochs,
            row["train_loss"],
            row["val_loss"],
            row["global_corr_chimera"],
            row["top5_overlap"],
            int(row.get("active_hits_top50", 0.0)),
        )

        torch.save(
            build_assistant_checkpoint_payload(
                model=model,
                model_cfg=model_cfg,
                feature_cols=feature_cols,
                feature_mean=fmean,
                feature_std=fstd,
                extra_meta={"epoch": epoch, "target_col": train_cfg.target_col},
            ),
            run_dir / "assistant_last.pt",
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = float(row["val_loss"])
        if is_better_metric(
            new_metrics=row,
            best_metrics=best_metrics,
            primary_key=train_cfg.best_metric,
            tie_breakers=("top5_overlap", "top10_overlap"),
        ):
            best_metrics = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
            best_epoch = epoch
            torch.save(
                build_assistant_checkpoint_payload(
                    model=model,
                    model_cfg=model_cfg,
                    feature_cols=feature_cols,
                    feature_mean=fmean,
                    feature_std=fstd,
                    extra_meta={"epoch": epoch, "target_col": train_cfg.target_col},
                ),
                run_dir / "assistant_best.pt",
            )

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(run_dir / "train_history.csv", index=False)
    summary = {
        "data_table": data_table,
        "n_rows": len(df),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_active_train": int(train_active),
        "n_active_val": int(val_active),
        "n_hard_neg_train": int(train_hard_neg),
        "n_hard_neg_val": int(val_hard_neg),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_metric": train_cfg.best_metric,
        "best_metrics": best_metrics or {},
        "final_metrics": history_rows[-1] if history_rows else {},
        "device": device,
        "feature_cols": list(feature_cols),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
