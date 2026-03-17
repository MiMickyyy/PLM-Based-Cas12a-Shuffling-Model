from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from cas12a_shuffling_model.composition.assistant_ranker import AssistantRankerScorer, detect_torch_device
from cas12a_shuffling_model.composition.chimera_repr import canonicalize_chimera_table, slot_columns
from cas12a_shuffling_model.composition.ranking_losses import (
    correlation_loss,
    filtered_pairwise_rank_loss,
    is_better_metric,
    ranking_metrics,
    top_focus_loss,
)
from cas12a_shuffling_model.composition.slot_scorer import (
    SlotScorerConfig,
    TinySlotScorer,
    build_slot_scorer_checkpoint_payload,
)
from cas12a_shuffling_model.composition.table_io import read_table, write_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotScorerTrainConfig:
    seed: int = 13
    batch_size: int = 1024
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    top_weight: float = 1.0
    corr_weight: float = 0.4
    pair_weight: float = 0.2
    pair_min_gap: float = 0.01
    pair_near_tie_gap: float = 0.01
    pair_easy_ratio: float = 0.2
    pair_medium_ratio: float = 0.5
    pair_hard_ratio: float = 0.3
    pair_pairs_per_batch: int = 1024
    pair_margin: float = 0.0
    pair_margin_alpha: float = 0.5
    pair_margin_min: float = 0.0
    pair_margin_max: float = 0.3
    top_fraction: float = 0.10
    top_pairs_per_batch: int = 512
    top_margin: float = 0.0
    oversample_top_fraction: float = 0.10
    oversample_weight: float = 2.0
    target_col: str = "assistant_score"
    best_metric: str = "global_corr_chimera"
    num_workers: int = 0
    device: str | None = None
    cpu_threads: int | None = None
    interop_threads: int | None = None


class _SlotDataset(Dataset):
    def __init__(self, slots: np.ndarray, target: np.ndarray):
        self.slots = slots
        self.target = target

    def __len__(self) -> int:
        return int(self.slots.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "slots": torch.tensor(self.slots[idx], dtype=torch.long),
            "target": torch.tensor(self.target[idx], dtype=torch.float32),
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpu_threads(cfg: SlotScorerTrainConfig) -> None:
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


def _run_epoch(
    *,
    model: TinySlotScorer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    cfg: SlotScorerTrainConfig,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats = {"loss": 0.0, "top": 0.0, "corr": 0.0, "pair": 0.0, "batches": 0}
    for batch in loader:
        slots = batch["slots"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(slots)
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
            loss = cfg.top_weight * l_top + cfg.corr_weight * l_corr + cfg.pair_weight * l_pair
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
        stats["loss"] += float(loss.item())
        stats["top"] += float(l_top.item())
        stats["corr"] += float(l_corr.item())
        stats["pair"] += float(l_pair.item())
        stats["batches"] += 1
    if stats["batches"] > 0:
        for k in ("loss", "top", "corr", "pair"):
            stats[k] /= stats["batches"]
    return stats


@torch.no_grad()
def _predict_loader(
    model: TinySlotScorer,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    for batch in loader:
        slots = batch["slots"].to(device)
        target = batch["target"].numpy()
        pred = model(slots).detach().cpu().numpy()
        y_true.append(target)
        y_pred.append(pred)
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


def train_slot_scorer(
    *,
    data_table: str,
    out_dir: str,
    model_cfg: SlotScorerConfig,
    train_cfg: SlotScorerTrainConfig,
    assistant_checkpoint: str | None = None,
    cache_scored_table: str | None = None,
) -> dict[str, Any]:
    _set_seed(train_cfg.seed)
    _set_cpu_threads(train_cfg)
    device = detect_torch_device(train_cfg.device)
    logger.info("Slot scorer device: %s", device)

    df = read_table(data_table)
    df = canonicalize_chimera_table(df, require_sequence=False)
    if train_cfg.target_col not in df.columns:
        if not assistant_checkpoint:
            raise KeyError(
                f"Missing target column `{train_cfg.target_col}` and no assistant checkpoint was provided"
            )
        assistant = AssistantRankerScorer(assistant_checkpoint, device=device)
        df = assistant.score_dataframe(df, batch_size=max(512, train_cfg.batch_size))
        if cache_scored_table:
            saved = write_table(df, cache_scored_table)
            logger.info("Wrote assistant-scored cache table: %s", saved)
    df = df[pd.to_numeric(df[train_cfg.target_col], errors="coerce").notna()].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Not enough rows for slot scorer training")

    slots = df[slot_columns()].to_numpy(dtype=np.int64)
    target = pd.to_numeric(df[train_cfg.target_col], errors="coerce").to_numpy(dtype=np.float32)
    train_idx, val_idx = _split_indices(len(df), train_cfg.val_fraction, train_cfg.seed)

    ds = _SlotDataset(slots=slots, target=target)
    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())

    sampler = None
    if float(train_cfg.oversample_top_fraction) > 0:
        yt = target[train_idx]
        thr = float(np.quantile(yt, 1.0 - float(train_cfg.oversample_top_fraction)))
        w = np.ones((len(train_idx),), dtype=np.float32)
        w[yt >= thr] = float(train_cfg.oversample_weight)
        sampler = WeightedRandomSampler(w.tolist(), num_samples=len(train_idx), replacement=True)

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

    model = TinySlotScorer(cfg=model_cfg).to(device)
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
        metrics = ranking_metrics(y_true, y_pred)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_top_loss": tr["top"],
            "train_corr_loss": tr["corr"],
            "train_pair_loss": tr["pair"],
            "val_loss": va["loss"],
            "val_top_loss": va["top"],
            "val_corr_loss": va["corr"],
            "val_pair_loss": va["pair"],
            **metrics,
        }
        history_rows.append(row)
        logger.info(
            "S_scan epoch %d/%d train_loss=%.4f val_loss=%.4f corr=%.4f top5=%.4f",
            epoch,
            train_cfg.epochs,
            row["train_loss"],
            row["val_loss"],
            row["global_corr_chimera"],
            row["top5_overlap"],
        )

        torch.save(
            build_slot_scorer_checkpoint_payload(
                model=model,
                cfg=model_cfg,
                extra_meta={"epoch": epoch, "target_col": train_cfg.target_col},
            ),
            run_dir / "slot_scorer_last.pt",
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
                build_slot_scorer_checkpoint_payload(
                    model=model,
                    cfg=model_cfg,
                    extra_meta={"epoch": epoch, "target_col": train_cfg.target_col},
                ),
                run_dir / "slot_scorer_best.pt",
            )

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(run_dir / "train_history.csv", index=False)
    summary = {
        "data_table": data_table,
        "n_rows": len(df),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_metric": train_cfg.best_metric,
        "best_metrics": best_metrics or {},
        "final_metrics": history_rows[-1] if history_rows else {},
        "device": device,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
