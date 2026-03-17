from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from cas12a_shuffling_model.composition.chimera_repr import (
    SLOT_COUNT,
    canonicalize_chimera_table,
    slot_columns,
)


@dataclass(frozen=True)
class AssistantModelConfig:
    slot_embed_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    use_extra_features: bool = True


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = input_dim
    n = max(1, int(num_layers))
    for _ in range(n - 1):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        d = hidden_dim
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class AssistantRanker(nn.Module):
    def __init__(self, *, cfg: AssistantModelConfig, extra_dim: int):
        super().__init__()
        self.cfg = cfg
        self.extra_dim = int(extra_dim)
        self.slot_embed = nn.Parameter(torch.randn(SLOT_COUNT, 4, cfg.slot_embed_dim) * 0.02)
        input_dim = SLOT_COUNT * cfg.slot_embed_dim + (self.extra_dim if cfg.use_extra_features else 0)
        self.head = _build_mlp(
            input_dim=input_dim,
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
        )

    def _slot_features(self, slots_int: torch.Tensor) -> torch.Tensor:
        parts = [self.slot_embed[i, slots_int[:, i], :] for i in range(SLOT_COUNT)]
        return torch.cat(parts, dim=1)

    def forward(self, slots_int: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        x = self._slot_features(slots_int)
        if self.cfg.use_extra_features and self.extra_dim > 0:
            if extra is None:
                extra = torch.zeros((slots_int.shape[0], self.extra_dim), device=slots_int.device)
            x = torch.cat([x, extra], dim=1)
        return self.head(x).squeeze(-1)


def detect_torch_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _prepare_slots_and_extra(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    canonical = canonicalize_chimera_table(df, require_sequence=False)
    slots = canonical[slot_columns()].to_numpy(dtype=np.int64, copy=True)
    if len(feature_cols) == 0:
        extra = np.zeros((len(canonical), 0), dtype=np.float32)
        return slots, extra
    extra_df = canonical.reindex(columns=list(feature_cols), fill_value=np.nan)
    extra = extra_df.to_numpy(dtype=np.float32)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_mean is not None and feature_std is not None and extra.shape[1] > 0:
        denom = np.where(np.abs(feature_std) < 1e-8, 1.0, feature_std)
        extra = (extra - feature_mean) / denom
    return slots, extra.astype(np.float32)


class AssistantRankerScorer:
    def __init__(self, checkpoint_path: str, *, device: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.device = detect_torch_device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mc = AssistantModelConfig(**dict(ckpt["model_config"]))
        self.feature_cols = list(ckpt.get("feature_cols", []))
        self.feature_mean = np.asarray(ckpt.get("feature_mean", []), dtype=np.float32)
        self.feature_std = np.asarray(ckpt.get("feature_std", []), dtype=np.float32)
        self.model = AssistantRanker(cfg=mc, extra_dim=len(self.feature_cols))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_dataframe(self, df: pd.DataFrame, *, batch_size: int = 1024) -> pd.DataFrame:
        slots, extra = _prepare_slots_and_extra(
            df,
            feature_cols=self.feature_cols,
            feature_mean=self.feature_mean if len(self.feature_cols) > 0 else None,
            feature_std=self.feature_std if len(self.feature_cols) > 0 else None,
        )
        out = canonicalize_chimera_table(df, require_sequence=False)
        preds: list[np.ndarray] = []
        for s in range(0, slots.shape[0], int(batch_size)):
            e = min(slots.shape[0], s + int(batch_size))
            slot_t = torch.as_tensor(slots[s:e], dtype=torch.long, device=self.device)
            if extra.shape[1] > 0:
                extra_t = torch.as_tensor(extra[s:e], dtype=torch.float32, device=self.device)
            else:
                extra_t = None
            p = self.model(slot_t, extra_t).detach().cpu().numpy()
            preds.append(p)
        out["assistant_score"] = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        return out

    def save_scored_table(self, df: pd.DataFrame, out_path: str, *, batch_size: int = 1024) -> str:
        out = self.score_dataframe(df, batch_size=batch_size)
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".parquet":
            try:
                out.to_parquet(p, index=False)
                return str(p)
            except Exception:
                p = p.with_suffix(".csv")
        out.to_csv(p, index=False)
        return str(p)


def build_assistant_checkpoint_payload(
    *,
    model: AssistantRanker,
    model_cfg: AssistantModelConfig,
    feature_cols: Sequence[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_cfg),
        "feature_cols": list(feature_cols),
        "feature_mean": [float(x) for x in np.asarray(feature_mean).reshape(-1).tolist()],
        "feature_std": [float(x) for x in np.asarray(feature_std).reshape(-1).tolist()],
    }
    if extra_meta:
        payload["meta"] = dict(extra_meta)
    return payload
