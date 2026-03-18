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
    dual_head: bool = False
    inference_alpha: float = 0.15


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
        if bool(cfg.dual_head):
            self.teacher_head = _build_mlp(
                input_dim=input_dim,
                hidden_dim=int(cfg.hidden_dim),
                num_layers=int(cfg.num_layers),
                dropout=float(cfg.dropout),
            )
            self.active_head = _build_mlp(
                input_dim=input_dim,
                hidden_dim=int(cfg.hidden_dim),
                num_layers=int(cfg.num_layers),
                dropout=float(cfg.dropout),
            )
        else:
            self.head = _build_mlp(
                input_dim=input_dim,
                hidden_dim=int(cfg.hidden_dim),
                num_layers=int(cfg.num_layers),
                dropout=float(cfg.dropout),
            )

    def _slot_features(self, slots_int: torch.Tensor) -> torch.Tensor:
        parts = [self.slot_embed[i, slots_int[:, i], :] for i in range(SLOT_COUNT)]
        return torch.cat(parts, dim=1)

    def forward(
        self,
        slots_int: torch.Tensor,
        extra: torch.Tensor | None = None,
        *,
        return_heads: bool = False,
        alpha: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._slot_features(slots_int)
        if self.cfg.use_extra_features and self.extra_dim > 0:
            if extra is None:
                extra = torch.zeros((slots_int.shape[0], self.extra_dim), device=slots_int.device)
            x = torch.cat([x, extra], dim=1)
        if bool(self.cfg.dual_head):
            teacher_score = self.teacher_head(x).squeeze(-1)
            active_score = self.active_head(x).squeeze(-1)
            w = float(alpha if alpha is not None else self.cfg.inference_alpha)
            w = min(1.0, max(0.0, w))
            combined = w * teacher_score + (1.0 - w) * active_score
            if return_heads:
                return combined, teacher_score, active_score
            return combined
        out = self.head(x).squeeze(-1)
        if return_heads:
            return out, out, out
        return out


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
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        mc = AssistantModelConfig(**dict(ckpt["model_config"]))
        self.feature_cols = list(ckpt.get("feature_cols", []))
        self.feature_mean = np.asarray(ckpt.get("feature_mean", []), dtype=np.float32)
        self.feature_std = np.asarray(ckpt.get("feature_std", []), dtype=np.float32)
        self.dual_head = bool(mc.dual_head)
        self.inference_alpha = float(mc.inference_alpha)
        self.model = AssistantRanker(cfg=mc, extra_dim=len(self.feature_cols))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_dataframe(
        self,
        df: pd.DataFrame,
        *,
        batch_size: int = 1024,
        dual_head_alpha: float | None = None,
    ) -> pd.DataFrame:
        slots, extra = _prepare_slots_and_extra(
            df,
            feature_cols=self.feature_cols,
            feature_mean=self.feature_mean if len(self.feature_cols) > 0 else None,
            feature_std=self.feature_std if len(self.feature_cols) > 0 else None,
        )
        out = canonicalize_chimera_table(df, require_sequence=False)
        preds: list[np.ndarray] = []
        teacher_preds: list[np.ndarray] = []
        active_preds: list[np.ndarray] = []
        for s in range(0, slots.shape[0], int(batch_size)):
            e = min(slots.shape[0], s + int(batch_size))
            slot_batch = np.ascontiguousarray(slots[s:e]).copy()
            slot_t = torch.as_tensor(slot_batch, dtype=torch.long, device=self.device)
            if extra.shape[1] > 0:
                extra_batch = np.ascontiguousarray(extra[s:e]).copy()
                extra_t = torch.as_tensor(extra_batch, dtype=torch.float32, device=self.device)
            else:
                extra_t = None
            if self.dual_head:
                p, p_t, p_a = self.model(
                    slot_t,
                    extra_t,
                    return_heads=True,
                    alpha=dual_head_alpha,
                )
                preds.append(p.detach().cpu().numpy())
                teacher_preds.append(p_t.detach().cpu().numpy())
                active_preds.append(p_a.detach().cpu().numpy())
            else:
                p = self.model(slot_t, extra_t).detach().cpu().numpy()
                preds.append(p)
        out["assistant_score"] = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.float32)
        if self.dual_head:
            out["assistant_teacher_head_score"] = (
                np.concatenate(teacher_preds, axis=0) if teacher_preds else np.zeros((0,), dtype=np.float32)
            )
            out["assistant_active_head_score"] = (
                np.concatenate(active_preds, axis=0) if active_preds else np.zeros((0,), dtype=np.float32)
            )
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
