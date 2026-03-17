from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cas12a_shuffling_model.composition.chimera_repr import SLOT_COUNT, index_to_slot_matrix, slot_matrix_to_codes


@dataclass(frozen=True)
class SlotScorerConfig:
    slot_embed_dim: int = 8
    mlp_hidden_dim: int = 64
    mlp_layers: int = 2
    dropout: float = 0.1
    enable_pairwise: bool = True


def _build_mlp(input_dim: int, hidden_dim: int, layers: int, dropout: float) -> nn.Sequential:
    modules: list[nn.Module] = []
    d = input_dim
    n = max(1, int(layers))
    for _ in range(n - 1):
        modules.append(nn.Linear(d, hidden_dim))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        d = hidden_dim
    modules.append(nn.Linear(d, 1))
    return nn.Sequential(*modules)


def _pair_indices() -> tuple[torch.Tensor, torch.Tensor]:
    ii = []
    jj = []
    for i in range(SLOT_COUNT):
        for j in range(i + 1, SLOT_COUNT):
            ii.append(i)
            jj.append(j)
    return torch.tensor(ii, dtype=torch.long), torch.tensor(jj, dtype=torch.long)


class TinySlotScorer(nn.Module):
    def __init__(self, *, cfg: SlotScorerConfig):
        super().__init__()
        self.cfg = cfg
        self.main_effect = nn.Parameter(torch.zeros(SLOT_COUNT, 4))
        self.slot_embed = nn.Parameter(torch.randn(SLOT_COUNT, 4, cfg.slot_embed_dim) * 0.02)
        pair_i, pair_j = _pair_indices()
        self.register_buffer("pair_i", pair_i)
        self.register_buffer("pair_j", pair_j)
        n_pairs = int(pair_i.numel())
        self.pair_effect = nn.Parameter(torch.zeros(n_pairs, 4, 4))
        self.mlp = _build_mlp(
            input_dim=SLOT_COUNT * cfg.slot_embed_dim,
            hidden_dim=int(cfg.mlp_hidden_dim),
            layers=int(cfg.mlp_layers),
            dropout=float(cfg.dropout),
        )

    def forward_components(self, slots_int: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = slots_int.shape[0]

        me = self.main_effect.unsqueeze(0).expand(bsz, -1, -1)
        main = me.gather(dim=2, index=slots_int.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        pi = slots_int[:, self.pair_i]
        pj = slots_int[:, self.pair_j]
        if bool(self.cfg.enable_pairwise):
            table = self.pair_effect.view(self.pair_effect.shape[0], 16)
            flat_idx = pi * 4 + pj
            te = table.unsqueeze(0).expand(bsz, -1, -1)
            pair = te.gather(dim=2, index=flat_idx.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        else:
            pair = torch.zeros_like(main)

        emb_parts = [self.slot_embed[i, slots_int[:, i], :] for i in range(SLOT_COUNT)]
        emb = torch.cat(emb_parts, dim=1)
        nonlinear = self.mlp(emb).squeeze(-1)
        total = main + pair + nonlinear
        return {
            "score": total,
            "main_effect": main,
            "pairwise_effect": pair,
            "nonlinear_effect": nonlinear,
        }

    def forward(self, slots_int: torch.Tensor) -> torch.Tensor:
        return self.forward_components(slots_int)["score"]


def detect_torch_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SlotScorerPredictor:
    def __init__(self, checkpoint_path: str, *, device: str | None = None):
        self.device = detect_torch_device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = SlotScorerConfig(**dict(ckpt["model_config"]))
        self.model = TinySlotScorer(cfg=cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_slot_matrix(
        self,
        slot_matrix: np.ndarray,
        *,
        return_components: bool = False,
    ) -> dict[str, np.ndarray]:
        arr = np.asarray(slot_matrix, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != SLOT_COUNT:
            raise ValueError(f"slot_matrix must be [N,{SLOT_COUNT}]")
        x = torch.as_tensor(arr, dtype=torch.long, device=self.device)
        comp = self.model.forward_components(x)
        out = {"score": comp["score"].detach().cpu().numpy()}
        if return_components:
            out["main_effect"] = comp["main_effect"].detach().cpu().numpy()
            out["pairwise_effect"] = comp["pairwise_effect"].detach().cpu().numpy()
            out["nonlinear_effect"] = comp["nonlinear_effect"].detach().cpu().numpy()
        return out

    @torch.no_grad()
    def score_indices(
        self,
        indices: np.ndarray,
        *,
        return_components: bool = False,
    ) -> dict[str, np.ndarray]:
        mat = index_to_slot_matrix(indices)
        out = self.score_slot_matrix(mat, return_components=return_components)
        out["slot_code_11"] = np.asarray(slot_matrix_to_codes(mat), dtype=object)
        return out


def build_slot_scorer_checkpoint_payload(
    *,
    model: TinySlotScorer,
    cfg: SlotScorerConfig,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(cfg),
    }
    if extra_meta:
        payload["meta"] = dict(extra_meta)
    return payload


def save_slot_scorer_checkpoint(payload: dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
