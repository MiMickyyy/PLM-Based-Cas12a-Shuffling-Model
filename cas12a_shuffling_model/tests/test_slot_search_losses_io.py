from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cas12a_shuffling_model.composition.chimera_repr import load_active_code_counts
from cas12a_shuffling_model.composition.ranking_losses import active_vs_background_rank_loss, hard_negative_rank_loss
from cas12a_shuffling_model.composition.table_io import read_table


def test_hard_negative_rank_loss_returns_finite():
    pred = torch.tensor([0.1, 0.2, 0.3, -0.1, -0.2, 0.5], dtype=torch.float32)
    target = torch.tensor([0.0, 0.1, 0.4, -0.2, -0.3, 0.6], dtype=torch.float32)
    loss = hard_negative_rank_loss(
        pred=pred,
        target=target,
        top_fraction=0.3,
        pairs_per_batch=4,
        margin=0.05,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_read_table_parquet_fallback_csv(monkeypatch, tmp_path):
    csv_path = Path(tmp_path) / "sample.csv"
    parquet_path = Path(tmp_path) / "sample.parquet"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv_path, index=False)

    def _raise(*args, **kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd, "read_parquet", _raise)
    out = read_table(parquet_path)
    assert len(out) == 2
    assert out["a"].tolist() == [1, 2]


def test_active_vs_background_rank_loss_returns_finite():
    pred = torch.tensor([0.8, 0.6, -0.1, -0.3, 0.2], dtype=torch.float32)
    target = torch.tensor([0.5, 0.4, -0.2, -0.4, 0.1], dtype=torch.float32)
    is_active = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    loss = active_vs_background_rank_loss(
        pred=pred,
        target=target,
        is_active=is_active,
        pairs_per_batch=16,
        margin=0.1,
        min_target_gap=-1.0,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_load_active_code_counts_from_slot_columns(tmp_path):
    table = pd.DataFrame(
        [
            {1: "A", 2: "L", 3: "F", 4: "M", 5: "A", 6: "L", 7: "F", 8: "M", 9: "A", 10: "L", 11: "F"},
            {1: "A", 2: "L", 3: "F", 4: "M", 5: "A", 6: "L", 7: "F", 8: "M", 9: "A", 10: "L", 11: "F"},
            {1: "L", 2: "L", 3: "L", 4: "L", 5: "L", 6: "L", 7: "L", 8: "L", 9: "L", 10: "L", 11: "L"},
        ]
    )
    p = Path(tmp_path) / "actives.csv"
    table.to_csv(p, index=False)
    counts = load_active_code_counts(p)
    assert counts["ALFMALFMALF"] == 2
    assert counts["LLLLLLLLLLL"] == 1
