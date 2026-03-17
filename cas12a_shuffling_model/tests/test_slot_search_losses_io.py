from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cas12a_shuffling_model.composition.ranking_losses import hard_negative_rank_loss
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
