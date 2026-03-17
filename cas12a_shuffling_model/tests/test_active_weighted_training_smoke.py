from pathlib import Path

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.assistant_ranker import AssistantModelConfig
from cas12a_shuffling_model.composition.chimera_repr import sample_slot_codes
from cas12a_shuffling_model.composition.train_assistant import AssistantTrainConfig, train_assistant_ranker
from cas12a_shuffling_model.composition.train_slot_scorer import SlotScorerTrainConfig, train_slot_scorer
from cas12a_shuffling_model.composition.slot_scorer import SlotScorerConfig


def _make_train_table(tmp_path: Path) -> tuple[Path, Path]:
    codes = sample_slot_codes(n=64, seed=7)
    rng = np.random.default_rng(7)
    table = pd.DataFrame(
        {
            "slot_code_11": codes,
            "teacher_seq_score_norm": rng.normal(0.0, 1.0, size=64),
            "assistant_score": rng.normal(0.0, 1.0, size=64),
            "teacher_junction_mean": rng.normal(0.0, 1.0, size=64),
            "teacher_junction_min": rng.normal(0.0, 1.0, size=64),
            "length": rng.integers(1200, 1400, size=64),
        }
    )
    train_table = tmp_path / "train.csv"
    table.to_csv(train_table, index=False)

    active_table = pd.DataFrame([{"slot_code_11": codes[0]}, {"slot_code_11": codes[1]}])
    active_path = tmp_path / "actives.csv"
    active_table.to_csv(active_path, index=False)
    return train_table, active_path


def test_assistant_active_weighted_smoke(tmp_path):
    train_table, active_path = _make_train_table(tmp_path)
    out_dir = Path(tmp_path) / "assistant"
    summary = train_assistant_ranker(
        data_table=str(train_table),
        out_dir=str(out_dir),
        model_cfg=AssistantModelConfig(
            slot_embed_dim=8,
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
            use_extra_features=False,
        ),
        train_cfg=AssistantTrainConfig(
            epochs=1,
            batch_size=16,
            lr=1e-3,
            num_workers=0,
            device="cpu",
            target_col="teacher_seq_score_norm",
            active_codes_path=str(active_path),
            active_sample_weight=4.0,
            active_loss_weight=0.5,
            active_pairs_per_batch=32,
            active_min_target_gap=-1.0,
        ),
        feature_cols=[],
    )
    assert summary["n_active_train"] >= 1
    assert (out_dir / "assistant_best.pt").exists()


def test_slot_scorer_active_weighted_smoke(tmp_path):
    train_table, active_path = _make_train_table(tmp_path)
    out_dir = Path(tmp_path) / "slot"
    summary = train_slot_scorer(
        data_table=str(train_table),
        out_dir=str(out_dir),
        model_cfg=SlotScorerConfig(
            slot_embed_dim=8,
            mlp_hidden_dim=32,
            mlp_layers=1,
            dropout=0.0,
            enable_pairwise=True,
        ),
        train_cfg=SlotScorerTrainConfig(
            epochs=1,
            batch_size=32,
            lr=1e-3,
            num_workers=0,
            device="cpu",
            target_col="assistant_score",
            active_codes_path=str(active_path),
            active_sample_weight=4.0,
            active_loss_weight=0.5,
            active_pairs_per_batch=32,
            active_min_target_gap=-1.0,
        ),
    )
    assert summary["n_active_train"] >= 1
    assert (out_dir / "slot_scorer_best.pt").exists()
