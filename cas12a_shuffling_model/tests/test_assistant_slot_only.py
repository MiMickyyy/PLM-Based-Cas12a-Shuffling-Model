import numpy as np
import pandas as pd
import torch

from cas12a_shuffling_model.composition.assistant_ranker import (
    AssistantModelConfig,
    AssistantRanker,
    AssistantRankerScorer,
    build_assistant_checkpoint_payload,
)
from cas12a_shuffling_model.composition.chimera_repr import canonicalize_chimera_table


def test_canonicalize_slot_only_without_sequence():
    df = pd.DataFrame([{"slot_code_11": "AAAAAAAAAAA"}])
    out = canonicalize_chimera_table(df, require_sequence=False)
    assert out.loc[0, "slot_01"] == 0
    assert out.loc[0, "slot_11"] == 0
    assert out.loc[0, "full_protein_sequence"] == ""
    assert np.isnan(out.loc[0, "length"])


def test_assistant_scorer_slot_only_input(tmp_path):
    cfg = AssistantModelConfig(
        slot_embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        use_extra_features=False,
    )
    model = AssistantRanker(cfg=cfg, extra_dim=0)
    payload = build_assistant_checkpoint_payload(
        model=model,
        model_cfg=cfg,
        feature_cols=[],
        feature_mean=np.zeros((0,), dtype=np.float32),
        feature_std=np.ones((0,), dtype=np.float32),
    )
    ckpt = tmp_path / "assistant.pt"
    torch.save(payload, ckpt)

    scorer = AssistantRankerScorer(str(ckpt), device="cpu")
    df = pd.DataFrame([{"slot_code_11": "ALFMALFMALF"}, {"slot_code_11": "LLLLLLLLLLL"}])
    scored = scorer.score_dataframe(df, batch_size=2)
    assert "assistant_score" in scored.columns
    assert len(scored) == 2
