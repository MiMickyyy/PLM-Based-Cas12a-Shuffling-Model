import numpy as np
import torch

from cas12a_shuffling_model.composition.slot_scorer import SlotScorerConfig, TinySlotScorer


def test_slot_scorer_forward_components_shapes():
    model = TinySlotScorer(cfg=SlotScorerConfig(slot_embed_dim=8, mlp_hidden_dim=32, mlp_layers=2))
    x = torch.randint(0, 4, (16, 11), dtype=torch.long)
    out = model.forward_components(x)
    assert set(out.keys()) == {"score", "main_effect", "pairwise_effect", "nonlinear_effect"}
    assert out["score"].shape == (16,)
    assert out["main_effect"].shape == (16,)
    assert out["pairwise_effect"].shape == (16,)
    assert out["nonlinear_effect"].shape == (16,)


def test_slot_scorer_without_pairwise():
    model = TinySlotScorer(
        cfg=SlotScorerConfig(
            slot_embed_dim=8,
            mlp_hidden_dim=32,
            mlp_layers=2,
            enable_pairwise=False,
        )
    )
    x = torch.randint(0, 4, (8, 11), dtype=torch.long)
    out = model.forward_components(x)
    assert np.allclose(out["pairwise_effect"].detach().cpu().numpy(), 0.0)

