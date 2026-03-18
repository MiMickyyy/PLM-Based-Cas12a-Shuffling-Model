from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.composition.active_prior import (
    ActivePriorConfig,
    active_prior_beta_sweep,
    active_similarity_score,
    apply_active_prior,
)
from cas12a_shuffling_model.composition.hard_negative_mining import (
    HardNegativeMiningConfig,
    build_active_local_training_table,
)
from cas12a_shuffling_model.composition.local_expand import LocalExpandConfig, expand_local_neighborhood


def test_active_prior_modes_and_sweep():
    codes = ["AAAAAAAAAAA", "AAAAAAAALAA", "LLLLLLLLLLL"]
    actives = ["AAAAAAAAAAA"]
    for mode in ("min_hamming_similarity", "weighted_slot_similarity", "kernel_density_over_actives"):
        sim = active_similarity_score(codes=codes, active_codes=actives, mode=mode, gamma=0.7)
        assert sim.shape[0] == len(codes)
    df = pd.DataFrame({"slot_code_11": codes, "assistant_score": [1.0, 0.9, 0.1]})
    out = apply_active_prior(
        df,
        score_col="assistant_score",
        active_codes=actives,
        cfg=ActivePriorConfig(mode="kernel_density_over_actives", beta=0.2, gamma=0.7),
    )
    assert "final_score" in out.columns
    sweep = active_prior_beta_sweep(
        df=out,
        base_score_col="assistant_score",
        active_codes=actives,
        mode="kernel_density_over_actives",
        betas=[0.0, 0.1, 0.2],
        gamma=0.7,
        top_ks=(2,),
    )
    assert len(sweep) == 3
    assert "hits_at_2" in sweep.columns


def test_hard_negative_mining_and_local_expand(tmp_path):
    base = pd.DataFrame(
        {
            "slot_code_11": ["AAAAAAAAAAA", "AAAAAAAALAA", "AAAAAAAAMAA", "LLLLLLLLLLL"],
            "teacher_seq_score_norm": [1.2, 0.8, 0.7, -1.0],
            "teacher_junction_mean": [0.2, 0.1, 0.1, -0.4],
            "teacher_junction_min": [0.1, 0.0, 0.0, -0.5],
            "length": [1300, 1300, 1300, 1300],
        }
    )
    rerank = pd.DataFrame(
        {
            "slot_code_11": ["AAAAAAAAAAA", "AAAAAAAALAA", "AAAAAAAAMAA", "AAAAAAAFAAA", "LLLLLLLLLLL"],
            "assistant_score": [1.1, 1.4, 1.3, 1.25, 0.2],
        }
    )
    base_path = Path(tmp_path) / "base.csv"
    rerank_path = Path(tmp_path) / "rerank.csv"
    out_path = Path(tmp_path) / "train_active_local.csv"
    base.to_csv(base_path, index=False)
    rerank.to_csv(rerank_path, index=False)

    summary = build_active_local_training_table(
        base_table=str(base_path),
        rerank_table=str(rerank_path),
        active_codes=["AAAAAAAAAAA"],
        out_table=str(out_path),
        cfg=HardNegativeMiningConfig(
            score_col="assistant_score",
            top_pool_size=100,
            max_negatives=10,
            min_distance=1,
            max_distance=4,
            active_score_quantile=0.1,
            include_missing_from_base=True,
        ),
    )
    assert summary["n_hard_negatives"] >= 1
    out_df = pd.read_csv(out_path)
    assert "is_hard_negative" in out_df.columns
    assert out_df["is_hard_negative"].sum() >= 1
    assert "active_local_target" in out_df.columns

    expanded_path = Path(tmp_path) / "expanded.csv"
    exp_summary = expand_local_neighborhood(
        seed_table=str(rerank_path),
        active_codes=["AAAAAAAAAAA"],
        out_table=str(expanded_path),
        cfg=LocalExpandConfig(
            top_seed_count=2,
            include_hamming_2_top_seeds=1,
            active_distance_cap=2,
        ),
    )
    assert exp_summary["n_expanded_rows"] > 0
    exp_df = pd.read_csv(expanded_path)
    assert "distance_from_seed" in exp_df.columns
