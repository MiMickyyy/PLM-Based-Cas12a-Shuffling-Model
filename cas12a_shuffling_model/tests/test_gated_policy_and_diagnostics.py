from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.composition.gated_diagnostics import (
    distance_stratified_active_analysis,
    evaluate_policy_variants,
    leave_one_active_out_eval,
    missing_active_diagnosis,
    novelty_vs_score_analysis,
)
from cas12a_shuffling_model.composition.gated_policy import GatedPolicyConfig, apply_gated_policy


def _toy_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slot_code_11": [
                "AAAAAAAAAAA",
                "AAAAAAAALAA",
                "AAAAAAAAMAA",
                "LLLLLLLLLLL",
                "FFFFFFFFFFF",
                "MMMMMMMMMMM",
            ],
            "assistant_teacher_head_score": [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
            "assistant_active_head_score": [0.6, 0.95, 0.85, 0.1, 0.05, 0.02],
            "assistant_score": [0.75, 0.88, 0.80, 0.15, 0.08, 0.01],
            "s_scan_score": [0.82, 0.81, 0.79, 0.2, 0.1, 0.05],
        }
    )


def test_apply_gated_policy_hard_and_soft():
    df = _toy_table()
    actives = ["AAAAAAAAAAA", "AAAAAAAALAA"]

    hard = apply_gated_policy(
        df,
        active_codes=actives,
        cfg=GatedPolicyConfig(
            policy_mode="hard_gated",
            gate_signal="min_hamming_to_active",
            hard_distance_threshold=1,
            alpha_far=0.8,
            alpha_near=0.1,
            similarity_beta=0.2,
            recall_pool_size=6,
        ),
    )
    assert "recall_stage_score" in hard.columns
    assert "rerank_stage_score" in hard.columns
    assert "final_gated_score" in hard.columns
    near = hard.loc[hard["slot_code_11"] == "AAAAAAAAMAA", "teacher_weight_dynamic"].iloc[0]
    far = hard.loc[hard["slot_code_11"] == "LLLLLLLLLLL", "teacher_weight_dynamic"].iloc[0]
    assert near < far

    soft = apply_gated_policy(
        df,
        active_codes=actives,
        cfg=GatedPolicyConfig(
            policy_mode="soft_gated",
            gate_signal="kernel_similarity_to_actives",
            soft_center=0.3,
            soft_scale=10.0,
            alpha_far=0.8,
            alpha_near=0.1,
            similarity_beta=0.2,
            recall_pool_size=6,
        ),
    )
    assert float(soft["gate_near_weight"].min()) >= 0.0
    assert float(soft["gate_near_weight"].max()) <= 1.0


def test_gated_diagnostics_smoke(tmp_path: Path):
    df = _toy_table()
    actives = ["AAAAAAAAAAA", "AAAAAAAALAA", "AAAAAAAAMAA"]
    variants = {
        "global_fixed": GatedPolicyConfig(policy_mode="global_fixed", similarity_beta=0.0, recall_pool_size=6),
        "hard_gated": GatedPolicyConfig(
            policy_mode="hard_gated",
            gate_signal="min_hamming_to_active",
            hard_distance_threshold=1,
            alpha_far=0.7,
            alpha_near=0.2,
            similarity_beta=0.1,
            recall_pool_size=6,
        ),
    }
    compare, details = evaluate_policy_variants(
        table_df=df,
        active_codes=actives,
        variants=variants,
        rerank_top_ks=(2, 3),
        recall_top_ks=(2, 3, 6),
    )
    assert len(compare) == 2
    assert "rerank_hits_at_2" in compare.columns
    best = details["hard_gated"].rerank_pool

    loo = leave_one_active_out_eval(
        table_df=df,
        active_codes=actives,
        cfg=variants["hard_gated"],
        rerank_top_ks=(2, 3),
    )
    assert len(loo) == len(actives)

    dist = distance_stratified_active_analysis(
        ranked_df=best,
        active_codes=actives,
        score_col="final_gated_score",
    )
    assert set(["active_code", "distance_bucket", "rank"]).issubset(set(dist.columns))

    nov = novelty_vs_score_analysis(
        ranked_df=best,
        active_codes=actives,
        score_cols=["final_gated_score", "score_teacher"],
        top_k=4,
    )
    assert len(nov) >= 1

    miss = missing_active_diagnosis(
        ranked_df=best,
        active_codes=actives,
        recall_pool_size=6,
    )
    assert len(miss) == len(actives)
