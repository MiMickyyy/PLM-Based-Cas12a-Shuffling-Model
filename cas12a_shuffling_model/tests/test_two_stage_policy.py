from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch
import numpy as np

from cas12a_shuffling_model.composition.assistant_ranker import (
    AssistantModelConfig,
    AssistantRanker,
    build_assistant_checkpoint_payload,
)
from cas12a_shuffling_model.composition.two_stage_policy import TwoStagePolicyConfig, apply_two_stage_policy


def _toy_df() -> pd.DataFrame:
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
            "assistant_teacher_head_score": [0.95, 0.70, 0.60, 0.40, 0.20, 0.10],
            "assistant_active_head_score": [0.50, 0.92, 0.88, 0.12, 0.06, 0.04],
            "assistant_score": [0.72, 0.84, 0.79, 0.2, 0.1, 0.0],
            "s_scan_score": [0.83, 0.81, 0.78, 0.25, 0.15, 0.03],
        }
    )


def test_two_stage_recall_and_rerank_are_separated():
    df = _toy_df()
    actives = ["AAAAAAAAAAA", "AAAAAAAALAA"]

    a = apply_two_stage_policy(
        df,
        active_codes=actives,
        cfg=TwoStagePolicyConfig(
            recall_policy="scan_only",
            final_rerank_policy="active_only",
            active_similarity_mode="kernel_density_over_actives",
            active_similarity_beta=0.2,
        ),
    )
    b = apply_two_stage_policy(
        df,
        active_codes=actives,
        cfg=TwoStagePolicyConfig(
            recall_policy="teacher_recall",
            final_rerank_policy="active_only",
            recall_teacher_weight=0.9,
            recall_scan_weight=0.1,
            active_similarity_mode="kernel_density_over_actives",
            active_similarity_beta=0.2,
        ),
    )
    # recall should differ by policy
    assert not np.allclose(a["recall_stage_score"].to_numpy(), b["recall_stage_score"].to_numpy())
    # final rerank remains active-only driven (same config => same final score)
    assert np.allclose(a["final_score"].to_numpy(), b["final_score"].to_numpy())


def test_eval_two_stage_matrix_cli_smoke(tmp_path: Path):
    shortlist = _toy_df()[["slot_code_11", "s_scan_score"]].copy()
    shortlist_path = tmp_path / "shortlist.csv"
    shortlist.to_csv(shortlist_path, index=False)

    active = pd.DataFrame({"slot_code_11": ["AAAAAAAAAAA", "AAAAAAAALAA", "AAAAAAAAMAA"]})
    active_path = tmp_path / "active.csv"
    active.to_csv(active_path, index=False)

    model_cfg = AssistantModelConfig(
        slot_embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        use_extra_features=False,
        dual_head=True,
        inference_alpha=0.15,
    )
    model = AssistantRanker(cfg=model_cfg, extra_dim=0)
    ckpt = build_assistant_checkpoint_payload(
        model=model,
        model_cfg=model_cfg,
        feature_cols=[],
        feature_mean=np.zeros((0,), dtype=np.float32),
        feature_std=np.ones((0,), dtype=np.float32),
    )
    ckpt_path = tmp_path / "assistant.pt"
    torch.save(ckpt, ckpt_path)

    out_dir = tmp_path / "matrix_out"
    cmd = [
        sys.executable,
        "-m",
        "cas12a_shuffling_model.cli.eval_two_stage_matrix",
        "--config",
        "cas12a_shuffling_model/configs/slot_search_smoke.yaml",
        "--shortlist-table",
        str(shortlist_path),
        "--assistant-checkpoint",
        str(ckpt_path),
        "--active-table",
        str(active_path),
        "--recall-pool-size",
        "6",
        "--disable-loo",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.check_call(cmd)
    summary = pd.read_csv(out_dir / "matrix_summary.csv")
    assert len(summary) >= 3
    assert "present@100k" in summary.columns
    assert "hits@50" in summary.columns
