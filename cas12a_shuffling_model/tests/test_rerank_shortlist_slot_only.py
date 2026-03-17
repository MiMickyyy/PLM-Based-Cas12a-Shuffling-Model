from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cas12a_shuffling_model.composition.assistant_ranker import (
    AssistantModelConfig,
    AssistantRanker,
    build_assistant_checkpoint_payload,
)
from cas12a_shuffling_model.composition.rerank import RerankConfig, rerank_shortlist
from cas12a_shuffling_model.teacher.protgpt2_scorer import TeacherScore


class _DummyTeacherScorer:
    model_fingerprint = "dummy"

    def score_many(self, *, seqs_aa, domain_lengths_list=None, batch_size=4):
        out = []
        for i, seq in enumerate(seqs_aa):
            seq = str(seq)
            raw = float((i + 1) * 0.1 + seq.count("A") * 0.001)
            out.append(
                TeacherScore(
                    seq_hash=f"h_{hash(seq) & 0xffff:x}",
                    seq_len=len(seq),
                    global_score=raw,
                    junction_scores=[raw] * 10,
                    from_cache=True,
                )
            )
        return out


def test_rerank_shortlist_with_slot_only_input(tmp_path):
    shortlist = pd.DataFrame(
        {
            "slot_code_11": ["AAAAAAAAAAA", "LLLLLLLLLLL", "FFFFFFFFFFF"],
            "s_scan_score": [1.0, 0.5, 0.2],
        }
    )
    shortlist_path = Path(tmp_path) / "shortlist.csv"
    shortlist.to_csv(shortlist_path, index=False)

    model_cfg = AssistantModelConfig(
        slot_embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        use_extra_features=False,
    )
    model = AssistantRanker(cfg=model_cfg, extra_dim=0)
    ckpt_payload = build_assistant_checkpoint_payload(
        model=model,
        model_cfg=model_cfg,
        feature_cols=[],
        feature_mean=np.zeros((0,), dtype=np.float32),
        feature_std=np.ones((0,), dtype=np.float32),
    )
    ckpt_path = Path(tmp_path) / "assistant.pt"
    torch.save(ckpt_payload, ckpt_path)

    out_dir = Path(tmp_path) / "rerank_out"
    outputs = rerank_shortlist(
        shortlist_table=str(shortlist_path),
        assistant_checkpoint=str(ckpt_path),
        out_dir=str(out_dir),
        cfg=RerankConfig(batch_size=2, top_k=2, include_sequence=False),
        validated_domains=None,
        device="cpu",
    )

    all_df = pd.read_csv(outputs["all_csv"])
    top_df = pd.read_csv(outputs["top_csv"])
    assert "assistant_score" in all_df.columns
    assert "assistant_rank" in all_df.columns
    assert len(all_df) == 3
    assert len(top_df) == 2


def test_rerank_shortlist_with_teacher_audit(tmp_path):
    shortlist = pd.DataFrame(
        {
            "slot_code_11": ["AAAAAAAAAAA", "LLLLLLLLLLL", "FFFFFFFFFFF"],
            "s_scan_score": [1.0, 0.5, 0.2],
        }
    )
    shortlist_path = Path(tmp_path) / "shortlist.csv"
    shortlist.to_csv(shortlist_path, index=False)

    model_cfg = AssistantModelConfig(
        slot_embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        use_extra_features=False,
    )
    model = AssistantRanker(cfg=model_cfg, extra_dim=0)
    ckpt_payload = build_assistant_checkpoint_payload(
        model=model,
        model_cfg=model_cfg,
        feature_cols=[],
        feature_mean=np.zeros((0,), dtype=np.float32),
        feature_std=np.ones((0,), dtype=np.float32),
    )
    ckpt_path = Path(tmp_path) / "assistant.pt"
    torch.save(ckpt_payload, ckpt_path)

    validated_domains = {}
    for parent in ("As", "Lb", "Fn", "Mb2"):
        for slot in range(1, 12):
            validated_domains[(parent, slot)] = "A"

    out_dir = Path(tmp_path) / "rerank_out_audit"
    outputs = rerank_shortlist(
        shortlist_table=str(shortlist_path),
        assistant_checkpoint=str(ckpt_path),
        out_dir=str(out_dir),
        cfg=RerankConfig(
            batch_size=2,
            top_k=3,
            include_sequence=True,
            teacher_audit=True,
            teacher_audit_top_k=3,
            teacher_audit_batch_size=2,
        ),
        validated_domains=validated_domains,
        device="cpu",
        teacher_scorer=_DummyTeacherScorer(),
    )

    assert "teacher_audit_csv" in outputs
    assert "teacher_audit_json" in outputs
    audit_df = pd.read_csv(outputs["teacher_audit_csv"])
    assert len(audit_df) == 3
    assert "teacher_global_score" in audit_df.columns
