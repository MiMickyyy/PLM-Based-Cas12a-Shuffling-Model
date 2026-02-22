import torch
import pandas as pd

from cas12a_shuffling_model.student.gru_model import GRUAutoregressiveLM
from cas12a_shuffling_model.student.score_student import StudentScorer
from cas12a_shuffling_model.student.vocab import build_default_vocab
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig


def _make_ckpt(path):
    vocab = build_default_vocab()
    model = GRUAutoregressiveLM(
        vocab_size=vocab.size,
        embed_dim=16,
        hidden_dim=24,
        num_layers=1,
        dropout=0.0,
        pad_idx=vocab.pad_id,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embed_dim": 16,
                "hidden_dim": 24,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "vocab_stoi": vocab.stoi,
            "epoch": 0,
        },
        path,
    )


def test_student_score_one_schema(tmp_path):
    ckpt = tmp_path / "student.pt"
    _make_ckpt(str(ckpt))
    scorer = StudentScorer(
        checkpoint_path=str(ckpt),
        window=JunctionWindowConfig(left=2, right=2),
        device="cpu",
    )
    score = scorer.score_one(sequence_aa="ACDEFGHIKLM", domain_lengths=[1] * 11)
    assert isinstance(score.sequence_hash, str)
    assert isinstance(score.global_score, float)
    assert len(score.junction_scores) == 10
    assert isinstance(score.junction_mean, float)
    assert isinstance(score.junction_min, float)


def test_student_score_batch_rows_schema(tmp_path):
    ckpt = tmp_path / "student.pt"
    _make_ckpt(str(ckpt))
    scorer = StudentScorer(
        checkpoint_path=str(ckpt),
        window=JunctionWindowConfig(left=2, right=2),
        device="cpu",
    )
    df = pd.DataFrame([{"combo_compact": "AAAAAAAAAAA", "sequence_aa": "ACDEFGHIKLM"}])
    out = scorer.score_batch_rows(rows_df=df, validated_domains=None)
    assert "sequence_hash" in out.columns
    assert "global_score" in out.columns
    assert "junction_mean" in out.columns
    assert "junction_min" in out.columns
    for i in range(1, 11):
        assert f"junction_{i:02d}" in out.columns

