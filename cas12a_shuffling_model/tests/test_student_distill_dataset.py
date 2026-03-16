import pandas as pd
import torch

from cas12a_shuffling_model.student.distill_dataset import (
    DistillRecord,
    DistillDataset,
    collate_distill_batch,
    load_distill_records_from_csv,
    split_indices,
)
from cas12a_shuffling_model.student.train_student import (
    StudentTrainConfig,
    _masked_mse,
    _normalize_teacher_global_scores,
    _pairwise_ranking_loss,
    _weighted_masked_mse,
)
from cas12a_shuffling_model.student.vocab import build_default_vocab


def _make_domains():
    domains = {}
    for slot in range(1, 12):
        domains[("As", slot)] = "A" * slot
    return domains


def test_load_distill_records_and_lengths(tmp_path):
    p = tmp_path / "distill.csv"
    df = pd.DataFrame(
        [
            {
                "combo_compact": "AAAAAAAAAAA",
                "global_score": -1.0,
                **{f"junction_{i:02d}": -2.0 for i in range(1, 11)},
            },
            {
                "combo_compact": "AAAAAAAAAAA",
                "global_score": -1.5,
                **{f"junction_{i:02d}": -2.5 for i in range(1, 11)},
            },
        ]
    )
    df.to_csv(p, index=False)
    records = load_distill_records_from_csv(csv_path=str(p), validated_domains=_make_domains())
    assert len(records) == 2
    assert records[0].sequence_aa == "A" * sum(range(1, 12))
    assert records[0].domain_lengths == list(range(1, 12))


def test_dataset_and_collate(tmp_path):
    p = tmp_path / "distill.csv"
    df = pd.DataFrame(
        [
            {
                "combo_compact": "AAAAAAAAAAA",
                "global_score": -1.0,
                **{f"junction_{i:02d}": -2.0 for i in range(1, 11)},
            },
            {
                "combo_compact": "AAAAAAAAAAA",
                "global_score": -1.5,
                **{f"junction_{i:02d}": -2.5 for i in range(1, 11)},
            },
        ]
    )
    df.to_csv(p, index=False)
    records = load_distill_records_from_csv(csv_path=str(p), validated_domains=_make_domains())
    vocab = build_default_vocab()
    dataset = DistillDataset(records, vocab)
    batch = collate_distill_batch([dataset[0], dataset[1]], pad_id=vocab.pad_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["teacher_junctions"].shape == (2, 10)
    assert len(batch["domain_lengths"]) == 2
    assert batch["source_type"] == ["chimera", "chimera"]
    assert batch["source_is_chimera"].tolist() == [1.0, 1.0]


def test_mixed_source_records_and_masked_junction(tmp_path):
    p = tmp_path / "distill_mixed.csv"
    df = pd.DataFrame(
        [
            {
                "source_type": "chimera",
                "combo_compact": "AAAAAAAAAAA",
                "global_score": -1.0,
                **{f"junction_{i:02d}": -2.0 for i in range(1, 11)},
            },
            {
                "source_type": "natural",
                "combo_compact": "",
                "sequence_aa": "ACDEFGHIKLMNPQRSTVWY",
                "global_score": -1.2,
                **{f"junction_{i:02d}": float("nan") for i in range(1, 11)},
            },
        ]
    )
    df.to_csv(p, index=False)
    records = load_distill_records_from_csv(csv_path=str(p), validated_domains=_make_domains())
    assert records[0].source_type == "chimera"
    assert records[1].source_type == "natural"
    vocab = build_default_vocab()
    dataset = DistillDataset(records, vocab)
    batch = collate_distill_batch([dataset[0], dataset[1]], pad_id=vocab.pad_id)
    assert batch["source_type"] == ["chimera", "natural"]
    assert batch["source_is_chimera"].tolist() == [1.0, 0.0]
    loss = _masked_mse(
        torch.tensor([[0.0] * 10, [1.0] * 10], dtype=torch.float32),
        batch["teacher_junctions"],
    )
    assert float(loss.item()) >= 0.0


def test_split_indices_has_train_and_val():
    train_idx, val_idx = split_indices(10, val_fraction=0.2, seed=13)
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert set(train_idx).isdisjoint(set(val_idx))


def test_split_indices_stratified_by_labels():
    labels = ["natural"] * 8 + ["chimera"] * 12
    train_idx, val_idx = split_indices(20, val_fraction=0.25, seed=13, labels=labels)
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    train_labels = {labels[i] for i in train_idx}
    val_labels = {labels[i] for i in val_idx}
    assert "natural" in train_labels and "chimera" in train_labels
    assert "natural" in val_labels and "chimera" in val_labels


def test_masked_mse_all_nan_returns_zero():
    pred = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    target = torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32)
    loss = _masked_mse(pred, target)
    assert float(loss.item()) == 0.0


def test_weighted_masked_mse_respects_sample_weights():
    pred = torch.tensor([1.0, 3.0], dtype=torch.float32)
    target = torch.tensor([0.0, 3.0], dtype=torch.float32)
    w = torch.tensor([0.0, 1.0], dtype=torch.float32)
    loss = _weighted_masked_mse(pred, target, sample_weights=w)
    assert float(loss.item()) == 0.0


def test_pairwise_ranking_loss_positive_when_order_wrong():
    cfg = StudentTrainConfig(
        pairwise_weight=1.0,
        pairwise_margin=0.1,
        pairwise_pairs_per_batch=8,
        pairwise_min_teacher_diff=0.01,
        pairwise_on_chimera_only=True,
    )
    student = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    teacher = torch.tensor([0.9, 0.8, 0.2, 0.1], dtype=torch.float32)
    source = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    loss = _pairwise_ranking_loss(
        student_global=student,
        teacher_global=teacher,
        source_is_chimera=source,
        cfg=cfg,
    )
    assert float(loss.item()) > 0.0


def test_pairwise_ranking_loss_zero_when_disabled():
    cfg = StudentTrainConfig(pairwise_weight=0.0)
    student = torch.tensor([0.2, 0.3], dtype=torch.float32)
    teacher = torch.tensor([0.9, 0.1], dtype=torch.float32)
    source = torch.tensor([1.0, 1.0], dtype=torch.float32)
    loss = _pairwise_ranking_loss(
        student_global=student,
        teacher_global=teacher,
        source_is_chimera=source,
        cfg=cfg,
    )
    assert float(loss.item()) == 0.0


def test_teacher_global_normalization_uses_source_and_length_bins():
    records = [
        DistillRecord(
            combo_compact="AAAAAAAAAAA",
            source_type="chimera",
            sequence_aa="A" * 100,
            sequence_hash="h1",
            domain_lengths=[10] * 10 + [0],
            teacher_global=-1.0,
            teacher_junctions=[float("nan")] * 10,
        ),
        DistillRecord(
            combo_compact="LLLLLLLLLLL",
            source_type="chimera",
            sequence_aa="A" * 104,
            sequence_hash="h2",
            domain_lengths=[10] * 10 + [4],
            teacher_global=-0.8,
            teacher_junctions=[float("nan")] * 10,
        ),
        DistillRecord(
            combo_compact="",
            source_type="natural",
            sequence_aa="A" * 700,
            sequence_hash="h3",
            domain_lengths=[63] * 10 + [70],
            teacher_global=-3.0,
            teacher_junctions=[float("nan")] * 10,
        ),
        DistillRecord(
            combo_compact="",
            source_type="natural",
            sequence_aa="A" * 740,
            sequence_hash="h4",
            domain_lengths=[67] * 10 + [70],
            teacher_global=-2.9,
            teacher_junctions=[float("nan")] * 10,
        ),
    ]
    cfg = StudentTrainConfig(
        normalize_teacher_global=True,
        normalize_length_bin_size=64,
        normalize_min_group_size=2,
    )
    out = _normalize_teacher_global_scores(records, cfg)
    vals = [r.teacher_global for r in out]
    assert len(vals) == 4
    assert max(vals) - min(vals) < 10.0
    assert all(torch.isfinite(torch.tensor(vals)))
