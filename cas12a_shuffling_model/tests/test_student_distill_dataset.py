import pandas as pd

from cas12a_shuffling_model.student.distill_dataset import (
    DistillDataset,
    collate_distill_batch,
    load_distill_records_from_csv,
    split_indices,
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


def test_split_indices_has_train_and_val():
    train_idx, val_idx = split_indices(10, val_fraction=0.2, seed=13)
    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert set(train_idx).isdisjoint(set(val_idx))

