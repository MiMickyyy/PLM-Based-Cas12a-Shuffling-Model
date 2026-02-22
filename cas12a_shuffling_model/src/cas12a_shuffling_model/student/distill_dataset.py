from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)
from cas12a_shuffling_model.student.vocab import AminoAcidVocab


def _to_float_or_nan(x: object) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


@dataclass(frozen=True)
class DistillRecord:
    combo_compact: str
    sequence_aa: str
    sequence_hash: str
    domain_lengths: list[int]
    teacher_global: float
    teacher_junctions: list[float]


class DistillDataset(Dataset):
    def __init__(self, records: Sequence[DistillRecord], vocab: AminoAcidVocab):
        self.records = list(records)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        token_ids = self.vocab.encode(rec.sequence_aa)
        if len(token_ids) == 0:
            raise ValueError("Empty sequence in distillation dataset")
        input_ids = [self.vocab.bos_id] + token_ids[:-1]
        target_ids = token_ids
        return {
            "combo_compact": rec.combo_compact,
            "sequence_aa": rec.sequence_aa,
            "sequence_hash": rec.sequence_hash,
            "domain_lengths": rec.domain_lengths,
            "teacher_global": rec.teacher_global,
            "teacher_junctions": rec.teacher_junctions,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "length": len(target_ids),
        }


def collate_distill_batch(batch: list[dict], pad_id: int) -> dict:
    max_len = max(item["length"] for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), fill_value=pad_id, dtype=torch.long)
    target_ids = torch.full((bsz, max_len), fill_value=pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)

    teacher_global = torch.tensor([item["teacher_global"] for item in batch], dtype=torch.float32)
    teacher_junctions = torch.tensor(
        [item["teacher_junctions"] for item in batch], dtype=torch.float32
    )

    domain_lengths = []
    combos = []
    seqs = []
    hashes = []
    lengths = []
    for i, item in enumerate(batch):
        n = item["length"]
        lengths.append(n)
        input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        target_ids[i, :n] = torch.tensor(item["target_ids"], dtype=torch.long)
        mask[i, :n] = True
        domain_lengths.append(item["domain_lengths"])
        combos.append(item["combo_compact"])
        seqs.append(item["sequence_aa"])
        hashes.append(item["sequence_hash"])

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "mask": mask,
        "lengths": lengths,
        "domain_lengths": domain_lengths,
        "teacher_global": teacher_global,
        "teacher_junctions": teacher_junctions,
        "combo_compact": combos,
        "sequence_aa": seqs,
        "sequence_hash": hashes,
    }


def load_distill_records_from_csv(
    *,
    csv_path: str,
    validated_domains: dict[tuple[str, int], str] | None = None,
) -> list[DistillRecord]:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("Distillation CSV is empty")

    required = {"global_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in distill CSV: {sorted(missing)}")

    rows: list[DistillRecord] = []
    for _, row in df.iterrows():
        combo = ""
        if "combo_compact" in df.columns and pd.notna(row.get("combo_compact")):
            combo = validate_combo_compact(str(row["combo_compact"]))

        seq = str(row.get("sequence_aa", "")).strip().upper()
        if not seq:
            if not combo or validated_domains is None:
                raise ValueError(
                    "Row has no sequence_aa and cannot reconstruct (combo_compact + validated_domains needed)"
                )
            seq = build_sequence_from_combo(combo, validated_domains)

        if combo and validated_domains is not None:
            domain_lengths = domain_lengths_from_combo(combo, validated_domains)
        else:
            L = len(seq)
            domain_lengths = [L // 11] * 11
            domain_lengths[-1] += max(0, L - sum(domain_lengths))

        jcols = [f"junction_{i:02d}" for i in range(1, 11)]
        teacher_junctions = []
        for c in jcols:
            teacher_junctions.append(_to_float_or_nan(row.get(c)))

        seq_hash = str(row.get("sequence_hash", "")).strip()
        if not seq_hash:
            seq_hash = ""

        rows.append(
            DistillRecord(
                combo_compact=combo,
                sequence_aa=seq,
                sequence_hash=seq_hash,
                domain_lengths=domain_lengths,
                teacher_global=_to_float_or_nan(row.get("global_score")),
                teacher_junctions=teacher_junctions,
            )
        )
    return rows


def split_indices(n: int, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if n < 2:
        raise ValueError("Need at least 2 examples for train/val split")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n - 1, n_val)
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return train_idx, val_idx

