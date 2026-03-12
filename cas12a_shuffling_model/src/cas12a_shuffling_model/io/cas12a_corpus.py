from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
VALID_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO]+")


@dataclass(frozen=True)
class Cas12aSequence:
    seq_id: str
    sequence_aa: str


def clean_aa_sequence(sequence: str) -> str:
    seq = str(sequence).strip().upper().replace("*", "")
    if not seq:
        return ""
    return VALID_AA_RE.sub("", seq)


def read_cas12a_fasta(
    fasta_path: str | Path,
    *,
    min_len: int = 300,
    max_len: int | None = None,
    deduplicate: bool = True,
    limit: int | None = None,
) -> list[Cas12aSequence]:
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"Cas12a FASTA not found: {path}")

    seen: set[str] = set()
    out: list[Cas12aSequence] = []
    n_raw = 0
    n_filtered = 0
    for rec in SeqIO.parse(str(path), "fasta"):
        n_raw += 1
        aa = clean_aa_sequence(str(rec.seq))
        if len(aa) < int(min_len):
            n_filtered += 1
            continue
        if max_len is not None and len(aa) > int(max_len):
            n_filtered += 1
            continue
        if deduplicate and aa in seen:
            continue
        seen.add(aa)
        out.append(Cas12aSequence(seq_id=str(rec.id), sequence_aa=aa))
        if limit is not None and len(out) >= int(limit):
            break

    logger.info(
        "Loaded Cas12a corpus: kept=%d raw=%d filtered=%d deduplicate=%s",
        len(out),
        n_raw,
        n_filtered,
        bool(deduplicate),
    )
    return out


def split_train_val_indices(
    n: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if n < 2:
        raise ValueError("Need at least 2 sequences to split train/val")
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n - 1, n_val)
    val_idx = sorted(idx[:n_val].tolist())
    train_idx = sorted(idx[n_val:].tolist())
    return train_idx, val_idx


def sample_sequences(
    sequences: Sequence[Cas12aSequence],
    *,
    n: int,
    seed: int,
) -> list[Cas12aSequence]:
    if n <= 0:
        return []
    if n >= len(sequences):
        return list(sequences)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(sequences), size=n, replace=False)
    picked = sorted(int(i) for i in idx.tolist())
    return [sequences[i] for i in picked]
