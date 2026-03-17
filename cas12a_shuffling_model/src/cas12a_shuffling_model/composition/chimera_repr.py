from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from cas12a_shuffling_model.io.loaders import sha256_text
from cas12a_shuffling_model.search.combo_compact import (
    ALLOWED_COMBO_LETTERS,
    COMBO_SLOTS,
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)

SLOT_ALPHABET: tuple[str, ...] = ALLOWED_COMBO_LETTERS
SLOT_COUNT: int = COMBO_SLOTS
SLOT_TO_INT = {ch: i for i, ch in enumerate(SLOT_ALPHABET)}
INT_TO_SLOT = {i: ch for ch, i in SLOT_TO_INT.items()}


def slot_columns(prefix: str = "slot_") -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(1, SLOT_COUNT + 1)]


def validate_slot_code_11(slot_code_11: str) -> str:
    return validate_combo_compact(slot_code_11, slots=SLOT_COUNT)


def slot_code_to_int_array(slot_code_11: str) -> np.ndarray:
    code = validate_slot_code_11(slot_code_11)
    return np.asarray([SLOT_TO_INT[ch] for ch in code], dtype=np.int64)


def slot_int_array_to_code(slot_ints: Sequence[int]) -> str:
    if len(slot_ints) != SLOT_COUNT:
        raise ValueError(f"slot_ints length must be {SLOT_COUNT}")
    chars = []
    for x in slot_ints:
        i = int(x)
        if i not in INT_TO_SLOT:
            raise ValueError(f"slot integer out of range [0,3]: {i}")
        chars.append(INT_TO_SLOT[i])
    return "".join(chars)


def slot_matrix_to_codes(slot_matrix: np.ndarray) -> list[str]:
    arr = np.asarray(slot_matrix, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != SLOT_COUNT:
        raise ValueError(f"slot_matrix must be [N,{SLOT_COUNT}]")
    return [slot_int_array_to_code(row.tolist()) for row in arr]


def chimera_id_from_slot_code(slot_code_11: str) -> str:
    code = validate_slot_code_11(slot_code_11)
    return f"chimera_{sha256_text(code)[:16]}"


def _parse_code_from_row(
    row: pd.Series,
    *,
    slot_code_col: str = "slot_code_11",
    combo_col: str = "combo_compact",
    letter_slot_cols: Sequence[str] | None = None,
) -> str:
    if slot_code_col in row and pd.notna(row.get(slot_code_col)):
        raw = str(row.get(slot_code_col)).strip().upper()
        if raw and raw != "NAN":
            return validate_slot_code_11(raw)

    if combo_col in row and pd.notna(row.get(combo_col)):
        raw = str(row.get(combo_col)).strip().upper()
        if raw and raw != "NAN":
            return validate_slot_code_11(raw)

    cols = list(letter_slot_cols or [])
    if not cols:
        numeric = []
        for i in range(1, SLOT_COUNT + 1):
            if str(i) in row.index:
                numeric.append(str(i))
            elif i in row.index:
                numeric.append(i)
        cols = numeric
    if len(cols) == SLOT_COUNT:
        code = "".join(str(row[c]).strip().upper() for c in cols)
        return validate_slot_code_11(code)

    raise ValueError("Row missing slot code fields (slot_code_11/combo_compact/11 slot columns)")


def canonicalize_chimera_table(
    df: pd.DataFrame,
    *,
    validated_domains: dict[tuple[str, int], str] | None = None,
    slot_code_col: str = "slot_code_11",
    combo_col: str = "combo_compact",
    sequence_col: str = "sequence_aa",
    letter_slot_cols: Sequence[str] | None = None,
    require_sequence: bool = True,
) -> pd.DataFrame:
    out_rows: list[dict] = []
    slot_cols = slot_columns()

    for _, row in df.iterrows():
        code = _parse_code_from_row(
            row,
            slot_code_col=slot_code_col,
            combo_col=combo_col,
            letter_slot_cols=letter_slot_cols,
        )
        seq = ""
        if sequence_col in df.columns and pd.notna(row.get(sequence_col)):
            seq = str(row.get(sequence_col)).strip().upper()
        if not seq and validated_domains is not None:
            seq = build_sequence_from_combo(code, validated_domains)
        if not seq and bool(require_sequence):
            if validated_domains is None:
                raise ValueError(
                    "sequence_aa missing and validated_domains is None; cannot reconstruct sequence"
                )
        if validated_domains is not None:
            dlen = domain_lengths_from_combo(code, validated_domains)
        else:
            L = len(seq)
            dlen = [L // SLOT_COUNT] * SLOT_COUNT
            dlen[-1] += max(0, L - sum(dlen))

        ints = slot_code_to_int_array(code)
        rec = row.to_dict()
        rec["chimera_id"] = chimera_id_from_slot_code(code)
        rec["slot_code_11"] = code
        rec["combo_compact"] = code
        for i, col in enumerate(slot_cols):
            rec[col] = int(ints[i])
        rec["full_protein_sequence"] = seq
        rec["sequence_aa"] = seq
        rec["length"] = len(seq) if seq else np.nan
        rec["domain_lengths"] = ",".join(str(int(x)) for x in dlen)
        out_rows.append(rec)
    out = pd.DataFrame(out_rows)
    ordered = ["chimera_id", "slot_code_11", "combo_compact", *slot_cols, "full_protein_sequence", "length"]
    head = [c for c in ordered if c in out.columns]
    tail = [c for c in out.columns if c not in head]
    return out[head + tail]


def sample_slot_codes(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    total = len(SLOT_ALPHABET) ** SLOT_COUNT
    if n > total:
        raise ValueError(f"n={n} exceeds search space size={total}")
    idx = rng.choice(total, size=n, replace=False)
    idx = np.sort(idx)
    mat = index_to_slot_matrix(idx)
    return slot_matrix_to_codes(mat)


def iter_all_slot_codes() -> Iterable[str]:
    for tup in itertools.product(SLOT_ALPHABET, repeat=SLOT_COUNT):
        yield "".join(tup)


def index_to_slot_matrix(indices: np.ndarray | Sequence[int]) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0, SLOT_COUNT), dtype=np.int64)
    out = np.zeros((idx.shape[0], SLOT_COUNT), dtype=np.int64)
    x = idx.copy()
    base = len(SLOT_ALPHABET)
    for pos in range(SLOT_COUNT - 1, -1, -1):
        out[:, pos] = x % base
        x //= base
    return out


def enumerate_slot_matrix_batches(
    *,
    batch_size: int,
    start: int = 0,
    stop: int | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    total = len(SLOT_ALPHABET) ** SLOT_COUNT
    begin = max(0, int(start))
    end = total if stop is None else min(total, int(stop))
    if begin >= end:
        return
    for s in range(begin, end, int(batch_size)):
        e = min(end, s + int(batch_size))
        idx = np.arange(s, e, dtype=np.int64)
        yield idx, index_to_slot_matrix(idx)


def load_active_code_counts(
    path: str | Path,
    *,
    slot_code_col: str = "slot_code_11",
    combo_col: str = "combo_compact",
    letter_slot_cols: Sequence[str] | None = None,
) -> dict[str, int]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif ext in {".csv", ".txt"}:
        df = pd.read_csv(p)
    elif ext in {".tsv"}:
        df = pd.read_csv(p, sep="\t")
    elif ext == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported active table format: {p}")

    counts: Counter[str] = Counter()
    for _, row in df.iterrows():
        try:
            code = _parse_code_from_row(
                row,
                slot_code_col=slot_code_col,
                combo_col=combo_col,
                letter_slot_cols=letter_slot_cols,
            )
        except Exception:
            continue
        counts[code] += 1
    return dict(counts)
