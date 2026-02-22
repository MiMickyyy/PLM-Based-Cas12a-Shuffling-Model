from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    validate_combo_compact,
)

logger = logging.getLogger(__name__)


def load_validated_domains_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"parent", "slot", "aa_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"validated domains missing columns: {sorted(missing)}")
    return df


def validated_domains_to_dict(df: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    out: Dict[Tuple[str, int], str] = {}
    for _, r in df.iterrows():
        out[(str(r["parent"]), int(r["slot"]))] = str(r["aa_sequence"])
    return out


def _is_letter(x: object, allowed: set[str]) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip().upper()
    return s in allowed


def extract_active_rows(
    df: pd.DataFrame, *, slot_columns: List[str] | None, allowed_letters: List[str]
) -> Tuple[pd.DataFrame, List[object]]:
    allowed = {c.upper() for c in allowed_letters}
    if slot_columns is None:
        slot_columns = list(df.columns[:11])
    if len(slot_columns) != 11:
        raise ValueError("slot_columns must have length 11")

    # If user supplied columns as strings, try to map them onto actual column labels.
    resolved_cols: List[object] = []
    for c in slot_columns:
        if c in df.columns:
            resolved_cols.append(c)
            continue
        if isinstance(c, str) and c.isdigit():
            ci = int(c)
            if ci in df.columns:
                resolved_cols.append(ci)
                continue
        raise KeyError(f"Slot column not found: {c!r}")

    sub = df[resolved_cols]
    mask = sub.apply(lambda col: col.map(lambda v: _is_letter(v, allowed))).all(axis=1)
    active = df.loc[mask].copy().reset_index(drop=True)
    return active, resolved_cols


def reconstruct_chimeras(
    active_df: pd.DataFrame,
    slot_columns: List[object],
    domains: Dict[Tuple[str, int], str],
) -> pd.DataFrame:
    rows = []
    for idx, r in active_df.iterrows():
        letters = [str(r[c]).strip().upper() for c in slot_columns]
        combo = validate_combo_compact("".join(letters))
        aa = build_sequence_from_combo(combo, domains)
        out = {"combo_compact": combo, "sequence_aa": aa, "aa_length": len(aa)}
        rows.append(out)
    out_df = pd.concat([active_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out_df
