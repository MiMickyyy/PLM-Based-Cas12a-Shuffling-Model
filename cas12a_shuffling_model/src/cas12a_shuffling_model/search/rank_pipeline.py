from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(1 for x, y in zip(a, b) if x != y)


def greedy_diversity_select(
    df: pd.DataFrame,
    *,
    top_k: int,
    score_col: str = "calibrated_prob",
    combo_col: str = "combo_compact",
    min_hamming: int = 2,
) -> pd.DataFrame:
    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    selected_rows = []
    selected_combos: list[str] = []
    selected_idx: set[int] = set()

    for i, row in ordered.iterrows():
        combo = str(row[combo_col])
        ok = True
        for chosen in selected_combos:
            if hamming_distance(combo, chosen) < min_hamming:
                ok = False
                break
        if ok:
            selected_rows.append(row.to_dict())
            selected_combos.append(combo)
            selected_idx.add(i)
        if len(selected_rows) >= top_k:
            break

    if len(selected_rows) < top_k:
        for i, row in ordered.iterrows():
            if i in selected_idx:
                continue
            selected_rows.append(row.to_dict())
            selected_idx.add(i)
            if len(selected_rows) >= top_k:
                break

    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    return out

