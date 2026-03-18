from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from cas12a_shuffling_model.composition.active_prior import ensure_active_codes, min_hamming_distance
from cas12a_shuffling_model.composition.chimera_repr import SLOT_ALPHABET, canonicalize_chimera_table
from cas12a_shuffling_model.composition.table_io import read_table, write_table


@dataclass(frozen=True)
class LocalExpandConfig:
    seed_score_col: str = "final_score"
    top_seed_count: int = 500
    active_distance_cap: int = 3
    include_active_seeds: bool = True
    include_hamming_1: bool = True
    include_hamming_2_top_seeds: int = 100
    drop_original_codes: bool = True


def _neighbors_hamming_1(code: str) -> set[str]:
    c = list(code)
    out: set[str] = set()
    for i, ch in enumerate(c):
        for alt in SLOT_ALPHABET:
            if alt == ch:
                continue
            x = c.copy()
            x[i] = alt
            out.add("".join(x))
    return out


def _neighbors_hamming_2(code: str) -> set[str]:
    c = list(code)
    out: set[str] = set()
    for i, j in itertools.combinations(range(len(c)), 2):
        for ai in SLOT_ALPHABET:
            if ai == c[i]:
                continue
            for aj in SLOT_ALPHABET:
                if aj == c[j]:
                    continue
                x = c.copy()
                x[i] = ai
                x[j] = aj
                out.add("".join(x))
    return out


def expand_local_neighborhood(
    *,
    seed_table: str,
    active_codes: Sequence[str],
    out_table: str,
    cfg: LocalExpandConfig = LocalExpandConfig(),
) -> dict[str, object]:
    seed_df = canonicalize_chimera_table(read_table(seed_table), require_sequence=False)
    score_col = cfg.seed_score_col if cfg.seed_score_col in seed_df.columns else "assistant_score"
    if score_col not in seed_df.columns:
        score_col = "s_scan_score" if "s_scan_score" in seed_df.columns else seed_df.columns[0]

    work = seed_df.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work[work["_score"].notna()].copy()
    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    active_set = set(ensure_active_codes(active_codes))

    work["min_hamming_to_active"] = min_hamming_distance(work["slot_code_11"].astype(str).tolist(), list(active_set))
    top_seeds = work.head(int(cfg.top_seed_count)).copy()
    local_seeds = work[work["min_hamming_to_active"] <= int(cfg.active_distance_cap)].copy()
    seeds = pd.concat([top_seeds, local_seeds], ignore_index=True)
    if bool(cfg.include_active_seeds):
        active_seed_rows = pd.DataFrame({"slot_code_11": sorted(active_set)})
        active_seed_rows["_score"] = float("nan")
        active_seed_rows["min_hamming_to_active"] = 0
        seeds = pd.concat([seeds, active_seed_rows], ignore_index=True)
    seeds = seeds.drop_duplicates(subset=["slot_code_11"]).reset_index(drop=True)

    expanded: dict[str, tuple[str, int]] = {}
    for _, row in seeds.iterrows():
        code = str(row["slot_code_11"])
        if bool(cfg.include_hamming_1):
            for n in _neighbors_hamming_1(code):
                expanded.setdefault(n, (code, 1))
    h2_n = max(0, min(int(cfg.include_hamming_2_top_seeds), len(seeds)))
    for _, row in seeds.head(h2_n).iterrows():
        code = str(row["slot_code_11"])
        for n in _neighbors_hamming_2(code):
            if n not in expanded:
                expanded[n] = (code, 2)

    rows = []
    for code, (seed_code, dist_seed) in expanded.items():
        rows.append(
            {
                "slot_code_11": code,
                "combo_compact": code,
                "seed_code": seed_code,
                "distance_from_seed": int(dist_seed),
            }
        )
    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        out_df = pd.DataFrame(columns=["slot_code_11", "combo_compact", "seed_code", "distance_from_seed"])
    out_df["min_hamming_to_active"] = (
        min_hamming_distance(out_df["slot_code_11"].astype(str).tolist(), list(active_set)) if len(out_df) > 0 else []
    )
    if bool(cfg.drop_original_codes) and len(out_df) > 0:
        src_codes = set(seed_df["slot_code_11"].astype(str).tolist())
        out_df = out_df[~out_df["slot_code_11"].astype(str).isin(src_codes)].copy()
    out_df = out_df.sort_values(["min_hamming_to_active", "distance_from_seed", "slot_code_11"]).reset_index(drop=True)
    saved = write_table(out_df, out_table)

    summary = {
        "seed_table": seed_table,
        "out_table": saved,
        "n_seed_rows": int(len(seed_df)),
        "n_selected_seeds": int(len(seeds)),
        "n_expanded_rows": int(len(out_df)),
        "config": asdict(cfg),
    }
    summary_path = Path(saved).with_name("local_expand_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary
