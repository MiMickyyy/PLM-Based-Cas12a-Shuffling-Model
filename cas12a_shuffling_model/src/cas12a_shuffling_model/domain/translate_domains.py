from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio.Seq import Seq

from cas12a_shuffling_model.domain.align_validate import (
    LocalAlignmentResult,
    best_local_alignment,
    build_local_aligner,
)
from cas12a_shuffling_model.io.loaders import DomainId, iter_domain_files, read_snapgene_dna_sequence

logger = logging.getLogger(__name__)


DNA_ALPHABET = set("ACGTN")


def _sanitize_dna(seq: str) -> str:
    seq = seq.upper()
    return "".join([c for c in seq if c in DNA_ALPHABET])


@dataclass(frozen=True)
class DomainCandidate:
    parent: str
    slot: int
    dna_path: str
    orientation: str  # "fwd" or "rev"
    frame: int  # 0/1/2
    aa_raw: str
    aa_trimmed: str
    query_start: int
    query_end: int
    parent_start: int
    parent_end: int
    align_score: float
    adjusted_score: float
    stop_count: int
    x_fraction: float


def _translate(seq_nt: str, frame: int) -> str:
    # Avoid partial codon warnings by trimming to a multiple of 3.
    start = frame
    end = len(seq_nt) - ((len(seq_nt) - start) % 3)
    return str(Seq(seq_nt[start:end]).translate(to_stop=False))


def _score_adjustment(align_score: float, aa_trimmed: str) -> Tuple[float, int, float]:
    stop_count = aa_trimmed.count("*")
    x_count = aa_trimmed.count("X")
    x_fraction = (x_count / max(1, len(aa_trimmed))) if aa_trimmed else 1.0
    # Penalize stops heavily; penalize uncertain 'X' moderately.
    adjusted = align_score - 50.0 * stop_count - 5.0 * x_count
    return adjusted, stop_count, x_fraction


def make_candidates_for_domain(
    *,
    domain_id: DomainId,
    dna_path: Path,
    parent_protein: str,
    aligner,
    try_reverse_complement: bool,
) -> List[DomainCandidate]:
    dna_seq = _sanitize_dna(read_snapgene_dna_sequence(dna_path))
    candidates: List[DomainCandidate] = []

    orientations = [("fwd", dna_seq)]
    if try_reverse_complement:
        orientations.append(("rev", str(Seq(dna_seq).reverse_complement())))

    for orientation, seq_nt in orientations:
        for frame in (0, 1, 2):
            aa_raw = _translate(seq_nt, frame=frame)
            aln: LocalAlignmentResult | None = best_local_alignment(aligner, parent_protein, aa_raw)
            if aln is None:
                continue
            aa_trimmed = aa_raw[aln.query_start : aln.query_end]
            adjusted, stop_count, x_fraction = _score_adjustment(aln.score, aa_trimmed)
            candidates.append(
                DomainCandidate(
                    parent=domain_id.parent,
                    slot=domain_id.slot,
                    dna_path=str(dna_path),
                    orientation=orientation,
                    frame=frame,
                    aa_raw=aa_raw,
                    aa_trimmed=aa_trimmed,
                    query_start=aln.query_start,
                    query_end=aln.query_end,
                    parent_start=aln.target_start,
                    parent_end=aln.target_end,
                    align_score=aln.score,
                    adjusted_score=adjusted,
                    stop_count=stop_count,
                    x_fraction=x_fraction,
                )
            )

    candidates.sort(key=lambda c: c.adjusted_score, reverse=True)
    return candidates


def _select_order_consistent_candidates(
    *,
    per_slot_candidates: Dict[int, List[DomainCandidate]],
    top_k_per_domain: int,
    allowed_parent_overlap_aa: int,
) -> Dict[int, DomainCandidate]:
    # Simple DP to enforce increasing parent_start/parent_end across slots.
    slots = sorted(per_slot_candidates.keys())
    options = {s: per_slot_candidates[s][:top_k_per_domain] for s in slots}
    dp: Dict[Tuple[int, int], Tuple[float, Tuple[int, int] | None]] = {}
    choice: Dict[Tuple[int, int], DomainCandidate] = {}

    # dp[(slot_idx, opt_idx)] = (best_score_up_to_here, prev_state)
    for i, slot in enumerate(slots):
        for j, cand in enumerate(options[slot]):
            state = (i, j)
            best_here = cand.adjusted_score
            if i == 0:
                dp[state] = (best_here, None)
                choice[state] = cand
                continue
            for pj, pcand in enumerate(options[slots[i - 1]]):
                prev_state = (i - 1, pj)
                if prev_state not in dp:
                    continue
                prev_score, _ = dp[prev_state]
                # Non-decreasing with small allowed overlap.
                if cand.parent_start < pcand.parent_end - allowed_parent_overlap_aa:
                    continue
                score = prev_score + best_here
                if score > dp.get(state, (-1e18, None))[0]:
                    dp[state] = (score, prev_state)
                    choice[state] = cand

    # Backtrack best end state; if DP fails (no valid chain), fall back to best per slot.
    last_i = len(slots) - 1
    end_states = [(s, v[0]) for s, v in dp.items() if s[0] == last_i]
    if not end_states:
        return {s: options[s][0] for s in slots}
    best_state, _ = max(end_states, key=lambda t: t[1])
    selected: Dict[int, DomainCandidate] = {}
    cur = best_state
    while cur is not None:
        i, _ = cur
        slot = slots[i]
        selected[slot] = choice[cur]
        cur = dp[cur][1]
    return {s: selected[s] for s in slots}


def build_validated_domain_table(
    *,
    domains_dir: str | Path,
    parent_proteins: Dict[str, str],
    out_dir: str | Path,
    try_reverse_complement: bool = True,
    top_k_per_domain: int = 5,
    ambiguous_delta_score: float = 25.0,
    enforce_parent_slot_order: bool = True,
    max_allowed_stop_in_trimmed: int = 0,
    allowed_parent_overlap_aa: int = 5,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aligner = build_local_aligner()

    all_candidates: Dict[Tuple[str, int], List[DomainCandidate]] = {}
    for domain_id, dna_path in iter_domain_files(domains_dir):
        if domain_id.parent not in parent_proteins:
            raise KeyError(f"Missing parent protein for {domain_id.parent}")
        cand_list = make_candidates_for_domain(
            domain_id=domain_id,
            dna_path=dna_path,
            parent_protein=parent_proteins[domain_id.parent],
            aligner=aligner,
            try_reverse_complement=try_reverse_complement,
        )
        if not cand_list:
            raise RuntimeError(f"No alignable candidates for {dna_path}")
        filtered = [c for c in cand_list if c.stop_count <= max_allowed_stop_in_trimmed]
        if filtered:
            cand_list = filtered
        all_candidates[(domain_id.parent, domain_id.slot)] = cand_list

    # Select one candidate per parent/slot.
    selected: Dict[Tuple[str, int], DomainCandidate] = {}
    parents = sorted({p for p, _ in all_candidates.keys()})
    for parent in parents:
        per_slot = {slot: all_candidates[(parent, slot)] for slot in range(1, 12)}
        if enforce_parent_slot_order:
            chosen_by_slot = _select_order_consistent_candidates(
                per_slot_candidates=per_slot,
                top_k_per_domain=top_k_per_domain,
                allowed_parent_overlap_aa=allowed_parent_overlap_aa,
            )
        else:
            chosen_by_slot = {slot: per_slot[slot][0] for slot in range(1, 12)}
        for slot, cand in chosen_by_slot.items():
            selected[(parent, slot)] = cand

    rows = []
    for (parent, slot), cand in sorted(selected.items()):
        cand_list = all_candidates[(parent, slot)]
        best = cand_list[0]
        second = cand_list[1] if len(cand_list) > 1 else None
        ambiguous = (
            second is not None
            and (best.adjusted_score - second.adjusted_score) < ambiguous_delta_score
        )
        warnings = []
        if ambiguous:
            warnings.append("ambiguous_top2")
        if cand.stop_count > 0:
            warnings.append("stop_in_trimmed")
        if cand.x_fraction > 0.10:
            warnings.append("high_X_fraction")
        rows.append(
            {
                "parent": parent,
                "slot": slot,
                "dna_path": cand.dna_path,
                "aa_sequence": cand.aa_trimmed.replace("*", ""),
                "aa_length": len(cand.aa_trimmed.replace("*", "")),
                "orientation": cand.orientation,
                "frame": cand.frame,
                "query_start": cand.query_start,
                "query_end": cand.query_end,
                "parent_start": cand.parent_start,
                "parent_end": cand.parent_end,
                "align_score": cand.align_score,
                "adjusted_score": cand.adjusted_score,
                "stop_count": cand.stop_count,
                "x_fraction": cand.x_fraction,
                "ambiguous": ambiguous,
                "warnings": ";".join(warnings) if warnings else "",
            }
        )

    df = pd.DataFrame(rows).sort_values(["parent", "slot"]).reset_index(drop=True)

    # Save top-k candidate interpretations for audit/debugging.
    alt_rows = []
    for (parent, slot), cand_list in sorted(all_candidates.items()):
        for rank, c in enumerate(cand_list[:top_k_per_domain], start=1):
            alt_rows.append(
                {
                    "parent": parent,
                    "slot": slot,
                    "rank": rank,
                    "dna_path": c.dna_path,
                    "orientation": c.orientation,
                    "frame": c.frame,
                    "aa_length_trimmed": len(c.aa_trimmed.replace("*", "")),
                    "parent_start": c.parent_start,
                    "parent_end": c.parent_end,
                    "query_start": c.query_start,
                    "query_end": c.query_end,
                    "align_score": c.align_score,
                    "adjusted_score": c.adjusted_score,
                    "stop_count": c.stop_count,
                    "x_fraction": c.x_fraction,
                }
            )
    df_alt = pd.DataFrame(alt_rows).sort_values(["parent", "slot", "rank"]).reset_index(
        drop=True
    )

    # Parent-slot consistency check
    qc_rows = []
    for parent in parents:
        dfp = df[df["parent"] == parent].sort_values("slot")
        ok = True
        prev_end = -1
        for _, r in dfp.iterrows():
            if int(r["parent_start"]) < prev_end - allowed_parent_overlap_aa:
                ok = False
            prev_end = int(r["parent_end"])
        qc_rows.append({"parent": parent, "slots_increasing": bool(ok)})
    qc = pd.DataFrame(qc_rows)

    df.to_csv(out_dir / "validated_domain_peptides.csv", index=False)
    df_alt.to_csv(out_dir / "domain_candidates_topk.csv", index=False)
    qc.to_csv(out_dir / "domain_translation_qc.csv", index=False)
    json_dict = {
        f"{r.parent}_{r.slot:02d}": r.aa_trimmed.replace("*", "")
        for r in [selected[k] for k in sorted(selected.keys())]
    }
    (out_dir / "validated_domain_peptides.json").write_text(
        json.dumps(json_dict, indent=2), encoding="utf-8"
    )
    return df
