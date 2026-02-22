from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class JunctionWindowConfig:
    left: int = 25
    right: int = 25


def compute_boundary_positions(domain_lengths: Sequence[int]) -> List[int]:
    if len(domain_lengths) != 11:
        raise ValueError("Expected 11 domain lengths for Cas12a shuffling slots.")
    positions = []
    cum = 0
    for i in range(10):
        cum += int(domain_lengths[i])
        positions.append(cum)  # boundary is between cum-1 and cum (0-based)
    return positions


def window_indices(
    *,
    seq_len: int,
    boundary_pos: int,
    window: JunctionWindowConfig,
) -> List[int]:
    # Residue indices (0-based). We will later drop index 0 because LM next-token
    # logprob is only defined for positions >= 1.
    start = max(0, boundary_pos - window.left)
    end = min(seq_len, boundary_pos + window.right)
    return list(range(start, end))


def score_windows_from_per_residue_ll(
    per_residue_ll: Sequence[float | None],
    boundary_positions: Sequence[int],
    window: JunctionWindowConfig,
) -> List[float]:
    seq_len = len(per_residue_ll)
    scores: List[float] = []
    for b in boundary_positions:
        idx = window_indices(seq_len=seq_len, boundary_pos=b, window=window)
        vals = [per_residue_ll[i] for i in idx if i >= 1 and per_residue_ll[i] is not None]
        if not vals:
            scores.append(float("nan"))
        else:
            scores.append(sum(vals) / len(vals))
    return scores

