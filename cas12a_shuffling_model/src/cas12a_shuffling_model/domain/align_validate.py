from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalAlignmentResult:
    score: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int


def build_local_aligner(
    *,
    matrix_name: str = "BLOSUM62",
    open_gap_score: float = -10.0,
    extend_gap_score: float = -0.5,
) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load(matrix_name)
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    return aligner


def best_local_alignment(
    aligner: PairwiseAligner, target: str, query: str
) -> Optional[LocalAlignmentResult]:
    if not target or not query:
        return None
    alignments = aligner.align(target, query)
    if len(alignments) == 0:
        return None
    aln = alignments[0]
    blocks_target, blocks_query = aln.aligned
    if len(blocks_target) == 0 or len(blocks_query) == 0:
        return None
    target_start = int(np.min(blocks_target[:, 0]))
    target_end = int(np.max(blocks_target[:, 1]))
    query_start = int(np.min(blocks_query[:, 0]))
    query_end = int(np.max(blocks_query[:, 1]))
    return LocalAlignmentResult(
        score=float(aln.score),
        query_start=query_start,
        query_end=query_end,
        target_start=target_start,
        target_end=target_end,
    )

