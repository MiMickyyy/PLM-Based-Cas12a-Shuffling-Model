from pathlib import Path

from cas12a_shuffling_model.teacher.junction_scoring import (
    JunctionWindowConfig,
    compute_boundary_positions,
    score_windows_from_per_residue_ll,
)
from cas12a_shuffling_model.teacher.score_cache import ScoreCache


def test_compute_boundary_positions():
    boundaries = compute_boundary_positions([1] * 11)
    assert boundaries == list(range(1, 11))


def test_score_windows_from_per_residue_ll_smoke():
    per_residue_ll = [None] + [1.0] * 99
    boundaries = [10]
    scores = score_windows_from_per_residue_ll(
        per_residue_ll, boundaries, JunctionWindowConfig(left=2, right=2)
    )
    assert scores == [1.0]


def test_score_cache_roundtrip(tmp_path: Path):
    cache = ScoreCache(tmp_path / "cache.sqlite")
    cache.set(
        key="k1",
        seq_hash="h",
        seq_len=3,
        global_score=-1.23,
        junction_scores=[-0.1] * 10,
        meta={"a": 1},
    )
    got = cache.get("k1")
    assert got is not None
    assert got.seq_hash == "h"
    assert got.seq_len == 3
    assert got.global_score == -1.23
    assert got.junction_scores == [-0.1] * 10
    assert got.meta["a"] == 1

