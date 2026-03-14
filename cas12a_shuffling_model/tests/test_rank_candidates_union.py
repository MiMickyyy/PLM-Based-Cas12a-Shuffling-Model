import pandas as pd

from cas12a_shuffling_model.cli.rank_candidates import (
    _build_union_shortlist_from_metric_heaps,
    _select_rerank_candidates_union,
)


def test_build_union_shortlist_from_metric_heaps_deduplicates_combo():
    row_a = {"combo_compact": "AAAAAAAAAAA", "student_rank_score": 1.0, "global_score": 1.0}
    row_b = {"combo_compact": "LLLLLLLLLLL", "student_rank_score": 0.8, "global_score": 0.9}
    metric_heaps = {
        "student_rank_score": [(1.0, "AAAAAAAAAAA", row_a)],
        "global_score": [(0.9, "LLLLLLLLLLL", row_b), (1.0, "AAAAAAAAAAA", row_a)],
    }
    out = _build_union_shortlist_from_metric_heaps(metric_heaps)
    assert len(out) == 2
    assert set(out["combo_compact"].tolist()) == {"AAAAAAAAAAA", "LLLLLLLLLLL"}
    src_map = dict(zip(out["combo_compact"], out["shortlist_sources"]))
    assert src_map["AAAAAAAAAAA"] == "global_score,student_rank_score"
    assert src_map["LLLLLLLLLLL"] == "global_score"


def test_select_rerank_candidates_union_interleaves_metrics():
    df = pd.DataFrame(
        [
            {"combo_compact": "AAAAAAAAAAA", "student_rank_score": 0.99, "global_score": 0.10, "junction_mean": 0.10, "junction_min": 0.10},
            {"combo_compact": "LLLLLLLLLLL", "student_rank_score": 0.10, "global_score": 0.99, "junction_mean": 0.10, "junction_min": 0.10},
            {"combo_compact": "FFFFFFFFFFF", "student_rank_score": 0.10, "global_score": 0.10, "junction_mean": 0.99, "junction_min": 0.10},
            {"combo_compact": "MMMMMMMMMMM", "student_rank_score": 0.10, "global_score": 0.10, "junction_mean": 0.10, "junction_min": 0.99},
            {"combo_compact": "ALFMALFMALF", "student_rank_score": 0.50, "global_score": 0.50, "junction_mean": 0.50, "junction_min": 0.50},
        ]
    )
    out = _select_rerank_candidates_union(
        shortlist_df=df,
        teacher_rerank_size=4,
        metrics=("student_rank_score", "global_score", "junction_mean", "junction_min"),
    )
    combos = set(out["combo_compact"].tolist())
    assert combos == {"AAAAAAAAAAA", "LLLLLLLLLLL", "FFFFFFFFFFF", "MMMMMMMMMMM"}
