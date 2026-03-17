import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.chimera_repr import (
    canonicalize_chimera_table,
    index_to_slot_matrix,
    slot_code_to_int_array,
    slot_int_array_to_code,
    slot_matrix_to_codes,
)
from cas12a_shuffling_model.composition.teacher_export import robust_normalize_teacher_scores


def test_slot_code_roundtrip():
    code = "ALFMALFMALF"
    arr = slot_code_to_int_array(code)
    back = slot_int_array_to_code(arr.tolist())
    assert back == code


def test_index_to_slot_matrix_and_codes():
    idx = np.array([0, 1, 2, 3, 4**11 - 1], dtype=np.int64)
    mat = index_to_slot_matrix(idx)
    codes = slot_matrix_to_codes(mat)
    assert codes[0] == "AAAAAAAAAAA"
    assert codes[1] == "AAAAAAAAAAL"
    assert codes[-1] == "MMMMMMMMMMM"


def test_canonicalize_chimera_table_from_slot_code():
    df = pd.DataFrame(
        [
            {"slot_code_11": "AAAAAAAAAAA", "sequence_aa": "A" * 100},
            {"slot_code_11": "LLLLLLLLLLL", "sequence_aa": "L" * 100},
        ]
    )
    out = canonicalize_chimera_table(df, validated_domains=None)
    assert "chimera_id" in out.columns
    assert out.loc[0, "slot_01"] == 0
    assert out.loc[1, "slot_01"] == 1
    assert out.loc[0, "length"] == 100


def test_robust_normalize_teacher_scores_smoke():
    df = pd.DataFrame(
        {
            "teacher_seq_score_raw": [-2.0, -1.8, -0.5, -0.4],
            "length": [100, 100, 400, 400],
        }
    )
    z = robust_normalize_teacher_scores(df, bin_size=50, min_group_size=2)
    assert len(z) == 4
    assert np.isfinite(z.to_numpy()).all()

