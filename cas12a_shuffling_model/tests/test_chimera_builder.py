import pandas as pd

from cas12a_shuffling_model.domain.chimera_builder import (
    extract_active_rows,
    reconstruct_chimeras,
)


def test_extract_active_rows_uses_first_11_cols_and_filters():
    df = pd.DataFrame(
        {
            1: ["A", "A", "bad"],
            2: ["L", "L", "L"],
            3: ["F", "F", "F"],
            4: ["M", "M", "M"],
            5: ["A", "A", "A"],
            6: ["A", "A", "A"],
            7: ["A", "A", "A"],
            8: ["A", "A", "A"],
            9: ["A", "A", "A"],
            10: ["A", "A", "A"],
            11: ["A", "A", "A"],
            "extra": [1, 2, 3],
        }
    )
    active, slot_cols = extract_active_rows(df, slot_columns=None, allowed_letters=["A", "L", "F", "M"])
    assert slot_cols == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert len(active) == 2


def test_reconstruct_chimeras_concatenates_slots():
    active = pd.DataFrame({i: ["A"] for i in range(1, 12)})
    slot_cols = list(range(1, 12))
    domains = {("As", i): f"X{i}" for i in range(1, 12)}
    out = reconstruct_chimeras(active, slot_cols, domains)
    assert out.loc[0, "combo_compact"] == "A" * 11
    assert out.loc[0, "sequence_aa"] == "".join([f"X{i}" for i in range(1, 12)])

