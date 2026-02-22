import pytest

from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    decode_combo_compact,
    encode_combo_compact,
    validate_combo_compact,
)
from cas12a_shuffling_model.search.sampler import combo_from_index, sample_combo_compacts


def test_combo_validate_ok():
    combo = validate_combo_compact("ALFMALFMALF")
    assert combo == "ALFMALFMALF"


def test_combo_validate_rejects_bad_chars():
    with pytest.raises(ValueError):
        validate_combo_compact("ALFMALFMALX")


def test_combo_encode_decode_roundtrip():
    parents = ["As", "Lb", "Fn", "Mb2", "As", "Lb", "Fn", "Mb2", "As", "Lb", "Fn"]
    combo = encode_combo_compact(parents)
    assert combo == "ALFMALFMALF"
    assert decode_combo_compact(combo) == parents


def test_build_sequence_from_combo():
    combo = "ALFMALFMALF"
    domains = {}
    for idx, parent in enumerate(decode_combo_compact(combo), start=1):
        domains[(parent, idx)] = f"P{idx}"
    seq = build_sequence_from_combo(combo, domains)
    assert seq == "P1P2P3P4P5P6P7P8P9P10P11"


def test_sample_combo_compacts_deterministic():
    a = sample_combo_compacts(n=5, seed=123)
    b = sample_combo_compacts(n=5, seed=123)
    assert a == b
    assert all(len(x) == 11 for x in a)


def test_combo_from_index_deterministic_and_order():
    assert combo_from_index(0) == "AAAAAAAAAAA"
    assert combo_from_index(1) == "AAAAAAAAAAL"
    assert combo_from_index(3) == "AAAAAAAAAAM"
    assert combo_from_index(4) == "AAAAAAAAALA"
