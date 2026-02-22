from __future__ import annotations

from typing import List, Sequence

ALLOWED_COMBO_LETTERS: tuple[str, ...] = ("A", "L", "F", "M")
COMBO_SLOTS: int = 11

LETTER_TO_PARENT = {"A": "As", "L": "Lb", "F": "Fn", "M": "Mb2"}
PARENT_TO_LETTER = {"As": "A", "Lb": "L", "Fn": "F", "Mb2": "M", "Mb": "M"}


def validate_combo_compact(combo: str, *, slots: int = COMBO_SLOTS) -> str:
    c = str(combo).strip().upper()
    if len(c) != slots:
        raise ValueError(f"combo_compact must have length {slots}, got {len(c)}")
    bad = sorted({ch for ch in c if ch not in ALLOWED_COMBO_LETTERS})
    if bad:
        raise ValueError(
            f"combo_compact contains invalid letters: {bad}; allowed={ALLOWED_COMBO_LETTERS}"
        )
    return c


def decode_combo_compact(combo: str) -> List[str]:
    c = validate_combo_compact(combo)
    return [LETTER_TO_PARENT[ch] for ch in c]


def encode_combo_compact(parents: Sequence[str], *, slots: int = COMBO_SLOTS) -> str:
    if len(parents) != slots:
        raise ValueError(f"Need {slots} parent slots, got {len(parents)}")
    letters = []
    for p in parents:
        key = str(p).strip()
        if key not in PARENT_TO_LETTER:
            raise ValueError(f"Unknown parent label: {p}")
        letters.append(PARENT_TO_LETTER[key])
    return "".join(letters)


def combo_to_domain_keys(combo: str) -> List[tuple[str, int]]:
    parents = decode_combo_compact(combo)
    return [(parent, idx) for idx, parent in enumerate(parents, start=1)]


def build_sequence_from_combo(
    combo: str,
    validated_domains: dict[tuple[str, int], str],
) -> str:
    parts = []
    for key in combo_to_domain_keys(combo):
        if key not in validated_domains:
            raise KeyError(f"Missing validated domain for key={key}")
        parts.append(validated_domains[key])
    return "".join(parts)


def domain_lengths_from_combo(
    combo: str,
    validated_domains: dict[tuple[str, int], str],
) -> List[int]:
    return [len(validated_domains[key]) for key in combo_to_domain_keys(combo)]
