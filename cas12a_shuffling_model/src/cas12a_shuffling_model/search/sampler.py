from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Optional, Sequence

from cas12a_shuffling_model.search.combo_compact import (
    ALLOWED_COMBO_LETTERS,
    COMBO_SLOTS,
    validate_combo_compact,
)


def sample_combo_compacts(
    *,
    n: int,
    seed: int,
    slots: int = COMBO_SLOTS,
    alphabet: Sequence[str] = ALLOWED_COMBO_LETTERS,
    fixed_prefix: str | None = None,
) -> List[str]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    combos: set[str] = set()
    prefix = ""
    if fixed_prefix:
        prefix = validate_combo_compact(fixed_prefix, slots=len(fixed_prefix))
        if len(prefix) > slots:
            raise ValueError("fixed_prefix is longer than slots")
    suffix_slots = slots - len(prefix)
    max_space = len(alphabet) ** suffix_slots
    if n > max_space:
        raise ValueError(f"Requested n={n} exceeds search space size={max_space}")

    while len(combos) < n:
        suffix = "".join(rng.choice(alphabet) for _ in range(suffix_slots))
        combos.add(prefix + suffix)
    return sorted(combos)


def enumerate_all_combo_compacts(
    *, slots: int = COMBO_SLOTS, alphabet: Sequence[str] = ALLOWED_COMBO_LETTERS
) -> Iterable[str]:
    for tup in itertools.product(alphabet, repeat=slots):
        yield "".join(tup)


def combo_from_index(
    index: int, *, slots: int = COMBO_SLOTS, alphabet: Sequence[str] = ALLOWED_COMBO_LETTERS
) -> str:
    base = len(alphabet)
    total = base**slots
    if index < 0 or index >= total:
        raise ValueError(f"index out of range: {index}, total={total}")
    digits = [0] * slots
    x = int(index)
    for pos in range(slots - 1, -1, -1):
        digits[pos] = x % base
        x //= base
    return "".join(alphabet[d] for d in digits)


def enumerate_combo_range(
    *,
    start: int,
    end: int,
    slots: int = COMBO_SLOTS,
    alphabet: Sequence[str] = ALLOWED_COMBO_LETTERS,
) -> Iterable[str]:
    for idx in range(start, end):
        yield combo_from_index(idx, slots=slots, alphabet=alphabet)
