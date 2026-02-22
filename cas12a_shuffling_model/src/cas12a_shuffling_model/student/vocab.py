from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "X"


@dataclass(frozen=True)
class AminoAcidVocab:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, seq_aa: str) -> List[int]:
        out: List[int] = []
        for ch in str(seq_aa).strip().upper():
            out.append(self.stoi.get(ch, self.unk_id))
        return out

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), UNK_TOKEN) for i in token_ids)


def build_default_vocab() -> AminoAcidVocab:
    # 20 canonical + common ambiguous/special amino-acid letters in protein datasets.
    letters = [
        PAD_TOKEN,
        BOS_TOKEN,
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "B",
        "J",
        "O",
        "U",
        "Z",
        UNK_TOKEN,
    ]
    stoi = {tok: i for i, tok in enumerate(letters)}
    itos = {i: tok for tok, i in stoi.items()}
    return AminoAcidVocab(stoi=stoi, itos=itos)

