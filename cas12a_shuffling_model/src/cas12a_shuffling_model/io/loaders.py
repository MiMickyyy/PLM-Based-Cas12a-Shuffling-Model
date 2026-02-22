from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import yaml
from Bio import SeqIO
from Bio.Seq import Seq

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_snapgene_dna_sequence(path: str | Path) -> str:
    rec = SeqIO.read(str(path), "snapgene")
    seq = str(rec.seq).upper()
    return seq


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")


def _extract_longest_run(text: str, allowed_chars: set[str], min_len: int) -> str | None:
    best = ""
    current = []
    for ch in text:
        if ch in allowed_chars:
            current.append(ch)
        else:
            if len(current) > len(best):
                best = "".join(current)
            current = []
    if len(current) > len(best):
        best = "".join(current)
    if len(best) >= min_len:
        return best
    return None


def read_snapgene_protein_sequence(path: str | Path, *, min_len: int = 500) -> str:
    raw = Path(path).read_bytes()
    # Try: sometimes the protein sequence appears as plain ASCII in the binary.
    text = raw.decode("latin-1", errors="ignore").upper()
    seq = _extract_longest_run(text, AA_ALPHABET, min_len=min_len)
    if not seq:
        raise ValueError(f"Failed to extract protein sequence from {path}")
    # Trim any trailing XML/metadata artifacts by stopping at first '<' if present.
    seq = seq.split("<", 1)[0]
    # SnapGene protein exports often end with '*'
    return seq


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_sequence_results_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class DomainId:
    parent: str
    slot: int


DOMAIN_FILE_RE = re.compile(r"^(As|Fn|Lb|Mb)_(\d{2})\.dna$", re.IGNORECASE)


def parse_domain_filename(filename: str) -> DomainId:
    m = DOMAIN_FILE_RE.match(filename)
    if not m:
        raise ValueError(f"Unexpected domain filename: {filename}")
    parent = m.group(1).capitalize()
    slot = int(m.group(2))
    if parent == "Mb":
        parent = "Mb2"
    return DomainId(parent=parent, slot=slot)


def iter_domain_files(domains_dir: str | Path) -> Iterable[Tuple[DomainId, Path]]:
    domains_dir = Path(domains_dir)
    for p in sorted(domains_dir.glob("*.dna")):
        domain_id = parse_domain_filename(p.name)
        yield domain_id, p

