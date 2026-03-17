from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.composition.assistant_ranker import AssistantRankerScorer
from cas12a_shuffling_model.composition.chimera_repr import canonicalize_chimera_table
from cas12a_shuffling_model.composition.table_io import read_table, write_table
from cas12a_shuffling_model.search.combo_compact import build_sequence_from_combo


@dataclass(frozen=True)
class RerankConfig:
    batch_size: int = 2048
    top_k: int = 50
    include_sequence: bool = False


def rerank_shortlist(
    *,
    shortlist_table: str,
    assistant_checkpoint: str,
    out_dir: str,
    cfg: RerankConfig,
    validated_domains: dict[tuple[str, int], str] | None = None,
    device: str | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shortlist = read_table(shortlist_table)
    canonical = canonicalize_chimera_table(shortlist, validated_domains=None, require_sequence=False)
    scorer = AssistantRankerScorer(assistant_checkpoint, device=device)
    scored = scorer.score_dataframe(canonical, batch_size=int(cfg.batch_size))

    if bool(cfg.include_sequence) and validated_domains is not None:
        scored["full_protein_sequence"] = scored["slot_code_11"].map(
            lambda code: build_sequence_from_combo(str(code), validated_domains)
        )

    scored = scored.sort_values("assistant_score", ascending=False).reset_index(drop=True)
    scored["assistant_rank"] = (scored.index + 1).astype(int)
    top = scored.head(int(cfg.top_k)).copy().reset_index(drop=True)
    top["final_rank"] = (top.index + 1).astype(int)

    all_path = out_path / "assistant_reranked_all.csv"
    top_path = out_path / "assistant_reranked_top.csv"
    scored.to_csv(all_path, index=False)
    top.to_csv(top_path, index=False)

    meta = {
        "shortlist_table": shortlist_table,
        "assistant_checkpoint": assistant_checkpoint,
        "n_shortlist": int(len(scored)),
        "top_k": int(cfg.top_k),
    }
    meta_path = out_path / "assistant_rerank_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "all_csv": str(all_path),
        "top_csv": str(top_path),
        "meta_json": str(meta_path),
    }
