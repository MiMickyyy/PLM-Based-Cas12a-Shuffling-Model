from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.domain.chimera_builder import (
    load_validated_domains_table,
    validated_domains_to_dict,
)
from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)
from cas12a_shuffling_model.teacher.junction_scoring import JunctionWindowConfig
from cas12a_shuffling_model.teacher.protgpt2_scorer import ProtGPT2Config, ProtGPT2Scorer
from cas12a_shuffling_model.teacher.score_cache import ScoreCache

logger = logging.getLogger(__name__)


def resolve_validated_domains_path(config: dict, cli_path: str | None = None) -> str:
    if cli_path:
        return cli_path
    out_processed = config.get("paths", {}).get("out_processed_dir")
    if out_processed:
        return str(Path(out_processed) / "validated_domain_peptides.csv")
    raise ValueError("validated domains path not found in config or CLI")


def build_teacher_scorer_from_config(config: dict, *, device: str | None = None) -> ProtGPT2Scorer:
    teacher_cfg = config.get("teacher", {})
    window_cfg = teacher_cfg.get("junction_window", {})
    cache_path = teacher_cfg.get("cache_sqlite")
    if not cache_path:
        out_processed = config.get("paths", {}).get("out_processed_dir")
        if not out_processed:
            raise ValueError("teacher.cache_sqlite is missing and no paths.out_processed_dir found")
        cache_path = str(Path(out_processed) / "teacher_scores.sqlite")

    scorer = ProtGPT2Scorer(
        config=ProtGPT2Config(
            model_name=str(teacher_cfg.get("model_name", "nferruz/ProtGPT2")),
            add_spaces=bool(teacher_cfg.get("add_spaces", True)),
            max_length=teacher_cfg.get("max_length"),
        ),
        window=JunctionWindowConfig(
            left=int(window_cfg.get("left", 25)),
            right=int(window_cfg.get("right", 25)),
        ),
        cache=ScoreCache(cache_path),
        device=device,
    )
    return scorer


def score_rows_with_teacher(
    *,
    rows_df: pd.DataFrame,
    scorer: ProtGPT2Scorer,
    validated_domains: dict[tuple[str, int], str] | None = None,
    combo_col: str = "combo_compact",
    seq_col: str = "sequence_aa",
    batch_size: int = 4,
) -> pd.DataFrame:
    prepared_rows: list[dict] = []
    seqs: list[str] = []
    dlen_list: list[list[int] | None] = []
    for _, row in rows_df.iterrows():
        combo = None
        if combo_col in rows_df.columns and pd.notna(row[combo_col]):
            combo = validate_combo_compact(str(row[combo_col]))

        seq = str(row[seq_col]).strip().upper() if seq_col in rows_df.columns and pd.notna(row[seq_col]) else ""
        if not seq:
            if combo is None or validated_domains is None:
                raise ValueError("Row missing sequence_aa and cannot reconstruct without combo+validated_domains")
            seq = build_sequence_from_combo(combo, validated_domains)

        domain_lengths = None
        if combo is not None and validated_domains is not None:
            domain_lengths = domain_lengths_from_combo(combo, validated_domains)

        prepared_rows.append(row.to_dict())
        seqs.append(seq)
        dlen_list.append(domain_lengths)

    scores = scorer.score_many(
        seqs_aa=seqs,
        domain_lengths_list=dlen_list,
        batch_size=int(batch_size),
    )

    out_rows = []
    for row_dict, seq, score in zip(prepared_rows, seqs, scores):
        rec = dict(row_dict)
        rec["sequence_aa"] = seq
        rec["sequence_hash"] = score.seq_hash
        rec["global_score"] = score.global_score
        rec["junction_mean"] = score.junction_mean
        rec["junction_min"] = score.junction_min
        rec["teacher_cache_hit"] = bool(score.from_cache)
        for i, v in enumerate(score.junction_scores, start=1):
            rec[f"junction_{i:02d}"] = v
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def load_validated_domains_dict(path: str) -> dict[tuple[str, int], str]:
    domains_df = load_validated_domains_table(path)
    return validated_domains_to_dict(domains_df)
