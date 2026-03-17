from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from cas12a_shuffling_model.composition.assistant_ranker import AssistantRankerScorer
from cas12a_shuffling_model.composition.chimera_repr import canonicalize_chimera_table
from cas12a_shuffling_model.composition.table_io import read_table
from cas12a_shuffling_model.search.combo_compact import build_sequence_from_combo
from cas12a_shuffling_model.teacher.protgpt2_scorer import ProtGPT2Scorer
from cas12a_shuffling_model.teacher.scoring_utils import score_rows_with_teacher


@dataclass(frozen=True)
class RerankConfig:
    batch_size: int = 2048
    top_k: int = 50
    include_sequence: bool = False
    teacher_audit: bool = False
    teacher_audit_top_k: int = 200
    teacher_audit_batch_size: int = 8


def rerank_shortlist(
    *,
    shortlist_table: str,
    assistant_checkpoint: str,
    out_dir: str,
    cfg: RerankConfig,
    validated_domains: dict[tuple[str, int], str] | None = None,
    device: str | None = None,
    teacher_scorer: ProtGPT2Scorer | None = None,
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
    outputs = {
        "all_csv": str(all_path),
        "top_csv": str(top_path),
        "meta_json": str(meta_path),
    }
    if bool(cfg.teacher_audit) and teacher_scorer is not None:
        audit_n = max(1, min(int(cfg.teacher_audit_top_k), int(len(scored))))
        audit_df = scored.head(audit_n).copy().reset_index(drop=True)
        audit_teacher = score_rows_with_teacher(
            rows_df=audit_df,
            scorer=teacher_scorer,
            validated_domains=validated_domains,
            combo_col="combo_compact",
            seq_col="full_protein_sequence",
            batch_size=int(cfg.teacher_audit_batch_size),
        )
        audit_teacher = audit_teacher.rename(
            columns={
                "global_score": "teacher_global_score",
                "junction_mean": "teacher_junction_mean",
                "junction_min": "teacher_junction_min",
            }
        )
        audit_teacher = audit_teacher.sort_values("teacher_global_score", ascending=False).reset_index(drop=True)
        audit_teacher["teacher_rank"] = (audit_teacher.index + 1).astype(int)

        audit_out = scored.head(audit_n).copy()
        audit_out = audit_out.merge(
            audit_teacher[
                [
                    "slot_code_11",
                    "sequence_hash",
                    "teacher_global_score",
                    "teacher_junction_mean",
                    "teacher_junction_min",
                    "teacher_rank",
                    "teacher_cache_hit",
                ]
            ],
            on="slot_code_11",
            how="left",
        )
        corr_pearson = float(audit_out["assistant_score"].corr(audit_out["teacher_global_score"], method="pearson"))
        if (
            int(audit_out["assistant_score"].nunique(dropna=True)) > 1
            and int(audit_out["teacher_global_score"].nunique(dropna=True)) > 1
        ):
            corr_spearman = float(
                spearmanr(
                    audit_out["assistant_score"].to_numpy(),
                    audit_out["teacher_global_score"].to_numpy(),
                ).correlation
            )
        else:
            corr_spearman = float("nan")
        audit_meta = {
            "audit_top_k": int(audit_n),
            "assistant_teacher_corr_pearson": corr_pearson,
            "assistant_teacher_corr_spearman": corr_spearman,
            "teacher_model_fingerprint": getattr(teacher_scorer, "model_fingerprint", None),
        }
        audit_csv = out_path / "assistant_teacher_audit_top.csv"
        audit_json = out_path / "assistant_teacher_audit_summary.json"
        audit_out.to_csv(audit_csv, index=False)
        audit_json.write_text(json.dumps(audit_meta, indent=2), encoding="utf-8")
        outputs["teacher_audit_csv"] = str(audit_csv)
        outputs["teacher_audit_json"] = str(audit_json)
    return outputs
