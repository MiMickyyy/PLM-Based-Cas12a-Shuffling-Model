from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cas12a_shuffling_model.composition.active_prior import (
    ActivePriorConfig,
    active_prior_beta_sweep,
    active_ranking_summary,
    apply_active_prior,
    ensure_active_codes,
)
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
    dual_head_alpha: float | None = None
    active_prior_mode: str = "none"
    active_prior_beta: float = 0.0
    active_prior_gamma: float = 0.7
    active_prior_slot_weights: tuple[float, ...] | None = None
    active_prior_beta_sweep: tuple[float, ...] = ()
    active_prior_base_score_col: str = "assistant_score"
    active_eval_top_ks: tuple[int, ...] = (50, 100)
    teacher_audit: bool = False
    teacher_audit_top_k: int = 200
    teacher_audit_batch_size: int = 8


def _scan_vs_rerank_active_suppression(
    *,
    shortlist: pd.DataFrame,
    reranked: pd.DataFrame,
    active_codes: Sequence[str],
    top_k: int,
) -> dict[str, object]:
    actives = set(ensure_active_codes(active_codes))
    if len(actives) == 0:
        return {}
    out: dict[str, object] = {}
    if "s_scan_score" in shortlist.columns:
        scan_top = shortlist.sort_values("s_scan_score", ascending=False).head(int(top_k))
        scan_top_active = sorted(set(scan_top["slot_code_11"].astype(str).tolist()) & actives)
    else:
        scan_top_active = []
    rerank_top = reranked.sort_values("final_score", ascending=False).head(int(top_k))
    rerank_top_active = sorted(set(rerank_top["slot_code_11"].astype(str).tolist()) & actives)
    suppressed = sorted(set(scan_top_active) - set(rerank_top_active))
    out["s_scan_top_active_codes"] = scan_top_active
    out["rerank_top_active_codes"] = rerank_top_active
    out["suppressed_active_codes"] = suppressed
    out["n_s_scan_top_active"] = int(len(scan_top_active))
    out["n_rerank_top_active"] = int(len(rerank_top_active))
    out["n_suppressed"] = int(len(suppressed))
    if len(suppressed) > 0:
        cols = ["slot_code_11", "assistant_score", "final_score"]
        for c in ("assistant_teacher_head_score", "assistant_active_head_score", "active_similarity"):
            if c in reranked.columns:
                cols.append(c)
        sup_df = reranked[reranked["slot_code_11"].astype(str).isin(suppressed)][cols].copy()
        out["suppressed_active_rows"] = sup_df.to_dict(orient="records")
    return out


def rerank_shortlist(
    *,
    shortlist_table: str,
    assistant_checkpoint: str,
    out_dir: str,
    cfg: RerankConfig,
    active_codes: Sequence[str] | None = None,
    validated_domains: dict[tuple[str, int], str] | None = None,
    device: str | None = None,
    teacher_scorer: ProtGPT2Scorer | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shortlist = read_table(shortlist_table)
    canonical = canonicalize_chimera_table(shortlist, validated_domains=None, require_sequence=False)
    scorer = AssistantRankerScorer(assistant_checkpoint, device=device)
    scored = scorer.score_dataframe(
        canonical,
        batch_size=int(cfg.batch_size),
        dual_head_alpha=cfg.dual_head_alpha,
    )

    final_score_col = "assistant_score"
    active_codes_norm = ensure_active_codes(active_codes or [])
    if len(active_codes_norm) > 0:
        scored = apply_active_prior(
            scored,
            score_col=str(cfg.active_prior_base_score_col or "assistant_score"),
            active_codes=active_codes_norm,
            cfg=ActivePriorConfig(
                mode=str(cfg.active_prior_mode),
                beta=float(cfg.active_prior_beta),
                gamma=float(cfg.active_prior_gamma),
                slot_weights=cfg.active_prior_slot_weights,
            ),
        )
        final_score_col = "final_score"
    else:
        scored["active_similarity"] = 0.0
        scored["min_hamming_to_active"] = np.nan
        scored["final_score"] = pd.to_numeric(scored["assistant_score"], errors="coerce")
        final_score_col = "final_score"

    if bool(cfg.include_sequence) and validated_domains is not None:
        scored["full_protein_sequence"] = scored["slot_code_11"].map(
            lambda code: build_sequence_from_combo(str(code), validated_domains)
        )

    scored = scored.sort_values(final_score_col, ascending=False).reset_index(drop=True)
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
        "final_score_col": final_score_col,
        "active_prior_mode": cfg.active_prior_mode,
        "active_prior_beta": float(cfg.active_prior_beta),
        "dual_head_alpha": cfg.dual_head_alpha,
    }
    meta_path = out_path / "assistant_rerank_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs = {
        "all_csv": str(all_path),
        "top_csv": str(top_path),
        "meta_json": str(meta_path),
    }
    if len(active_codes_norm) > 0:
        active_summary = active_ranking_summary(
            df=scored,
            score_col=final_score_col,
            active_codes=active_codes_norm,
            top_ks=cfg.active_eval_top_ks,
            distance_ks=cfg.active_eval_top_ks,
        )
        suppression = _scan_vs_rerank_active_suppression(
            shortlist=canonical,
            reranked=scored,
            active_codes=active_codes_norm,
            top_k=int(cfg.top_k),
        )
        active_summary.update(suppression)
        active_summary_path = out_path / "assistant_active_ranking_summary.json"
        active_summary_path.write_text(json.dumps(active_summary, indent=2), encoding="utf-8")
        outputs["active_summary_json"] = str(active_summary_path)

        if len(cfg.active_prior_beta_sweep) > 0 and str(cfg.active_prior_mode).lower() != "none":
            sweep_df = active_prior_beta_sweep(
                df=scored,
                base_score_col=str(cfg.active_prior_base_score_col or "assistant_score"),
                active_codes=active_codes_norm,
                mode=str(cfg.active_prior_mode),
                betas=cfg.active_prior_beta_sweep,
                gamma=float(cfg.active_prior_gamma),
                slot_weights=cfg.active_prior_slot_weights,
                top_ks=cfg.active_eval_top_ks,
            )
            sweep_path = out_path / "active_prior_beta_sweep.csv"
            sweep_df.to_csv(sweep_path, index=False)
            outputs["active_prior_beta_sweep_csv"] = str(sweep_path)

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
