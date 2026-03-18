from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cas12a_shuffling_model.composition.active_prior import (
    active_prior_beta_sweep,
    active_ranking_summary,
    ensure_active_codes,
)
from cas12a_shuffling_model.composition.assistant_ranker import AssistantRankerScorer
from cas12a_shuffling_model.composition.chimera_repr import canonicalize_chimera_table
from cas12a_shuffling_model.composition.gated_policy import GatedPolicyConfig, apply_gated_policy
from cas12a_shuffling_model.composition.two_stage_policy import TwoStagePolicyConfig, apply_two_stage_policy
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
    policy_mode: str = "global_fixed"
    gate_signal: str = "kernel_similarity_to_actives"
    alpha_far: float = 0.60
    alpha_near: float = 0.15
    similarity_beta: float = 0.15
    kernel_gamma: float = 0.70
    density_radius: int = 3
    density_gamma: float = 1.0
    hard_distance_threshold: int = 3
    hard_similarity_threshold: float = 0.55
    soft_center: float | None = None
    soft_scale: float = 8.0
    recall_policy: str = "teacher_recall"
    recall_teacher_weight: float = 0.70
    recall_active_weight: float = 0.10
    recall_scan_weight: float = 0.20
    recall_pool_size: int | None = 100000
    teacher_usage_mode: str = "none"
    teacher_plausibility_quantile: float = 0.05
    teacher_plausibility_penalty: float = 0.50
    active_prior_mode: str = "none"
    active_prior_beta: float = 0.0
    active_prior_gamma: float = 0.7
    active_prior_slot_weights: tuple[float, ...] | None = None
    active_prior_beta_sweep: tuple[float, ...] = ()
    active_prior_base_score_col: str = "assistant_score"
    active_eval_top_ks: tuple[int, ...] = (50, 100)
    recall_eval_top_ks: tuple[int, ...] = (20000, 50000, 100000)
    use_two_stage_policy: bool = True
    final_rerank_policy: str = "active_only"
    recall_diversity_weight: float = 0.05
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
    active_codes_norm = ensure_active_codes(active_codes or [])
    if bool(cfg.use_two_stage_policy):
        sim_mode = str(cfg.active_prior_mode).strip().lower()
        if sim_mode in {"", "none"}:
            sim_mode = "kernel_density_over_actives"
        scored = apply_two_stage_policy(
            scored,
            active_codes=active_codes_norm,
            cfg=TwoStagePolicyConfig(
                recall_policy=str(cfg.recall_policy),
                final_rerank_policy=str(cfg.final_rerank_policy),
                recall_teacher_weight=float(cfg.recall_teacher_weight),
                recall_scan_weight=float(cfg.recall_scan_weight),
                recall_active_weight=float(cfg.recall_active_weight),
                recall_diversity_weight=float(cfg.recall_diversity_weight),
                recall_pool_size=(int(cfg.recall_pool_size) if cfg.recall_pool_size is not None else None),
                active_similarity_mode=sim_mode,
                active_similarity_beta=float(cfg.similarity_beta),
                active_similarity_gamma=float(cfg.active_prior_gamma),
                active_similarity_slot_weights=cfg.active_prior_slot_weights,
                teacher_plausibility_quantile=float(cfg.teacher_plausibility_quantile),
                teacher_plausibility_penalty=float(cfg.teacher_plausibility_penalty),
            ),
        )
    else:
        scored = apply_gated_policy(
            scored,
            active_codes=active_codes_norm,
            cfg=GatedPolicyConfig(
                policy_mode=str(cfg.policy_mode),
                gate_signal=str(cfg.gate_signal),
                alpha_far=float(cfg.alpha_far),
                alpha_near=float(cfg.alpha_near),
                similarity_beta=float(cfg.similarity_beta),
                kernel_gamma=float(cfg.kernel_gamma),
                density_radius=int(cfg.density_radius),
                density_gamma=float(cfg.density_gamma),
                hard_distance_threshold=int(cfg.hard_distance_threshold),
                hard_similarity_threshold=float(cfg.hard_similarity_threshold),
                soft_center=(float(cfg.soft_center) if cfg.soft_center is not None else None),
                soft_scale=float(cfg.soft_scale),
                recall_policy=str(cfg.recall_policy),
                recall_teacher_weight=float(cfg.recall_teacher_weight),
                recall_active_weight=float(cfg.recall_active_weight),
                recall_scan_weight=float(cfg.recall_scan_weight),
                recall_pool_size=(int(cfg.recall_pool_size) if cfg.recall_pool_size is not None else None),
                teacher_usage_mode=str(cfg.teacher_usage_mode),
                teacher_plausibility_quantile=float(cfg.teacher_plausibility_quantile),
                teacher_plausibility_penalty=float(cfg.teacher_plausibility_penalty),
                slot_weights=cfg.active_prior_slot_weights,
            ),
        )

    if bool(cfg.include_sequence) and validated_domains is not None:
        scored["full_protein_sequence"] = scored["slot_code_11"].map(
            lambda code: build_sequence_from_combo(str(code), validated_domains)
        )

    recall_sorted = scored.sort_values("recall_stage_score", ascending=False).reset_index(drop=True)
    recall_sorted["recall_stage_rank"] = (recall_sorted.index + 1).astype(int)
    pool_size = int(cfg.recall_pool_size) if cfg.recall_pool_size is not None else int(len(recall_sorted))
    pool_size = max(1, min(pool_size, int(len(recall_sorted))))
    rerank_pool = recall_sorted.head(pool_size).copy().reset_index(drop=True)
    rerank_pool = rerank_pool.sort_values("final_gated_score", ascending=False).reset_index(drop=True)
    rerank_pool["assistant_rank"] = (rerank_pool.index + 1).astype(int)
    rerank_pool["rerank_stage_rank"] = rerank_pool["assistant_rank"]
    rerank_pool["final_score"] = pd.to_numeric(rerank_pool["final_gated_score"], errors="coerce")
    scored = rerank_pool
    recall_all_path = out_path / "recall_ranked_all.csv"
    recall_sorted.to_csv(recall_all_path, index=False)
    top = scored.head(int(cfg.top_k)).copy().reset_index(drop=True)
    top["final_rank"] = (top.index + 1).astype(int)

    all_path = out_path / "assistant_reranked_all.csv"
    top_path = out_path / "assistant_reranked_top.csv"
    scored.to_csv(all_path, index=False)
    top.to_csv(top_path, index=False)

    meta = {
        "shortlist_table": shortlist_table,
        "assistant_checkpoint": assistant_checkpoint,
        "n_input_shortlist": int(len(canonical)),
        "n_shortlist": int(len(scored)),
        "n_recall_pool": int(pool_size),
        "top_k": int(cfg.top_k),
        "final_score_col": "final_gated_score",
        "use_two_stage_policy": bool(cfg.use_two_stage_policy),
        "recall_policy": cfg.recall_policy,
        "final_rerank_policy": cfg.final_rerank_policy,
        "recall_diversity_weight": float(cfg.recall_diversity_weight),
        "policy_mode": cfg.policy_mode,
        "gate_signal": cfg.gate_signal,
        "alpha_far": float(cfg.alpha_far),
        "alpha_near": float(cfg.alpha_near),
        "similarity_beta": float(cfg.similarity_beta),
        "teacher_usage_mode": cfg.teacher_usage_mode,
        "teacher_plausibility_quantile": float(cfg.teacher_plausibility_quantile),
        "teacher_plausibility_penalty": float(cfg.teacher_plausibility_penalty),
        "recall_pool_size": cfg.recall_pool_size,
        "dual_head_alpha": cfg.dual_head_alpha,
    }
    meta_path = out_path / "assistant_rerank_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    outputs = {
        "all_csv": str(all_path),
        "recall_csv": str(recall_all_path),
        "top_csv": str(top_path),
        "meta_json": str(meta_path),
    }
    if len(active_codes_norm) > 0:
        recall_summary = active_ranking_summary(
            df=recall_sorted,
            score_col="recall_stage_score",
            active_codes=active_codes_norm,
            top_ks=cfg.recall_eval_top_ks,
            distance_ks=cfg.recall_eval_top_ks,
        )
        rerank_summary = active_ranking_summary(
            df=scored,
            score_col="final_gated_score",
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
        active_summary = {f"recall_{k}": v for k, v in recall_summary.items()}
        active_summary.update({f"rerank_{k}": v for k, v in rerank_summary.items()})
        active_summary.update(suppression)
        active_summary["active_codes_total"] = int(len(active_codes_norm))
        active_summary["active_missing_from_recall_pool"] = int(
            len(set(active_codes_norm) - set(scored["slot_code_11"].astype(str).tolist()))
        )
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
        corr_pearson = float(audit_out["final_gated_score"].corr(audit_out["teacher_global_score"], method="pearson"))
        if (
            int(audit_out["final_gated_score"].nunique(dropna=True)) > 1
            and int(audit_out["teacher_global_score"].nunique(dropna=True)) > 1
        ):
            corr_spearman = float(
                spearmanr(
                    audit_out["final_gated_score"].to_numpy(),
                    audit_out["teacher_global_score"].to_numpy(),
                ).correlation
            )
        else:
            corr_spearman = float("nan")
        teacher_head_corr = float("nan")
        active_head_corr = float("nan")
        if "assistant_teacher_head_score" in audit_out.columns:
            teacher_head_corr = float(
                audit_out["assistant_teacher_head_score"].corr(audit_out["teacher_global_score"], method="pearson")
            )
        if "assistant_active_head_score" in audit_out.columns:
            active_head_corr = float(
                audit_out["assistant_active_head_score"].corr(audit_out["teacher_global_score"], method="pearson")
            )
        audit_meta = {
            "audit_top_k": int(audit_n),
            "final_teacher_corr_pearson": corr_pearson,
            "final_teacher_corr_spearman": corr_spearman,
            "teacher_head_teacher_corr_pearson": teacher_head_corr,
            "active_head_teacher_corr_pearson": active_head_corr,
            "teacher_model_fingerprint": getattr(teacher_scorer, "model_fingerprint", None),
        }
        audit_csv = out_path / "assistant_teacher_audit_top.csv"
        audit_json = out_path / "assistant_teacher_audit_summary.json"
        audit_out.to_csv(audit_csv, index=False)
        audit_json.write_text(json.dumps(audit_meta, indent=2), encoding="utf-8")
        outputs["teacher_audit_csv"] = str(audit_csv)
        outputs["teacher_audit_json"] = str(audit_json)
    return outputs
