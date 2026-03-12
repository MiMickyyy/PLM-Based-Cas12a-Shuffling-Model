from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cas12a_shuffling_model.io.cas12a_corpus import read_cas12a_fasta, sample_sequences
from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
)
from cas12a_shuffling_model.search.sampler import sample_combo_compacts
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    score_rows_with_teacher,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherAdaptEvalConfig:
    seed: int = 13
    natural_eval_size: int = 128
    background_size: int = 128
    score_batch_size: int = 1
    make_plots: bool = True


def _distribution_stats(series: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
    }


def _score_natural_sequences(
    *,
    scorer,
    sequences: list[str],
    batch_size: int,
) -> pd.DataFrame:
    scores = scorer.score_many(
        seqs_aa=sequences,
        domain_lengths_list=[None] * len(sequences),
        batch_size=batch_size,
    )
    rows = []
    for seq, sc in zip(sequences, scores):
        nll = float(-sc.global_score)
        ppl = float(math.exp(min(20.0, nll)))
        rows.append(
            {
                "sequence_aa": seq,
                "sequence_hash": sc.seq_hash,
                "global_score": sc.global_score,
                "nll": nll,
                "perplexity": ppl,
            }
        )
    return pd.DataFrame(rows)


def run_teacher_adaptation_eval(
    *,
    root_config: dict,
    adapted_model_path: str,
    validated_domains: dict[tuple[str, int], str],
    out_dir: str,
    cfg: TeacherAdaptEvalConfig,
    active_csv: str,
    natural_fasta: str,
    device: str | None,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # baseline
    baseline_cfg = json.loads(json.dumps(root_config))
    baseline_cfg.setdefault("teacher", {})
    baseline_cfg["teacher"]["adapter_path"] = None
    if baseline_cfg["teacher"].get("model_name_or_path") is None:
        baseline_cfg["teacher"]["model_name_or_path"] = baseline_cfg["teacher"].get(
            "model_name", "nferruz/ProtGPT2"
        )
    baseline_scorer = build_teacher_scorer_from_config(baseline_cfg, device=device)

    # adapted
    adapted_cfg = json.loads(json.dumps(root_config))
    adapted_cfg.setdefault("teacher", {})
    adapted_cfg["teacher"]["model_source"] = "local"
    adapted_cfg["teacher"]["model_name_or_path"] = str(adapted_model_path)
    adapted_cfg["teacher"]["adapter_path"] = None
    adapted_scorer = build_teacher_scorer_from_config(adapted_cfg, device=device)

    natural_records = read_cas12a_fasta(
        natural_fasta,
        min_len=int(root_config.get("teacher_adapt", {}).get("min_len", 300)),
        max_len=root_config.get("teacher_adapt", {}).get("max_len_filter"),
        deduplicate=True,
    )
    natural_picked = sample_sequences(
        natural_records,
        n=int(cfg.natural_eval_size),
        seed=int(cfg.seed) + 101,
    )
    natural_seqs = [r.sequence_aa for r in natural_picked]
    nat_base_df = _score_natural_sequences(
        scorer=baseline_scorer, sequences=natural_seqs, batch_size=cfg.score_batch_size
    )
    nat_adapt_df = _score_natural_sequences(
        scorer=adapted_scorer, sequences=natural_seqs, batch_size=cfg.score_batch_size
    )
    nat_cmp = nat_base_df.rename(
        columns={
            "global_score": "global_score_baseline",
            "nll": "nll_baseline",
            "perplexity": "ppl_baseline",
        }
    ).merge(
        nat_adapt_df.rename(
            columns={
                "global_score": "global_score_adapted",
                "nll": "nll_adapted",
                "perplexity": "ppl_adapted",
            }
        ),
        on=["sequence_aa", "sequence_hash"],
        how="inner",
    )
    nat_cmp.to_csv(out_path / "natural_eval_scores.csv", index=False)

    active_df_raw = pd.read_csv(active_csv)
    active_base = score_rows_with_teacher(
        rows_df=active_df_raw,
        scorer=baseline_scorer,
        validated_domains=validated_domains,
        combo_col="combo_compact",
        seq_col="sequence_aa",
        batch_size=cfg.score_batch_size,
    )
    active_adapt = score_rows_with_teacher(
        rows_df=active_df_raw,
        scorer=adapted_scorer,
        validated_domains=validated_domains,
        combo_col="combo_compact",
        seq_col="sequence_aa",
        batch_size=cfg.score_batch_size,
    )

    bg_combos = sample_combo_compacts(n=int(cfg.background_size), seed=int(cfg.seed) + 313)
    bg_rows = []
    for combo in bg_combos:
        bg_rows.append(
            {
                "combo_compact": combo,
                "sequence_aa": build_sequence_from_combo(combo, validated_domains),
                "domain_lengths": domain_lengths_from_combo(combo, validated_domains),
            }
        )
    bg_df_raw = pd.DataFrame(bg_rows)
    bg_base = score_rows_with_teacher(
        rows_df=bg_df_raw,
        scorer=baseline_scorer,
        validated_domains=validated_domains,
        combo_col="combo_compact",
        seq_col="sequence_aa",
        batch_size=cfg.score_batch_size,
    )
    bg_adapt = score_rows_with_teacher(
        rows_df=bg_df_raw,
        scorer=adapted_scorer,
        validated_domains=validated_domains,
        combo_col="combo_compact",
        seq_col="sequence_aa",
        batch_size=cfg.score_batch_size,
    )

    active_base.to_csv(out_path / "active_scores_baseline.csv", index=False)
    active_adapt.to_csv(out_path / "active_scores_adapted.csv", index=False)
    bg_base.to_csv(out_path / "background_scores_baseline.csv", index=False)
    bg_adapt.to_csv(out_path / "background_scores_adapted.csv", index=False)

    report = {
        "config": asdict(cfg),
        "adapted_model_path": str(adapted_model_path),
        "natural_eval": {
            "n": len(nat_cmp),
            "baseline_nll": _distribution_stats(nat_cmp["nll_baseline"]),
            "adapted_nll": _distribution_stats(nat_cmp["nll_adapted"]),
            "baseline_ppl": _distribution_stats(nat_cmp["ppl_baseline"]),
            "adapted_ppl": _distribution_stats(nat_cmp["ppl_adapted"]),
        },
        "active_vs_background": {
            "baseline": {
                "active_global": _distribution_stats(active_base["global_score"]),
                "background_global": _distribution_stats(bg_base["global_score"]),
                "active_junction_mean": _distribution_stats(active_base["junction_mean"]),
                "background_junction_mean": _distribution_stats(bg_base["junction_mean"]),
                "active_junction_min": _distribution_stats(active_base["junction_min"]),
                "background_junction_min": _distribution_stats(bg_base["junction_min"]),
            },
            "adapted": {
                "active_global": _distribution_stats(active_adapt["global_score"]),
                "background_global": _distribution_stats(bg_adapt["global_score"]),
                "active_junction_mean": _distribution_stats(active_adapt["junction_mean"]),
                "background_junction_mean": _distribution_stats(bg_adapt["junction_mean"]),
                "active_junction_min": _distribution_stats(active_adapt["junction_min"]),
                "background_junction_min": _distribution_stats(bg_adapt["junction_min"]),
            },
        },
    }

    # Simple effect size style deltas.
    report["effect_sizes"] = {
        "baseline_global_active_minus_bg": float(
            report["active_vs_background"]["baseline"]["active_global"]["mean"]
            - report["active_vs_background"]["baseline"]["background_global"]["mean"]
        ),
        "adapted_global_active_minus_bg": float(
            report["active_vs_background"]["adapted"]["active_global"]["mean"]
            - report["active_vs_background"]["adapted"]["background_global"]["mean"]
        ),
        "baseline_jmin_active_minus_bg": float(
            report["active_vs_background"]["baseline"]["active_junction_min"]["mean"]
            - report["active_vs_background"]["baseline"]["background_junction_min"]["mean"]
        ),
        "adapted_jmin_active_minus_bg": float(
            report["active_vs_background"]["adapted"]["active_junction_min"]["mean"]
            - report["active_vs_background"]["adapted"]["background_junction_min"]["mean"]
        ),
    }

    if cfg.make_plots:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            logger.warning("matplotlib unavailable; skip adaptation eval plots")
        else:
            plt.figure(figsize=(7, 5))
            plt.scatter(
                nat_cmp["global_score_baseline"].astype(float),
                nat_cmp["global_score_adapted"].astype(float),
                s=10,
                alpha=0.7,
            )
            plt.xlabel("Baseline global_score")
            plt.ylabel("Adapted global_score")
            plt.title("Natural Cas12a: baseline vs adapted global_score")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(out_path / "scatter_baseline_vs_adapted_global.png", dpi=200)
            plt.close()

            plt.figure(figsize=(7, 5))
            plt.hist(
                nat_cmp["nll_baseline"].astype(float),
                bins=30,
                alpha=0.6,
                label="baseline",
            )
            plt.hist(
                nat_cmp["nll_adapted"].astype(float),
                bins=30,
                alpha=0.6,
                label="adapted",
            )
            plt.xlabel("NLL")
            plt.ylabel("Count")
            plt.title("Natural Cas12a NLL distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path / "natural_nll_overlay.png", dpi=200)
            plt.close()

    (out_path / "adaptation_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
