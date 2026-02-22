from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.domain.chimera_builder import extract_active_rows
from cas12a_shuffling_model.io.loaders import load_yaml, read_sequence_results_table
from cas12a_shuffling_model.search.combo_compact import (
    build_sequence_from_combo,
    domain_lengths_from_combo,
    validate_combo_compact,
)
from cas12a_shuffling_model.search.sampler import sample_combo_compacts
from cas12a_shuffling_model.teacher.scoring_utils import (
    build_teacher_scorer_from_config,
    load_validated_domains_dict,
    resolve_validated_domains_path,
)
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _actives_combo_list(config: dict) -> list[str]:
    act_cfg = config.get("actives", {})
    seq_path = config.get("paths", {}).get("sequence_results")
    if not seq_path:
        return []
    df = read_sequence_results_table(seq_path)
    active_df, slot_cols = extract_active_rows(
        df,
        slot_columns=act_cfg.get("slot_columns"),
        allowed_letters=act_cfg.get("allowed_letters", ["A", "L", "F", "M"]),
    )
    combos = []
    for _, row in active_df.iterrows():
        combo = validate_combo_compact("".join(str(row[c]).strip().upper() for c in slot_cols))
        combos.append(combo)
    return combos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cas12a_shuffling_model/configs/default.yaml")
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--include-actives", action="store_true")
    ap.add_argument("--device", default=None, help="cpu/cuda/mps; default auto")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    config = load_yaml(args.config)
    dist_cfg = config.get("distill", {})

    seed = int(args.seed) if args.seed is not None else int(config.get("seed", 13))
    n_samples = int(args.n_samples) if args.n_samples is not None else int(dist_cfg.get("n_samples", 64))

    out_csv = args.out_csv
    if not out_csv:
        out_csv = dist_cfg.get("output_csv")
    if not out_csv:
        out_processed = config.get("paths", {}).get("out_processed_dir")
        if not out_processed:
            raise ValueError("Missing output path: set --out-csv or distill.output_csv")
        out_csv = str(Path(out_processed) / "distill_teacher_scores.csv")

    vd_path = resolve_validated_domains_path(config, cli_path=args.validated_domains)
    domains = load_validated_domains_dict(vd_path)
    scorer = build_teacher_scorer_from_config(config, device=args.device)

    combos = sample_combo_compacts(n=n_samples, seed=seed)
    if args.include_actives or bool(dist_cfg.get("include_actives", False)):
        combos.extend(_actives_combo_list(config))
    combos = sorted(set(combos))

    run_id = f"distill_{int(time.time())}_{seed}"
    rows = []
    for combo in combos:
        seq = build_sequence_from_combo(combo, domains)
        lengths = domain_lengths_from_combo(combo, domains)
        score = scorer.score_one(seq_aa=seq, domain_lengths=lengths)
        rec = {
            "source_run_id": run_id,
            "seed": seed,
            "combo_compact": combo,
            "sequence_aa": seq,
            "sequence_hash": score.seq_hash,
            "global_score": score.global_score,
            "junction_mean": score.junction_mean,
            "junction_min": score.junction_min,
            "teacher_cache_hit": bool(score.from_cache),
        }
        for i, v in enumerate(score.junction_scores, start=1):
            rec[f"junction_{i:02d}"] = v
        rows.append(rec)

    out_df = pd.DataFrame(rows).sort_values(["combo_compact"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info(
        "Wrote distill set rows=%d, cache_hits=%d, output=%s",
        len(out_df),
        int(out_df["teacher_cache_hit"].sum()),
        out_csv,
    )


if __name__ == "__main__":
    main()

