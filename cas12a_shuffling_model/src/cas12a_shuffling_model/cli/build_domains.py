from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cas12a_shuffling_model.domain.translate_domains import build_validated_domain_table
from cas12a_shuffling_model.io.loaders import load_yaml, read_snapgene_protein_sequence
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_parents(kv_list: list[str]) -> dict[str, str]:
    parents = {}
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Expected KEY=PATH, got: {item}")
        k, v = item.split("=", 1)
        parents[k.strip()] = v.strip()
    return parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (optional)")
    ap.add_argument("--domains-dir", default=None)
    ap.add_argument(
        "--parents",
        nargs="+",
        default=None,
        help="Example: As=../AsCas12a.prot Fn=../FnCas12a.prot Lb=../LbCas12a.prot Mb2=../Mb2Cas12a.prot",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--no-revcomp", action="store_true")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--ambiguous-delta", type=float, default=None)
    ap.add_argument("--no-enforce-order", action="store_true")
    args = ap.parse_args()

    setup_logging(args.log_level)

    cfg = load_yaml(args.config) if args.config else {}
    domains_dir = args.domains_dir or cfg.get("paths", {}).get("domains_dir")
    out_dir = args.out_dir or cfg.get("paths", {}).get("out_processed_dir")

    if args.parents is not None:
        parent_paths = _parse_parents(args.parents)
    else:
        parent_paths = cfg.get("paths", {}).get("parents", {})

    if not domains_dir or not out_dir or not parent_paths:
        raise SystemExit(
            "Missing required inputs. Provide --config or pass --domains-dir, --out-dir, and --parents."
        )

    parent_proteins = {k: read_snapgene_protein_sequence(v) for k, v in parent_paths.items()}

    dt_cfg = cfg.get("domain_translation", {})
    df = build_validated_domain_table(
        domains_dir=domains_dir,
        parent_proteins=parent_proteins,
        out_dir=out_dir,
        try_reverse_complement=False
        if args.no_revcomp
        else bool(dt_cfg.get("try_reverse_complement", True)),
        top_k_per_domain=int(args.top_k)
        if args.top_k is not None
        else int(dt_cfg.get("top_k_per_domain", 5)),
        ambiguous_delta_score=float(args.ambiguous_delta)
        if args.ambiguous_delta is not None
        else float(dt_cfg.get("ambiguous_delta_score", 25.0)),
        enforce_parent_slot_order=False
        if args.no_enforce_order
        else bool(dt_cfg.get("enforce_parent_slot_order", True)),
    )
    logger.info("Wrote validated domains: %s rows", len(df))


if __name__ == "__main__":
    main()
