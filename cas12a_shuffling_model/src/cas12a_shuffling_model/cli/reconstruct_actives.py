from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cas12a_shuffling_model.domain.chimera_builder import (
    extract_active_rows,
    load_validated_domains_table,
    reconstruct_chimeras,
    validated_domains_to_dict,
)
from cas12a_shuffling_model.io.loaders import ensure_dir, load_yaml, read_sequence_results_table
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (optional)")
    ap.add_argument("--sequence-results", default=None)
    ap.add_argument("--validated-domains", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--slot-columns", nargs="*", default=None)
    args = ap.parse_args()

    setup_logging(args.log_level)

    cfg = load_yaml(args.config) if args.config else {}
    sequence_results = args.sequence_results or cfg.get("paths", {}).get("sequence_results")
    validated_domains = args.validated_domains or cfg.get("paths", {}).get("out_processed_dir", "")
    out_dir = args.out_dir or cfg.get("paths", {}).get("out_active_dir")
    if validated_domains and Path(validated_domains).is_dir():
        validated_domains = str(Path(validated_domains) / "validated_domain_peptides.csv")

    if not sequence_results or not validated_domains or not out_dir:
        raise SystemExit(
            "Missing required inputs. Provide --config or pass --sequence-results, --validated-domains, and --out-dir."
        )

    df_in = read_sequence_results_table(sequence_results)
    active, slot_cols = extract_active_rows(
        df_in,
        slot_columns=args.slot_columns,
        allowed_letters=["A", "L", "F", "M"],
    )
    logger.info("Detected %d active rows using slot columns: %s", len(active), slot_cols)

    domains_df = load_validated_domains_table(validated_domains)
    domain_dict = validated_domains_to_dict(domains_df)

    out_df = reconstruct_chimeras(active, slot_cols, domain_dict)

    out_dir = ensure_dir(out_dir)
    out_path = out_dir / "active_chimeras_reconstructed.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()
