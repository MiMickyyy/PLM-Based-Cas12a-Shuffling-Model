from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cas12a_shuffling_model.composition.active_prior import (
    active_ranking_summary,
    load_active_codes,
)
from cas12a_shuffling_model.composition.table_io import read_table
from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_named_tables(text: str) -> list[tuple[str, str]]:
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    out: list[tuple[str, str]] = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Invalid table spec: {it}; expected label=path")
        k, v = it.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def _pick_score_col(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in ("final_score", "assistant_score", "s_scan_score", "teacher_global_score"):
        if c in df.columns:
            return c
    raise KeyError("Cannot infer score column (final_score/assistant_score/s_scan_score/teacher_global_score)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--named-tables", required=True, help="label=path,label=path")
    ap.add_argument("--active-table", required=True)
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--top-ks", default="50,100")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    active_codes = load_active_codes(args.active_table)
    top_ks = tuple(int(x.strip()) for x in str(args.top_ks).split(",") if x.strip())
    rows = []
    summary = {"active_table": args.active_table, "top_ks": top_ks, "results": {}}

    for label, path in _parse_named_tables(args.named_tables):
        df = read_table(path)
        score_col = _pick_score_col(df, args.score_col)
        s = active_ranking_summary(
            df=df,
            score_col=score_col,
            active_codes=active_codes,
            top_ks=top_ks,
            distance_ks=top_ks,
        )
        summary["results"][label] = {"table": path, "score_col": score_col, **s}
        flat = {"label": label, "table": path, "score_col": score_col}
        for k, v in s.items():
            if isinstance(v, (int, float)) or v is None:
                flat[k] = v
        rows.append(flat)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    logger.info("Active rerank report saved: %s", out_json)


if __name__ == "__main__":
    main()
