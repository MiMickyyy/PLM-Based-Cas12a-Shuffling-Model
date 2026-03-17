from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(p)
    if ext in {".tsv"}:
        return pd.read_csv(p, sep="\t")
    raise ValueError(f"Unsupported table format: {p}")


def write_table(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext == ".parquet":
        try:
            df.to_parquet(p, index=False)
            return str(p)
        except Exception as e:
            fallback = p.with_suffix(".csv")
            logger.warning(
                "Failed to write parquet (%s), fallback to csv: %s",
                e,
                fallback,
            )
            df.to_csv(fallback, index=False)
            return str(fallback)

    if ext in {".csv", ".txt"}:
        df.to_csv(p, index=False)
        return str(p)
    if ext in {".tsv"}:
        df.to_csv(p, sep="\t", index=False)
        return str(p)
    raise ValueError(f"Unsupported table format for write: {p}")

