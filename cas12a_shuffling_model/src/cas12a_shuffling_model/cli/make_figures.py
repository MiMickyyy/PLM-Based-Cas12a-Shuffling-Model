from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


PARENTS = ["A", "L", "F", "M"]


def _rank_score_curve(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("final_rank").reset_index(drop=True)
    x = d["final_rank"].astype(int).to_numpy()
    y = d["calibrated_prob"].astype(float).to_numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, lw=2)
    plt.xlabel("Rank")
    plt.ylabel("Calibrated probability")
    plt.title("Rank-Score Curve")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _slot_parent_heatmap(df: pd.DataFrame, out_path: Path, top_n: int) -> None:
    d = df.sort_values("final_rank").head(top_n).copy()
    mat = np.zeros((11, 4), dtype=float)
    for combo in d["combo_compact"].astype(str):
        for i, ch in enumerate(combo):
            if ch in PARENTS:
                mat[i, PARENTS.index(ch)] += 1
    if len(d) > 0:
        mat = mat / float(len(d))

    plt.figure(figsize=(7, 5))
    im = plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Frequency")
    plt.xticks(range(len(PARENTS)), PARENTS)
    plt.yticks(range(11), [f"S{i}" for i in range(1, 12)])
    plt.xlabel("Parent letter")
    plt.ylabel("Slot")
    plt.title(f"Slot × Parent Frequency (Top {top_n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _topk_overlap_curve(df1: pd.DataFrame, df2: pd.DataFrame, out_path: Path, max_k: int) -> None:
    a = df1.sort_values("final_rank")["combo_compact"].astype(str).tolist()
    b = df2.sort_values("final_rank")["combo_compact"].astype(str).tolist()
    max_k = min(max_k, len(a), len(b))
    ks = []
    overlaps = []
    set_a = set()
    set_b = set()
    for k in range(1, max_k + 1):
        set_a.add(a[k - 1])
        set_b.add(b[k - 1])
        inter = len(set_a.intersection(set_b))
        ks.append(k)
        overlaps.append(inter / float(k))

    plt.figure(figsize=(8, 5))
    plt.plot(ks, overlaps, lw=2)
    plt.xlabel("Top-K")
    plt.ylabel("Overlap fraction")
    plt.title("Top-K Overlap Stability Curve")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(df[x_col].astype(float), df[y_col].astype(float), s=12, alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking-csv", required=True, help="Primary candidate_top.csv")
    ap.add_argument("--ranking-csv-2", default=None, help="Secondary candidate_top.csv for overlap curve")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n-heatmap", type=int, default=50)
    ap.add_argument("--max-k-overlap", type=int, default=50)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ranking_csv)
    _rank_score_curve(df, out_dir / "rank_score_curve.png")
    _slot_parent_heatmap(df, out_dir / "slot_parent_heatmap.png", top_n=int(args.top_n_heatmap))
    _scatter(
        df,
        x_col="global_score",
        y_col="calibrated_prob",
        title="calibrated_prob vs global_score",
        out_path=out_dir / "scatter_prob_vs_global.png",
    )
    _scatter(
        df,
        x_col="junction_min",
        y_col="calibrated_prob",
        title="calibrated_prob vs junction_min",
        out_path=out_dir / "scatter_prob_vs_junction_min.png",
    )

    if args.ranking_csv_2:
        df2 = pd.read_csv(args.ranking_csv_2)
        _topk_overlap_curve(
            df,
            df2,
            out_dir / "topk_overlap_stability.png",
            max_k=int(args.max_k_overlap),
        )
    else:
        logger.warning("No --ranking-csv-2 provided; skip top-k overlap stability figure")

    logger.info("Figures written to %s", out_dir)


if __name__ == "__main__":
    main()

