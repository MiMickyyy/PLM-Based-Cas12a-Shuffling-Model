from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cas12a_shuffling_model.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _find_best_teacher_run(outputs_root: Path) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    candidates: list[tuple[int, float, Path, dict[str, Any], pd.DataFrame]] = []
    for metrics_path in outputs_root.rglob("metrics_summary.json"):
        if "teacher_adapt" not in str(metrics_path):
            continue
        run_dir = metrics_path.parent
        history_path = run_dir / "train_history.csv"
        model_dir = run_dir / "adapted_teacher_model"
        if not history_path.exists() or not model_dir.exists():
            continue
        try:
            history_df = pd.read_csv(history_path)
        except Exception:
            continue
        if history_df.empty:
            continue
        metrics = _load_json(metrics_path)
        candidates.append(
            (
                int(len(history_df)),
                float(metrics_path.stat().st_mtime),
                run_dir,
                metrics,
                history_df,
            )
        )

    if not candidates:
        raise FileNotFoundError("No successful teacher adaptation run with metrics/history found.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, run_dir, metrics, history_df = candidates[0]
    return run_dir, metrics, history_df


def _find_best_student_run(outputs_root: Path) -> tuple[Path, dict[str, Any], pd.DataFrame | None]:
    candidates: list[tuple[int, int, float, Path, dict[str, Any], pd.DataFrame | None]] = []
    for metrics_path in outputs_root.rglob("metrics_summary.json"):
        if "/student/" not in str(metrics_path):
            continue
        try:
            metrics = _load_json(metrics_path)
        except Exception:
            continue
        final_metrics = metrics.get("final_metrics", {})
        if not isinstance(final_metrics, dict) or "global_corr" not in final_metrics:
            continue
        run_dir = metrics_path.parent
        history_path = run_dir / "train_history.csv"
        history_df = pd.read_csv(history_path) if history_path.exists() else None
        has_source_specific = int(
            any(k in final_metrics for k in ("global_corr_natural", "global_corr_chimera"))
        )
        history_len = int(len(history_df)) if history_df is not None else 0
        candidates.append(
            (
                has_source_specific,
                history_len,
                float(metrics_path.stat().st_mtime),
                run_dir,
                metrics,
                history_df,
            )
        )

    if not candidates:
        raise FileNotFoundError("No successful student training run with metrics found.")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, _, run_dir, metrics, history_df = candidates[0]
    return run_dir, metrics, history_df


def _teacher_curve_data(
    *,
    run_dir: Path,
    metrics: dict[str, Any],
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    data = history_df.copy()
    if "epoch" not in data.columns:
        data["epoch"] = np.arange(1, len(data) + 1)
    if "global_step" not in data.columns:
        data["global_step"] = np.nan
    if "val_ppl" not in data.columns and "best_val_ppl" in metrics:
        data["val_ppl"] = np.nan
    if "train_ppl" not in data.columns:
        data["train_ppl"] = np.nan

    best_val = pd.to_numeric(data.get("val_loss"), errors="coerce")
    data["best_val_point"] = False
    if best_val.notna().any():
        best_idx = best_val.idxmin()
        data.loc[best_idx, "best_val_point"] = True

    data.insert(0, "run_dir", str(run_dir))
    data.insert(1, "run_name", run_dir.name)
    return data[
        [
            "run_dir",
            "run_name",
            "epoch",
            "global_step",
            "train_loss",
            "val_loss",
            "train_ppl",
            "val_ppl",
            "best_val_point",
        ]
    ].copy()


def _plot_teacher_curve(data: pd.DataFrame, out_png: Path, out_pdf: Path, run_name: str) -> None:
    x = data["epoch"].astype(float).to_numpy()
    train_loss = pd.to_numeric(data["train_loss"], errors="coerce").to_numpy(dtype=float)
    val_loss = pd.to_numeric(data["val_loss"], errors="coerce").to_numpy(dtype=float)
    val_ppl = pd.to_numeric(data["val_ppl"], errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax0, ax1 = axes

    ax0.plot(x, train_loss, color="#1f4e79", marker="o", lw=1.8, label="Train loss")
    ax0.plot(x, val_loss, color="#b22222", marker="s", lw=1.8, label="Validation loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.set_title("A. Adaptation Loss")
    ax0.legend(frameon=False, loc="best")
    ax0.grid(axis="y", alpha=0.18)

    best_mask = data["best_val_point"].astype(bool).to_numpy()
    if best_mask.any():
        best_x = x[best_mask][0]
        best_y = val_loss[best_mask][0]
        ax0.scatter([best_x], [best_y], color="#b22222", s=55, zorder=4)
        ax0.annotate(
            f"best val={best_y:.2f}",
            xy=(best_x, best_y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#b22222",
        )

    ax1.plot(x, val_ppl, color="#2f6b2f", marker="o", lw=1.8, label="Validation perplexity")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("B. Validation Perplexity")
    ax1.grid(axis="y", alpha=0.18)
    if best_mask.any():
        best_x = x[best_mask][0]
        best_y = val_ppl[best_mask][0]
        ax1.scatter([best_x], [best_y], color="#2f6b2f", s=55, zorder=4)
        ax1.annotate(
            f"best ppl={best_y:,.0f}",
            xy=(best_x, best_y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#2f6b2f",
        )

    fig.suptitle(f"Teacher Adaptation Dynamics ({run_name})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _student_metric_rows(run_dir: Path, metrics: dict[str, Any]) -> pd.DataFrame:
    final_metrics = metrics.get("final_metrics", {})
    rows = [
        {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "panel": "global",
            "metric": "Overall",
            "value": final_metrics.get("global_corr"),
        },
        {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "panel": "global",
            "metric": "Natural",
            "value": final_metrics.get("global_corr_natural"),
        },
        {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "panel": "global",
            "metric": "Chimera",
            "value": final_metrics.get("global_corr_chimera"),
        },
        {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "panel": "junction",
            "metric": "Overall",
            "value": final_metrics.get("junction_mean_corr"),
        },
        {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "panel": "junction",
            "metric": "Chimera",
            "value": final_metrics.get("junction_mean_corr_chimera"),
        },
    ]
    return pd.DataFrame(rows)


def _plot_student_metrics(
    data: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    run_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
    colors = {"Overall": "#1f4e79", "Natural": "#5b8c5a", "Chimera": "#b5651d"}

    for ax, panel_name, title in zip(
        axes,
        ["global", "junction"],
        ["A. Global Correlation", "B. Junction Correlation"],
    ):
        sub = data[data["panel"] == panel_name].copy()
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub[sub["value"].notna()].reset_index(drop=True)
        x = np.arange(len(sub))
        vals = sub["value"].to_numpy(dtype=float)
        labels = sub["metric"].tolist()
        bar_colors = [colors.get(label, "#666666") for label in labels]
        bars = ax.bar(x, vals, color=bar_colors, width=0.62)
        ax.axhline(0.0, color="#444444", lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylim(min(-0.35, np.nanmin(vals) - 0.08), max(0.45, np.nanmax(vals) + 0.12))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        for rect, val in zip(bars, vals):
            offset = 0.02 if val >= 0 else -0.04
            va = "bottom" if val >= 0 else "top"
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                val + offset,
                f"{val:.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    axes[0].set_ylabel("Pearson correlation")
    fig.suptitle(f"Student Distillation Performance ({run_name})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_notes(
    *,
    out_dir: Path,
    teacher_run: Path,
    student_run: Path,
    teacher_metrics: dict[str, Any],
    student_metrics: dict[str, Any],
) -> None:
    final_metrics = student_metrics.get("final_metrics", {})
    note = f"""# Paper Figure Notes

## Teacher figure source
- run: `{teacher_run}`
- best_val_loss: {teacher_metrics.get("best_val_loss")}
- best_val_ppl: {teacher_metrics.get("best_val_ppl")}

## Student figure source
- run: `{student_run}`
- global_corr: {final_metrics.get("global_corr")}
- global_corr_natural: {final_metrics.get("global_corr_natural")}
- global_corr_chimera: {final_metrics.get("global_corr_chimera")}
- junction_mean_corr: {final_metrics.get("junction_mean_corr")}
- junction_mean_corr_chimera: {final_metrics.get("junction_mean_corr_chimera")}
"""
    (out_dir / "paper_figure_notes.md").write_text(note, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="cas12a_shuffling_model/outputs")
    ap.add_argument("--out-dir", default="cas12a_shuffling_model/outputs/paper_figures")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    _configure_matplotlib()

    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_run, teacher_metrics, teacher_history = _find_best_teacher_run(outputs_root)
    student_run, student_metrics, _student_history = _find_best_student_run(outputs_root)

    teacher_data = _teacher_curve_data(
        run_dir=teacher_run,
        metrics=teacher_metrics,
        history_df=teacher_history,
    )
    teacher_data.to_csv(out_dir / "teacher_adaptation_curve_data.csv", index=False)
    _plot_teacher_curve(
        teacher_data,
        out_png=out_dir / "teacher_adaptation_curve.png",
        out_pdf=out_dir / "teacher_adaptation_curve.pdf",
        run_name=teacher_run.name,
    )

    student_data = _student_metric_rows(student_run, student_metrics)
    student_data.to_csv(out_dir / "student_distillation_metrics_data.csv", index=False)
    _plot_student_metrics(
        student_data,
        out_png=out_dir / "student_distillation_metrics.png",
        out_pdf=out_dir / "student_distillation_metrics.pdf",
        run_name=student_run.name,
    )

    manifest = {
        "teacher_run": str(teacher_run),
        "student_run": str(student_run),
        "teacher_outputs": {
            "png": str(out_dir / "teacher_adaptation_curve.png"),
            "pdf": str(out_dir / "teacher_adaptation_curve.pdf"),
            "csv": str(out_dir / "teacher_adaptation_curve_data.csv"),
        },
        "student_outputs": {
            "png": str(out_dir / "student_distillation_metrics.png"),
            "pdf": str(out_dir / "student_distillation_metrics.pdf"),
            "csv": str(out_dir / "student_distillation_metrics_data.csv"),
        },
    }
    (out_dir / "paper_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_notes(
        out_dir=out_dir,
        teacher_run=teacher_run,
        student_run=student_run,
        teacher_metrics=teacher_metrics,
        student_metrics=student_metrics,
    )
    logger.info("Paper figures written to %s", out_dir)


if __name__ == "__main__":
    main()
