from __future__ import annotations

import heapq
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cas12a_shuffling_model.composition.chimera_repr import (
    SLOT_COUNT,
    enumerate_slot_matrix_batches,
    slot_columns,
    slot_matrix_to_codes,
)
from cas12a_shuffling_model.composition.slot_scorer import SlotScorerPredictor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullScanConfig:
    batch_size: int = 65536
    shortlist_size: int = 20000
    top_k: int = 50
    save_all_scores: bool = False
    include_decomposition: bool = True
    progress_every_batches: int = 4
    start_index: int = 0
    end_index: int | None = None


def _heap_push_topk(heap_rows: list[tuple[float, int]], item: tuple[float, int], max_size: int) -> None:
    if len(heap_rows) < max_size:
        heapq.heappush(heap_rows, item)
        return
    if item[0] > heap_rows[0][0]:
        heapq.heapreplace(heap_rows, item)


def _index_rows_with_components(
    *,
    predictor: SlotScorerPredictor,
    indices: np.ndarray,
    include_decomposition: bool,
) -> pd.DataFrame:
    out = predictor.score_indices(indices, return_components=include_decomposition)
    rows = pd.DataFrame(
        {
            "combo_index": indices.astype(np.int64),
            "slot_code_11": out["slot_code_11"],
            "s_scan_score": out["score"].astype(np.float64),
        }
    )
    slots = np.asarray([list(code) for code in rows["slot_code_11"]], dtype=object)
    letter_to_int = {"A": 0, "L": 1, "F": 2, "M": 3}
    for i, col in enumerate(slot_columns()):
        rows[col] = [letter_to_int[ch] for ch in slots[:, i].tolist()]
    if include_decomposition:
        rows["s_scan_main_effect"] = out["main_effect"].astype(np.float64)
        rows["s_scan_pairwise_effect"] = out["pairwise_effect"].astype(np.float64)
        rows["s_scan_nonlinear_effect"] = out["nonlinear_effect"].astype(np.float64)
    return rows


def scan_full_space(
    *,
    slot_scorer_checkpoint: str,
    out_dir: str,
    cfg: FullScanConfig,
    device: str | None = None,
) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    predictor = SlotScorerPredictor(slot_scorer_checkpoint, device=device)

    shortlist_heap: list[tuple[float, int]] = []
    all_csv_path = out_path / "s_scan_all_scores.csv"
    wrote_header = False
    started = time.time()
    processed = 0
    total = (4 ** SLOT_COUNT) if cfg.end_index is None else max(0, int(cfg.end_index) - int(cfg.start_index))

    for batch_no, (idx, slot_mat) in enumerate(
        enumerate_slot_matrix_batches(
            batch_size=int(cfg.batch_size),
            start=int(cfg.start_index),
            stop=cfg.end_index,
        ),
        start=1,
    ):
        score_out = predictor.score_slot_matrix(slot_mat, return_components=False)
        scores = score_out["score"].astype(np.float64)
        processed += int(scores.shape[0])

        keep_n = min(int(cfg.shortlist_size), int(scores.shape[0]))
        top_local = np.argpartition(scores, -keep_n)[-keep_n:]
        for li in top_local.tolist():
            _heap_push_topk(
                shortlist_heap,
                (float(scores[li]), int(idx[li])),
                max_size=int(cfg.shortlist_size),
            )

        if bool(cfg.save_all_scores):
            rows = pd.DataFrame(
                {
                    "combo_index": idx.astype(np.int64),
                    "slot_code_11": slot_matrix_to_codes(slot_mat),
                    "s_scan_score": scores,
                }
            )
            rows.to_csv(all_csv_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True

        if batch_no % max(1, int(cfg.progress_every_batches)) == 0:
            elapsed = max(1e-6, time.time() - started)
            rate = processed / elapsed
            eta_min = ((total - processed) / max(rate, 1e-6)) / 60.0 if total > 0 else float("nan")
            logger.info(
                "S_scan full-space progress: %d/%d (%.2f%%), rate=%.1f combos/s, eta=%.1f min",
                processed,
                total,
                100.0 * processed / max(1, total),
                rate,
                eta_min,
            )

    shortlist_sorted = sorted(shortlist_heap, key=lambda x: x[0], reverse=True)
    shortlist_indices = np.asarray([idx for _, idx in shortlist_sorted], dtype=np.int64)
    shortlist_df = _index_rows_with_components(
        predictor=predictor,
        indices=shortlist_indices,
        include_decomposition=bool(cfg.include_decomposition),
    )
    shortlist_df = shortlist_df.sort_values("s_scan_score", ascending=False).reset_index(drop=True)
    shortlist_df["s_scan_rank"] = np.arange(1, len(shortlist_df) + 1, dtype=np.int64)
    top_df = shortlist_df.head(int(cfg.top_k)).copy().reset_index(drop=True)
    top_df["final_rank"] = np.arange(1, len(top_df) + 1, dtype=np.int64)

    shortlist_path = out_path / "s_scan_shortlist.csv"
    top_path = out_path / "s_scan_top.csv"
    shortlist_df.to_csv(shortlist_path, index=False)
    top_df.to_csv(top_path, index=False)

    summary = {
        "slot_scorer_checkpoint": slot_scorer_checkpoint,
        "processed": int(processed),
        "total": int(total),
        "elapsed_sec": float(time.time() - started),
        "shortlist_size": int(cfg.shortlist_size),
        "top_k": int(cfg.top_k),
        "batch_size": int(cfg.batch_size),
    }
    summary_path = out_path / "s_scan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    outputs = {
        "shortlist_csv": str(shortlist_path),
        "top_csv": str(top_path),
        "summary_json": str(summary_path),
    }
    if bool(cfg.save_all_scores):
        outputs["all_scores_csv"] = str(all_csv_path)
    return outputs
