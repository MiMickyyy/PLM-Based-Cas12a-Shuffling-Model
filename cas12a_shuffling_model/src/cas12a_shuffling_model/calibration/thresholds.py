from __future__ import annotations

import numpy as np


def compute_s_min_threshold(active_junction_min: list[float], quantile: float = 0.1) -> float:
    vals = np.asarray(active_junction_min, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    q = min(1.0, max(0.0, float(quantile)))
    return float(np.quantile(vals, q))

