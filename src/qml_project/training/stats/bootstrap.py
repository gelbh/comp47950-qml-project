"""Bootstrap mean confidence intervals and per-group bootstrap summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def bootstrap_mean_ci(
    values: np.ndarray | Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the sample mean."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        v = float(vals[0])
        return (v, v)

    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    means = vals[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (lo, hi)


def _grouped_bootstrap_summary(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    *,
    bootstrap_random_state: int = 42,
) -> pd.DataFrame:
    """Mean, std, and bootstrap CI for numeric columns within each group."""
    if df.empty:
        return df
    grouped = df.groupby(list(group_cols), dropna=False)
    rows: list[dict[str, float | int | str]] = []
    for keys, sub in grouped:
        row: dict[str, float | int | str] = {}
        if isinstance(keys, tuple):
            for k, v in zip(group_cols, keys):
                row[k] = v
        else:
            row[group_cols[0]] = keys
        row["n_runs"] = int(len(sub))
        for m in metric_cols:
            vals = sub[m].dropna().to_numpy(dtype=np.float64)
            if vals.size == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_ci_low"] = float("nan")
                row[f"{m}_ci_high"] = float("nan")
                continue
            ci_low, ci_high = bootstrap_mean_ci(
                vals, random_state=bootstrap_random_state
            )
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals))
            row[f"{m}_ci_low"] = float(ci_low)
            row[f"{m}_ci_high"] = float(ci_high)
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)
