"""Bootstrap summaries, learning-curve fits, and sample-efficiency tests."""

from __future__ import annotations

import math
import warnings
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
    """Bootstrap confidence interval for the mean."""
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


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples based on within-seed deltas."""
    diff = np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    if diff.size < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if np.isclose(sd, 0.0):
        return 0.0
    return float(np.mean(diff) / sd)


def rank_biserial_from_deltas(deltas: np.ndarray) -> float:
    """Rank-biserial sign effect size from paired deltas."""
    d = np.asarray(deltas, dtype=np.float64)
    nonzero = d[np.abs(d) > 1e-15]
    if nonzero.size == 0:
        return 0.0
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    return float((n_pos - n_neg) / (n_pos + n_neg))


def sample_efficiency_stat_tests(
    df: pd.DataFrame,
    *,
    metric: str,
    train_sizes: Sequence[int],
    seed_col: str = "seed",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise train-size significance tests for one metric.

    Uses paired Wilcoxon signed-rank on common seeds with Bonferroni correction.
    """
    if metric not in df.columns:
        return pd.DataFrame()

    try:
        from scipy.stats import wilcoxon
    except Exception:
        return pd.DataFrame()

    if len(train_sizes) < 2:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str | bool]] = []
    n_sizes = len(train_sizes)
    m_tests = max(1, math.comb(n_sizes, 2))
    for i in range(n_sizes):
        for j in range(i + 1, n_sizes):
            a, b = train_sizes[i], train_sizes[j]
            sa = (
                df.loc[df["train_size"] == a, [seed_col, metric]]
                .dropna()
                .drop_duplicates(subset=[seed_col])
                .set_index(seed_col)[metric]
            )
            sb = (
                df.loc[df["train_size"] == b, [seed_col, metric]]
                .dropna()
                .drop_duplicates(subset=[seed_col])
                .set_index(seed_col)[metric]
            )
            common = sa.index.intersection(sb.index)
            if common.empty:
                continue
            x = sa.loc[common].to_numpy(dtype=np.float64)
            y = sb.loc[common].to_numpy(dtype=np.float64)
            if x.size == 0:
                continue
            if np.allclose(x, y):
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = wilcoxon(
                    x, y, zero_method="wilcox", alternative="two-sided"
                )
            p_corr = min(1.0, float(p_val) * m_tests)
            deltas = y - x
            rows.append(
                {
                    "metric": metric,
                    "size_a": int(a),
                    "size_b": int(b),
                    "n_pairs": int(x.size),
                    "mean_a": float(np.mean(x)),
                    "std_a": float(np.std(x)),
                    "mean_b": float(np.mean(y)),
                    "std_b": float(np.std(y)),
                    "mean_delta_b_minus_a": float(np.mean(deltas)),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_val),
                    "p_value_bonferroni": float(p_corr),
                    "reject_null_alpha": bool(p_corr < alpha),
                    "cohens_d_paired": float(paired_cohens_d(x, y)),
                    "rank_biserial": float(rank_biserial_from_deltas(deltas)),
                }
            )
    return pd.DataFrame(rows)


def fit_power_law_learning_curve(
    train_sizes: Sequence[float],
    metric_values: Sequence[float],
) -> dict[str, float]:
    """Fit accuracy = a - b * n^(-c) and return fit diagnostics."""
    x = np.asarray(train_sizes, dtype=np.float64)
    y = np.asarray(metric_values, dtype=np.float64)
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "r2": float("nan"),
        }

    def model(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a - b * np.power(n, -c)

    try:
        from scipy.optimize import OptimizeWarning, curve_fit

        p0 = [float(np.max(y)), 0.2, 0.5]
        bounds = ([0.0, 0.0, 1e-6], [2.0, 10.0, 10.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                model, x, y, p0=p0, bounds=bounds, maxfev=50_000
            )
        y_hat = model(x, *params)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float("nan") if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot
        return {
            "a": float(params[0]),
            "b": float(params[1]),
            "c": float(params[2]),
            "r2": float(r2),
        }
    except Exception:
        return {
            "a": float("nan"),
            "b": float("nan"),
            "c": float("nan"),
            "r2": float("nan"),
        }
