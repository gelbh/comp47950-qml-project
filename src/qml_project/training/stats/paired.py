"""Paired effect sizes and Wilcoxon-based significance tests with Bonferroni."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


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


def _paired_wilcoxon_with_bonferroni(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_tests: int,
    wilcoxon: Any,
) -> tuple[float, float, float]:
    """Return ``(stat, p_value, p_value_bonferroni)`` for a paired Wilcoxon test.

    Skips the Wilcoxon call when the two sample vectors are numerically identical
    and returns ``(0.0, 1.0)`` in that case. The corrected p-value is clamped to 1.
    """
    if np.allclose(x, y):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    p_corr = min(1.0, float(p_val) * max(1, int(n_tests)))
    return float(stat), float(p_val), float(p_corr)


def sample_efficiency_stat_tests(
    df: pd.DataFrame,
    *,
    metric: str,
    train_sizes: Sequence[int],
    seed_col: str = "seed",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pairwise train-size significance tests for one metric.

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
            stat, p_val, p_corr = _paired_wilcoxon_with_bonferroni(
                x, y, n_tests=m_tests, wilcoxon=wilcoxon
            )
            deltas = y - x
            rows.append(
                {
                    "metric": metric,
                    "family_scope": "within_pipeline_train_size",
                    "correction_method": "bonferroni",
                    "n_tests_family": int(m_tests),
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


def paired_cross_pipeline_stat_tests(
    df: pd.DataFrame,
    *,
    metric: str,
    train_sizes: Sequence[int],
    pipelines: Sequence[str] | None = None,
    group_cols: Sequence[str] = (),
    pipeline_col: str = "pipeline",
    train_size_col: str = "train_size",
    seed_col: str = "seed",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired Wilcoxon/effect-size tests between pipelines at fixed train size.

    For each ``train_size`` and optional group in ``group_cols``, aligns runs by
    common seeds and performs paired two-sided Wilcoxon signed-rank tests for
    every pipeline pair. Bonferroni correction is applied within each fixed-size
    family.
    """
    required = {metric, pipeline_col, train_size_col, seed_col}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    try:
        from scipy.stats import wilcoxon
    except Exception:
        return pd.DataFrame()

    work = df.copy()
    work = work.dropna(subset=[metric, pipeline_col, train_size_col, seed_col])
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str | bool]] = []
    for size in train_sizes:
        at_size = work.loc[work[train_size_col] == size]
        if at_size.empty:
            continue

        if group_cols:
            grouped_iter = at_size.groupby(list(group_cols), dropna=False)
        else:
            grouped_iter = [((), at_size)]

        for group_key, sub in grouped_iter:
            available = (
                sorted(sub[pipeline_col].dropna().astype(str).unique().tolist())
                if pipelines is None
                else [p for p in pipelines if p in set(sub[pipeline_col].astype(str))]
            )
            if len(available) < 2:
                continue
            m_tests = max(1, math.comb(len(available), 2))
            base_family_id = f"train_size={int(size)}"

            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    pa, pb = available[i], available[j]
                    sa = (
                        sub.loc[sub[pipeline_col].astype(str) == pa, [seed_col, metric]]
                        .groupby(seed_col, dropna=False)[metric]
                        .mean()
                    )
                    sb = (
                        sub.loc[sub[pipeline_col].astype(str) == pb, [seed_col, metric]]
                        .groupby(seed_col, dropna=False)[metric]
                        .mean()
                    )
                    common = sa.index.intersection(sb.index)
                    if common.empty:
                        continue

                    x = sa.loc[common].to_numpy(dtype=np.float64)
                    y = sb.loc[common].to_numpy(dtype=np.float64)
                    if x.size == 0:
                        continue
                    stat, p_val, p_corr = _paired_wilcoxon_with_bonferroni(
                        x, y, n_tests=m_tests, wilcoxon=wilcoxon
                    )
                    deltas = y - x

                    row: dict[str, float | int | str | bool] = {
                        "metric": metric,
                        "family_scope": "cross_pipeline_fixed_train_size",
                        "correction_method": "bonferroni",
                        "family_id": base_family_id,
                        "n_tests_family": int(m_tests),
                        "train_size": int(size),
                        "pipeline_a": str(pa),
                        "pipeline_b": str(pb),
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
                    if group_cols:
                        family_id = base_family_id
                        if not isinstance(group_key, tuple):
                            group_key = (group_key,)
                        for col, val in zip(group_cols, group_key):
                            row[col] = val
                            family_id += f"|{col}={val}"
                        row["family_id"] = family_id
                    rows.append(row)

    return pd.DataFrame(rows)
