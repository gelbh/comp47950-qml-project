"""Canonical cost/speed metric helpers for cross-pipeline reporting.

These helpers standardise metric naming and define explicit NA policies so
notebook/report tables can compare classical, QSVM, and VQC pipelines without
hand-written per-section logic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def add_cost_metric_contract_columns(
    df: pd.DataFrame,
    *,
    pipeline: str,
    training_time_col: str | None = None,
    inference_time_col: str | None = None,
    balanced_accuracy_col: str | None = None,
    shots_col: str | None = None,
    train_size_col: str = "train_size",
    max_iter_col: str = "max_iter",
    n_inference_samples: int | None = None,
    random_baseline_bal_acc: float = 0.5,
) -> pd.DataFrame:
    """Return a copy with canonical cost/speed columns.

    Canonical outputs:
    - ``pipeline``
    - ``training_time_s``
    - ``inference_time_s``
    - ``per_sample_latency_ms`` (NA unless ``n_inference_samples`` is provided)
    - ``balanced_accuracy``
    - ``bal_acc_gain_over_random``
    - ``cost_per_bal_acc_point_s`` (NA for non-positive gain)
    - ``total_shots_estimate`` (NA unless shots/iters/train size are available)
    """

    out = df.copy()
    out["pipeline"] = pipeline

    train_col = training_time_col or _first_existing(out, ("training_time", "train_time_s"))
    infer_col = inference_time_col or _first_existing(out, ("inference_time", "inference_time_s"))
    bal_col = balanced_accuracy_col or _first_existing(
        out,
        (
            "balanced_accuracy",
            "balanced_accuracy_raw",
        ),
    )
    shot_col = shots_col or _first_existing(out, ("shots",))

    out["training_time_s"] = (
        pd.to_numeric(out[train_col], errors="coerce")
        if train_col is not None
        else np.nan
    )
    out["inference_time_s"] = (
        pd.to_numeric(out[infer_col], errors="coerce")
        if infer_col is not None
        else np.nan
    )
    out["balanced_accuracy"] = (
        pd.to_numeric(out[bal_col], errors="coerce")
        if bal_col is not None
        else np.nan
    )

    out["bal_acc_gain_over_random"] = out["balanced_accuracy"] - float(
        random_baseline_bal_acc
    )
    # Seconds per +1 percentage-point balanced-accuracy gain over random.
    gain_pp = out["bal_acc_gain_over_random"] * 100.0
    out["cost_per_bal_acc_point_s"] = np.where(
        gain_pp > 0.0,
        out["training_time_s"] / gain_pp,
        np.nan,
    )

    if n_inference_samples is None or n_inference_samples <= 0:
        out["per_sample_latency_ms"] = np.nan
    else:
        out["per_sample_latency_ms"] = (
            out["inference_time_s"] / float(n_inference_samples) * 1000.0
        )

    if (
        shot_col is not None
        and max_iter_col in out.columns
        and train_size_col in out.columns
    ):
        shots = pd.to_numeric(out[shot_col], errors="coerce")
        iters = pd.to_numeric(out[max_iter_col], errors="coerce")
        sizes = pd.to_numeric(out[train_size_col], errors="coerce")
        out["total_shots_estimate"] = shots * iters * sizes
    else:
        out["total_shots_estimate"] = np.nan

    return out


def shots_to_target_balanced_accuracy(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    target_balanced_accuracy: float = 0.9,
    shots_col: str = "shots",
    balanced_accuracy_col: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Compute minimum shots needed to hit target balanced accuracy.

    Returns one row per group with ``shots_to_target_bal_acc``.
    If no row in a group reaches the target (or required columns are missing),
    ``shots_to_target_bal_acc`` is NA.
    """

    if shots_col not in df.columns or balanced_accuracy_col not in df.columns:
        cols = list(group_cols) + ["shots_to_target_bal_acc"]
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work[shots_col] = pd.to_numeric(work[shots_col], errors="coerce")
    work[balanced_accuracy_col] = pd.to_numeric(
        work[balanced_accuracy_col], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    for keys, grp in work.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        hit = grp.loc[grp[balanced_accuracy_col] >= float(target_balanced_accuracy)]
        shots_val = float(hit[shots_col].min()) if not hit.empty else np.nan
        row = {col: key for col, key in zip(group_cols, keys)}
        row["shots_to_target_bal_acc"] = shots_val
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cost_performance(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("pipeline",),
) -> pd.DataFrame:
    """Aggregate canonical cost/speed columns as mean/std per group."""

    metric_cols = [
        "balanced_accuracy",
        "training_time_s",
        "per_sample_latency_ms",
        "total_shots_estimate",
        "cost_per_bal_acc_point_s",
    ]
    use_metrics = [c for c in metric_cols if c in df.columns]
    if not use_metrics:
        return pd.DataFrame()
    grouped = df.groupby(list(group_cols), dropna=False)[use_metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()

