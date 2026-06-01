"""Learning-curve fits."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd


def fit_power_law_learning_curve(
    train_sizes: Sequence[float],
    metric_values: Sequence[float],
) -> dict[str, float]:
    """Fit ``accuracy = a - b * n^(-c)`` and return ``a``, ``b``, ``c``, and ``r2``.

    Returns ``nan`` for every key if the fit cannot be performed (fewer than 3
    points, mismatched sizes, or ``curve_fit`` failure).
    """
    x = np.asarray(train_sizes, dtype=np.float64)
    y = np.asarray(metric_values, dtype=np.float64)
    nan_result = {
        "a": float("nan"),
        "b": float("nan"),
        "c": float("nan"),
        "r2": float("nan"),
    }
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return nan_result

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
        return nan_result


def learning_curve_xy_for_power_law(
    curve: pd.DataFrame,
    *,
    metric_col: str,
    full_train_n: int,
    train_size_col: str = "train_size",
    full_sentinel: Any = "full",
) -> tuple[list[float], list[float]]:
    """Collect numeric train sizes and metric means from a learning-curve frame.

    When ``train_size`` equals ``full_sentinel``, ``full_train_n`` is used as the
    effective size (full-data aggregation point). Rows that cannot be coerced
    to floats are skipped.
    """
    sizes: list[float] = []
    values: list[float] = []
    for _, row in curve.iterrows():
        ts = row[train_size_col]
        if ts == full_sentinel:
            ts = full_train_n
        try:
            sizes.append(float(ts))
            values.append(float(row[metric_col]))
        except (TypeError, ValueError, KeyError):
            continue
    return sizes, values


def power_law_fit_from_learning_curve_dataframe(
    curve: pd.DataFrame,
    *,
    metric_col: str,
    full_train_n: int,
    train_size_col: str = "train_size",
    full_sentinel: Any = "full",
    min_points: int = 3,
) -> dict[str, float] | None:
    """Run :func:`fit_power_law_learning_curve` on extracted learning-curve pairs.

    Returns ``None`` when fewer than ``min_points`` valid pairs are collected.
    Otherwise returns the fit mapping (possibly NaN-filled if the optimiser
    fails).
    """
    sizes, vals = learning_curve_xy_for_power_law(
        curve,
        metric_col=metric_col,
        full_train_n=full_train_n,
        train_size_col=train_size_col,
        full_sentinel=full_sentinel,
    )
    if len(sizes) < min_points:
        return None
    return dict(fit_power_law_learning_curve(sizes, vals))
