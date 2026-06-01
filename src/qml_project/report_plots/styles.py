"""Pipeline style dictionaries and small helpers shared by every figure."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

QUANTUM_WINNER_PIPELINE_STYLES: dict[str, dict[str, Any]] = {
    "vqc": {"color": "#b5179e", "marker": "^"},
    "qsvm": {"color": "#f77f00", "marker": "*"},
}

FINAL_COMPARISON_PIPELINE_STYLES: dict[str, dict[str, Any]] = {
    "classical_raw_best": {"color": "#4361ee", "marker": "o", "linestyle": "-"},
    "classical_parity_best": {"color": "#2a9d8f", "marker": "s", "linestyle": "-"},
    "classical_pool": {
        "color": "#8d99ae",
        "marker": "x",
        "linestyle": ":",
        "alpha": 0.55,
    },
    "classical": {"color": "#4361ee", "marker": "o", "linestyle": "-"},
    "sim_quantum_vqc": {"color": "#b5179e", "marker": "^", "linestyle": "-"},
    "sim_quantum_qsvm": {"color": "#f77f00", "marker": "*", "linestyle": "-"},
    "sim_quantum_vqc_heap_only": {"color": "#9d4edd", "marker": "v", "linestyle": "--"},
    "sim_quantum_qsvm_heap_only": {"color": "#fb8500", "marker": "d", "linestyle": "--"},
    "sim_quantum": {"color": "#2a9d8f", "marker": "s", "linestyle": "-"},
    "device_quantum": {"color": "#e63946", "marker": "D", "linestyle": "--"},
    "device_quantum_vqc": {"color": "#7209b7", "marker": "^", "linestyle": "--"},
    "device_quantum_qsvm": {"color": "#d62828", "marker": "*", "linestyle": "--"},
}

METRIC_TO_CURVE_KEY = {
    "balanced_accuracy": "bal_curve",
    "win_rate": "win_curve",
}


def train_sizes_to_float_array(train_sizes: Any, *, full_n: int) -> np.ndarray:
    """Map ``train_size`` column values to floats; ``\"full\"`` → *full_n*."""
    out: list[float] = []
    for ts in train_sizes:
        if ts == "full":
            out.append(float(full_n))
            continue
        try:
            out.append(float(ts))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def learning_curve_mean_std_by_train_size(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-train-size mean/std/count for a scalar metric column (notebook §9 helper)."""
    if metric not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("train_size", dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n"})
    )


__all__ = [
    "FINAL_COMPARISON_PIPELINE_STYLES",
    "METRIC_TO_CURVE_KEY",
    "QUANTUM_WINNER_PIPELINE_STYLES",
    "learning_curve_mean_std_by_train_size",
    "train_sizes_to_float_array",
]
