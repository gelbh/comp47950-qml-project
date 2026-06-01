"""Parity feature-set ablation for classical SVM (RBF) baselines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from qml_project.baselines.features import FEATURE_SET_DESCRIPTIONS, PARITY_ABLATION_FEATURE_SETS
from qml_project.baselines.sweep import run_classical_sweep
from qml_project.baselines.sweep_results import SweepResults

_METRIC_COLS = (
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "mcc_mean",
    "mcc_std",
)


def run_parity_feature_ablation_sweep(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    mlflow_experiment: str,
    use_cache: bool,
    max_workers: int | None,
    M: int = 7,
    seeds: Sequence[int] | None = None,
    train_sizes: Sequence[int | str] | None = None,
    n_games_win_rate: int = 200,
    c_svc: float = 1.0,
) -> SweepResults:
    """Run the parity-style feature ablation grid (SVM RBF, no symmetry aug).

    Matches the notebook §3.4 design: train subsets 25/50/100/150/full, ten seeds,
    OOD evaluation, no win-rate games (``compute_win_rate=False``).
    """
    if seeds is None:
        seeds = tuple(range(10))
    if train_sizes is None:
        train_sizes = (25, 50, 100, 150, "full")
    return run_classical_sweep(
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        model_names=("SVM (RBF)",),
        feature_sets=PARITY_ABLATION_FEATURE_SETS,
        symmetry_variants=("none",),
        train_sizes=train_sizes,
        seeds=seeds,
        M=M,
        compute_win_rate=False,
        n_games_win_rate=n_games_win_rate,
        mlflow_experiment=mlflow_experiment,
        use_cache=use_cache,
        max_workers=max_workers,
        c_svc=c_svc,
    )


def parity_ablation_summary_display(
    sweep: SweepResults,
    *,
    group_cols: Sequence[str] = ("feature_set", "train_size"),
    round_metrics: int = 3,
) -> pd.DataFrame:
    """Aggregate over seeds and attach human-readable feature-set descriptions."""
    summary = sweep.summary(group_cols=group_cols)
    display_df = summary[["feature_set", "train_size", *_METRIC_COLS]].copy()
    display_df.loc[:, list(_METRIC_COLS)] = display_df.loc[:, list(_METRIC_COLS)].round(
        round_metrics
    )
    display_df["description"] = display_df["feature_set"].map(FEATURE_SET_DESCRIPTIONS)
    return display_df


__all__ = [
    "parity_ablation_summary_display",
    "run_parity_feature_ablation_sweep",
]
