"""Aggregated result container for classical baseline sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from qml_project.baselines.evaluation import ClassicalResult
from qml_project.training.stats import (
    _grouped_bootstrap_summary,
    fit_power_law_learning_curve,
    sample_efficiency_stat_tests,
)


@dataclass
class SweepResults:
    """Aggregated results from the full classical baseline sweep."""

    results: list[ClassicalResult] = field(default_factory=list)
    sweep_metadata: dict[str, float | int | None] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a tidy DataFrame (one row per run)."""
        cols = (
            ("model", "model_name"),
            ("feature_set", "feature_set"),
            ("symmetry", "symmetry"),
            ("train_size", "train_size"),
            ("seed", "seed"),
            ("regime", "regime"),
            ("c_svc", "c_svc"),
            ("accuracy", "accuracy"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("mcc", "mcc"),
            ("f1", "f1"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("train_time_s", "train_time_s"),
            ("inference_time_s", "inference_time_s"),
            ("win_rate", "win_rate"),
        )
        rows = [{out: getattr(r, attr) for out, attr in cols} for r in self.results]
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = (
            "model",
            "feature_set",
            "symmetry",
            "train_size",
            "regime",
            "c_svc",
        ),
        *,
        bootstrap_random_state: int = 42,
    ) -> pd.DataFrame:
        """Aggregate over seeds with mean/std and bootstrap confidence intervals."""
        return _grouped_bootstrap_summary(
            self.to_dataframe(),
            group_cols,
            (
                "accuracy",
                "balanced_accuracy",
                "mcc",
                "f1",
                "train_time_s",
                "win_rate",
            ),
            bootstrap_random_state=bootstrap_random_state,
        )

    def statistical_tests(
        self,
        *,
        metrics: Sequence[str] = ("balanced_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Paired Wilcoxon/effect-size tests across train sizes per config."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())

        frames: list[pd.DataFrame] = []
        group_cols = ["model", "feature_set", "symmetry", "regime", "c_svc"]
        for keys, g in df.groupby(group_cols, dropna=False):
            for metric in metrics:
                if metric not in g.columns:
                    continue
                stats = sample_efficiency_stat_tests(
                    g,
                    metric=metric,
                    train_sizes=train_sizes,
                    alpha=alpha,
                )
                if stats.empty:
                    continue
                stats = stats.copy()
                for col, value in zip(group_cols, keys):
                    stats[col] = value
                frames.append(stats)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def power_law_fits(
        self,
        *,
        metrics: Sequence[str] = ("balanced_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Power-law fits of metric means over train sizes per config."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())

        rows: list[dict[str, float | str]] = []
        group_cols = ["model", "feature_set", "symmetry", "regime", "c_svc"]
        for keys, g in df.groupby(group_cols, dropna=False):
            for metric in metrics:
                if metric not in g.columns:
                    continue
                means: list[float] = []
                valid_sizes: list[float] = []
                for size in train_sizes:
                    vals = g.loc[g["train_size"] == size, metric].dropna().to_numpy()
                    if vals.size == 0:
                        continue
                    valid_sizes.append(float(size))
                    means.append(float(np.mean(vals)))
                if len(valid_sizes) < 3:
                    continue
                fit = fit_power_law_learning_curve(valid_sizes, means)
                row: dict[str, float | str] = {
                    "metric": metric,
                    **fit,
                    "n_points": float(len(valid_sizes)),
                }
                for col, value in zip(group_cols, keys):
                    row[col] = str(value)
                rows.append(row)
        return pd.DataFrame(rows)
