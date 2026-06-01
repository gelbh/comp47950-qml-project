"""Result containers with aggregation helpers for VQC sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from qml_project.training.stats import (
    _grouped_bootstrap_summary,
    fit_power_law_learning_curve,
    sample_efficiency_stat_tests,
)
from qml_project.training.types import SimulatedVQCRunResult, VqcNoiseSweepRunResult


@dataclass
class SimulatedVQCSweepResults:
    """Collection of simulated VQC runs across train sizes and seeds."""

    results: list[SimulatedVQCRunResult] = field(default_factory=list)
    sweep_metadata: dict[str, float | int | None] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str | None]] = []
        for r in self.results:
            rows.append(
                {
                    "train_size": r.train_size,
                    "seed": r.seed,
                    "test_accuracy": r.test_accuracy,
                    "balanced_accuracy": r.balanced_accuracy,
                    "mcc": r.mcc,
                    "win_rate": r.win_rate,
                    "training_time": r.training_time,
                    "inference_time": r.inference_time,
                    "final_loss": r.final_loss,
                    "ansatz": r.ansatz,
                    "observable": r.observable,
                    "decision_rule": r.decision_rule,
                    "loss_name": r.loss_name,
                }
            )
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = ("train_size", "ansatz", "loss_name"),
    ) -> pd.DataFrame:
        """Aggregate per-seed metrics as mean/std and bootstrap CI."""
        return _grouped_bootstrap_summary(
            self.to_dataframe(),
            group_cols,
            (
                "test_accuracy",
                "balanced_accuracy",
                "mcc",
                "win_rate",
                "training_time",
                "inference_time",
            ),
        )

    def statistical_tests(
        self,
        *,
        metrics: Sequence[str] = ("test_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Paired Wilcoxon + effect-size tests across train sizes."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())
        frames = [
            sample_efficiency_stat_tests(
                df,
                metric=m,
                train_sizes=train_sizes,
                alpha=alpha,
            )
            for m in metrics
            if m in df.columns
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def power_law_fits(
        self,
        *,
        metrics: Sequence[str] = ("test_accuracy", "win_rate"),
        train_sizes: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Fit power-law learning curves for selected metrics."""
        df = self.to_dataframe()
        if df.empty:
            return df
        if train_sizes is None:
            train_sizes = sorted(df["train_size"].dropna().unique().tolist())
        rows: list[dict[str, float | str]] = []
        for metric in metrics:
            if metric not in df.columns:
                continue
            means: list[float] = []
            valid_sizes: list[float] = []
            for size in train_sizes:
                vals = df.loc[df["train_size"] == size, metric].dropna().to_numpy()
                if vals.size == 0:
                    continue
                valid_sizes.append(float(size))
                means.append(float(np.mean(vals)))
            if len(valid_sizes) < 3:
                continue
            fit = fit_power_law_learning_curve(valid_sizes, means)
            rows.append(
                {
                    "metric": metric,
                    **fit,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class VqcNoiseSweepResults:
    """Collection of VQC noise-sweep runs."""

    results: list[VqcNoiseSweepRunResult] = field(default_factory=list)
    sweep_metadata: dict[str, float | int | None] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float | int | str | None]] = []
        for r in self.results:
            rows.append(
                {
                    "noise_profile": r.noise_profile,
                    "noise_level": r.noise_level,
                    "shots": r.shots,
                    "seed": r.seed,
                    "ansatz": r.ansatz,
                    "training_time": r.training_time,
                    "inference_time": r.inference_time,
                    "final_loss": r.final_loss,
                    "test_accuracy_raw": r.test_accuracy_raw,
                    "balanced_accuracy_raw": r.balanced_accuracy_raw,
                    "mcc_raw": r.mcc_raw,
                    "test_accuracy_readout": r.test_accuracy_readout,
                    "balanced_accuracy_readout": r.balanced_accuracy_readout,
                    "mcc_readout": r.mcc_readout,
                    "test_accuracy_zne": r.test_accuracy_zne,
                    "balanced_accuracy_zne": r.balanced_accuracy_zne,
                    "mcc_zne": r.mcc_zne,
                    "test_accuracy_readout_zne": r.test_accuracy_readout_zne,
                    "balanced_accuracy_readout_zne": r.balanced_accuracy_readout_zne,
                    "mcc_readout_zne": r.mcc_readout_zne,
                }
            )
        return pd.DataFrame(rows)

    def summary(
        self,
        group_cols: Sequence[str] = ("noise_profile", "noise_level", "shots"),
    ) -> pd.DataFrame:
        """Aggregate run metrics as mean/std and bootstrap confidence intervals."""
        return _grouped_bootstrap_summary(
            self.to_dataframe(),
            group_cols,
            (
                "test_accuracy_raw",
                "balanced_accuracy_raw",
                "mcc_raw",
                "test_accuracy_readout",
                "balanced_accuracy_readout",
                "mcc_readout",
                "test_accuracy_zne",
                "balanced_accuracy_zne",
                "mcc_zne",
                "test_accuracy_readout_zne",
                "balanced_accuracy_readout_zne",
                "mcc_readout_zne",
                "training_time",
                "inference_time",
            ),
        )
