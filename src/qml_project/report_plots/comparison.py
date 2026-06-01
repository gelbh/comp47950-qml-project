"""Quantum-selection, deep-dive, and final comparison figures (§7, §9, §11)."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qml_project.training.selection import Winner

from .styles import (
    FINAL_COMPARISON_PIPELINE_STYLES,
    METRIC_TO_CURVE_KEY,
    QUANTUM_WINNER_PIPELINE_STYLES,
    train_sizes_to_float_array,
)


def plot_quantum_selection_pareto(
    selection_table: pd.DataFrame,
    quantum_winners: Mapping[str, Winner],
    *,
    figsize: tuple[float, float] = (7.0, 4.5),
    pipeline_styles: Mapping[str, Mapping[str, Any]] | None = None,
    fallback_winner_style: Mapping[str, Any] | None = None,
) -> tuple[Any, Any] | None:
    """Figure 7.1: mean OOD balanced accuracy vs mean training time at deploy train size.

    The x-axis uses a log scale when every finite ``mean_cost`` is strictly
    positive (typical QSVM vs VQC wall-clock spread); Pareto membership is
    still computed in linear time upstream.

    Non-winner points split into dominated (grey) and Pareto-front (dark
    diamonds); per-pipeline winners use ``pipeline_styles`` (defaults to
    ``QUANTUM_WINNER_PIPELINE_STYLES``).
    """
    import matplotlib.pyplot as plt

    if selection_table.empty:
        return None
    if pipeline_styles is None:
        pipeline_styles = QUANTUM_WINNER_PIPELINE_STYLES
    if fallback_winner_style is None:
        fallback_winner_style = {"marker": "P", "color": "#2a9d8f"}

    if selection_table["mean_cost"].notna().sum() == 0:
        warnings.warn(
            "Cost column training_time_s is entirely NaN in selection_table; "
            "Pareto scatter would be empty. Check upstream frames expose this column.",
            stacklevel=2,
        )

    fig, ax = plt.subplots(figsize=figsize)
    _non_winner = selection_table.loc[~selection_table["winner"]]
    for is_pareto, color, marker, size, label in [
        (False, "#bbbbbb", "o", 30, "dominated"),
        (True, "#264653", "D", 60, "pareto front"),
    ]:
        sub = _non_winner[_non_winner["pareto"] == is_pareto]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_cost"],
            sub["mean_accuracy"],
            color=color,
            marker=marker,
            s=size,
            label=label,
            alpha=0.75,
            edgecolor="none",
        )
    for _pipeline, _w in quantum_winners.items():
        style = pipeline_styles.get(_pipeline, fallback_winner_style)
        ax.scatter(
            [_w.mean_cost] if _w.mean_cost is not None else [np.nan],
            [_w.mean_accuracy],
            color=style["color"],
            marker=style["marker"],
            s=220,
            edgecolor="black",
            linewidth=1.0,
            label=f"winner ({_pipeline})",
            zorder=5,
        )
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    _cost_vals = pd.to_numeric(selection_table["mean_cost"], errors="coerce").to_numpy()
    for _w in quantum_winners.values():
        if _w.mean_cost is not None:
            _cost_vals = np.append(_cost_vals, float(_w.mean_cost))
    _finite_costs = _cost_vals[np.isfinite(_cost_vals)]
    _use_log_x = _finite_costs.size > 0 and bool(np.all(_finite_costs > 0))
    if _use_log_x:
        ax.set_xscale("log")
        ax.set_xlabel("Mean training time (s, log scale)")
    else:
        ax.set_xlabel("Mean training time (s)")
        if _finite_costs.size and not bool(np.all(_finite_costs > 0)):
            warnings.warn(
                "mean_cost contains non-positive values; x-axis stays linear for Figure 7.1.",
                stacklevel=2,
            )
    ax.set_ylabel("Mean OOD balanced accuracy")
    ax.set_title("Quantum selection — Pareto front (per-pipeline winners marked)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, which="both" if _use_log_x else "major")
    plt.tight_layout()
    return fig, ax


def plot_quantum_winner_two_metric_panels(
    quantum_deep_dive_by_pipeline: Mapping[str, Mapping[str, Any]],
    *,
    full_train_n: int,
    pipeline_styles: Mapping[str, Mapping[str, Any]] | None = None,
    figsize: tuple[float, float] = (11.0, 4.0),
) -> tuple[Any, np.ndarray] | None:
    """Figure 9.1: balanced accuracy and win rate vs train size per pipeline winner."""
    import matplotlib.pyplot as plt

    if not quantum_deep_dive_by_pipeline:
        return None
    if pipeline_styles is None:
        pipeline_styles = QUANTUM_WINNER_PIPELINE_STYLES

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for title_idx, (title, mcol) in enumerate(
        [("Balanced accuracy (OOD)", "balanced_accuracy"), ("Win rate vs random", "win_rate")]
    ):
        ax = axes[title_idx]
        curve_key = METRIC_TO_CURVE_KEY[mcol]
        for pipeline, bundle in quantum_deep_dive_by_pipeline.items():
            curve = bundle.get(curve_key, pd.DataFrame())
            if curve.empty or f"{mcol}_mean" not in curve.columns:
                continue
            x = train_sizes_to_float_array(curve["train_size"], full_n=full_train_n)
            y = curve[f"{mcol}_mean"].to_numpy()
            e = curve[f"{mcol}_std"].fillna(0).to_numpy()
            style = pipeline_styles.get(pipeline, {"color": None, "marker": "o"})
            ax.plot(x, y, label=pipeline, **style)
            ax.fill_between(
                x,
                np.maximum(0, y - e),
                np.minimum(1, y + e),
                alpha=0.2,
                color=style["color"],
            )
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_xlabel("Train size")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Metric")
    plt.tight_layout()
    return fig, axes


def plot_quantum_vs_classical_balanced_accuracy_overlay(
    quantum_deep_dive_by_pipeline: Mapping[str, Mapping[str, Any]],
    classical_curve: pd.DataFrame | None,
    *,
    full_train_n: int,
    pipeline_styles: Mapping[str, Mapping[str, Any]] | None = None,
    classical_color: str = "#4361ee",
    classical_label: str = "classical baseline",
    figsize: tuple[float, float] = (8.0, 4.5),
    title: str = (
        "Figure 9.2: Sample efficiency — quantum winners vs classical raw best"
    ),
) -> tuple[Any, Any] | None:
    """Figure 9.2: quantum winners vs a classical learning curve (caller supplies rows).

    The notebook passes the §11 ``classical_raw_best`` row slice (raw features,
    best model×symmetry by mean balanced accuracy across train sizes) so the
    overlay matches the Tier A classical comparator; use ``classical_label`` if
    the curve is instead a pooled ``main`` sweep mean.
    """
    import matplotlib.pyplot as plt

    if not quantum_deep_dive_by_pipeline:
        return None
    if pipeline_styles is None:
        pipeline_styles = QUANTUM_WINNER_PIPELINE_STYLES

    fig, ax = plt.subplots(figsize=figsize)
    for pipeline, bundle in quantum_deep_dive_by_pipeline.items():
        curve = bundle["bal_curve"]
        if curve.empty:
            continue
        x = train_sizes_to_float_array(curve["train_size"], full_n=full_train_n)
        y = curve["balanced_accuracy_mean"].to_numpy()
        e = curve["balanced_accuracy_std"].fillna(0).to_numpy()
        style = pipeline_styles.get(pipeline, {"color": None, "marker": "o"})
        ax.plot(x, y, label=f"quantum / {pipeline}", linewidth=2, **style)
        ax.fill_between(
            x,
            np.maximum(0, y - e),
            np.minimum(1, y + e),
            alpha=0.15,
            color=style["color"],
        )

    if classical_curve is not None and not classical_curve.empty:
        x = train_sizes_to_float_array(classical_curve["train_size"], full_n=full_train_n)
        y = classical_curve["balanced_accuracy_mean"].to_numpy()
        e = classical_curve["balanced_accuracy_std"].fillna(0).to_numpy()
        ax.plot(
            x,
            y,
            label=classical_label,
            color=classical_color,
            marker="o",
            linewidth=2,
        )
        ax.fill_between(
            x,
            np.maximum(0, y - e),
            np.minimum(1, y + e),
            alpha=0.12,
            color=classical_color,
        )

    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Balanced accuracy (OOD)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig, ax


def plot_final_comparison_balanced_accuracy(
    comparison_train_size_summary: pd.DataFrame,
    *,
    pipeline_styles: Mapping[str, Any] | None = None,
    skip_error_band_for: frozenset[str] = frozenset({"classical_pool"}),
    figsize: tuple[float, float] = (8.5, 4.8),
) -> tuple[Any, Any] | None:
    """Section 11.6: mean ± std balanced accuracy vs train size by pipeline."""
    import matplotlib.pyplot as plt

    if comparison_train_size_summary.empty:
        return None
    if pipeline_styles is None:
        pipeline_styles = FINAL_COMPARISON_PIPELINE_STYLES

    fig, ax = plt.subplots(figsize=figsize)
    bal = comparison_train_size_summary.loc[
        comparison_train_size_summary["metric"] == "balanced_accuracy"
    ].copy()
    for pipeline, grp in bal.groupby("pipeline"):
        grp = grp.sort_values("train_size")
        x = grp["train_size"].to_numpy()
        y = grp["mean"].to_numpy()
        e = grp["std"].fillna(0).to_numpy()
        style = pipeline_styles.get(
            str(pipeline),
            {"color": None, "marker": "o", "linestyle": "-"},
        )
        ax.plot(x, y, label=str(pipeline), **style)
        if str(pipeline) in skip_error_band_for:
            continue
        ax.fill_between(
            x,
            np.maximum(0, y - e),
            np.minimum(1, y + e),
            alpha=0.15,
            color=style.get("color"),
        )
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Train size")
    ax.set_ylabel("Balanced accuracy (OOD)")
    ax.set_title("Final comparison — classical × sim-quantum × device")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig, ax


__all__ = [
    "plot_final_comparison_balanced_accuracy",
    "plot_quantum_selection_pareto",
    "plot_quantum_vs_classical_balanced_accuracy_overlay",
    "plot_quantum_winner_two_metric_panels",
]
