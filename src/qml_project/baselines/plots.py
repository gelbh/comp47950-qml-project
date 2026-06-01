"""Matplotlib figures for classical baseline sweeps (notebook §3.x)."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from qml_project.baselines.sweep_results import SweepResults

CLASSICAL_CFG_LEARNING_CURVE_COLORS: Mapping[str, str] = {
    "raw | none": "#4c72b0",
    "raw | augmented": "#64b5f6",
    "raw | canonical": "#1f77b4",
    "parity | none": "#dd8452",
    "parity | augmented": "#ffb74d",
    "parity | canonical": "#c44e52",
}

KERNEL_ALIGNED_MODEL_COLORS: Mapping[str, str] = {
    "SVM (RBF)": "#4c72b0",
    "SVM (Angle Kernel)": "#c44e52",
}

_CLASSICAL_SWEEP_DISPLAY_COLS = [
    "model",
    "feature_set",
    "symmetry",
    "train_size",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "balanced_accuracy_ci_low",
    "balanced_accuracy_ci_high",
    "mcc_mean",
    "mcc_std",
    "mcc_ci_low",
    "mcc_ci_high",
    "win_rate_mean",
    "win_rate_std",
    "win_rate_ci_low",
    "win_rate_ci_high",
    "n_runs",
]

_ROUND_COLS = [
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "balanced_accuracy_ci_low",
    "balanced_accuracy_ci_high",
    "mcc_mean",
    "mcc_std",
    "mcc_ci_low",
    "mcc_ci_high",
    "win_rate_mean",
    "win_rate_std",
    "win_rate_ci_low",
    "win_rate_ci_high",
]


def format_classical_sweep_summary_display(
    sweep: SweepResults,
    *,
    bootstrap_random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap summary table with ``cfg`` column for learning-curve plots."""
    sweep_summary = sweep.summary(bootstrap_random_state=bootstrap_random_state)
    display_cols = [c for c in _CLASSICAL_SWEEP_DISPLAY_COLS if c in sweep_summary.columns]
    summary_display = sweep_summary[display_cols].copy()
    for col in _ROUND_COLS:
        if col in summary_display.columns:
            summary_display[col] = summary_display[col].round(3)
    sort_key = ["model", "feature_set", "symmetry", "train_size"]
    summary_sorted = summary_display.sort_values(sort_key)
    return summary_sorted.assign(
        cfg=lambda d: d["feature_set"].astype(str) + " | " + d["symmetry"].astype(str)
    )


def plot_classical_sample_efficiency_curves(
    summary_sorted: pd.DataFrame,
    *,
    metric_specs: Sequence[tuple[str, str]] | None = None,
    cfg_colors: Mapping[str, str] | None = None,
    full_train_n: int = 215,
    train_size_cat_margin: float = 0.35,
    figsize_row_height: float = 5.0,
    figsize_col_width: float = 3.65,
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    tight_layout_rect: tuple[float, float, float, float] = (0, 0, 0.82, 1),
) -> tuple[object, np.ndarray]:
    """Figure 5 style: metrics × models grid with mean line and CI band."""
    import matplotlib.pyplot as plt

    if metric_specs is None:
        metric_specs = [
            ("balanced_accuracy", "Balanced accuracy (OOD)"),
            ("win_rate", "Win rate vs random"),
        ]
    if cfg_colors is None:
        cfg_colors = CLASSICAL_CFG_LEARNING_CURVE_COLORS

    models = list(summary_sorted["model"].unique())
    train_size_order = sorted(summary_sorted["train_size"].unique())
    cat_x = np.arange(len(train_size_order), dtype=float)
    train_size_to_x = {ts: cat_x[i] for i, ts in enumerate(train_size_order)}
    xtick_labels = [
        "full" if float(ts) == float(full_train_n) else str(int(ts)) for ts in train_size_order
    ]

    n_models = len(models)
    n_metrics = len(metric_specs)
    fig, axes = plt.subplots(
        n_metrics,
        n_models,
        figsize=(max(11.0, figsize_col_width * n_models), figsize_row_height),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(n_metrics, n_models)

    for j, (mkey, mlabel) in enumerate(metric_specs):
        for k, model in enumerate(models):
            ax = axes[j, k]
            sub = summary_sorted[summary_sorted["model"] == model]
            mean_c = f"{mkey}_mean"
            lo_c, hi_c = f"{mkey}_ci_low", f"{mkey}_ci_high"
            std_c = f"{mkey}_std"
            for cfg, g in sub.groupby("cfg", sort=False):
                g2 = g.sort_values("train_size")
                x = np.array([train_size_to_x[ts] for ts in g2["train_size"]], dtype=float)
                y = g2[mean_c].to_numpy()
                if lo_c in g2.columns and hi_c in g2.columns:
                    lo = g2[lo_c].to_numpy()
                    hi = g2[hi_c].to_numpy()
                else:
                    e = g2[std_c].fillna(0).to_numpy()
                    lo, hi = np.clip(y - e, 0, 1), np.clip(y + e, 0, 1)
                c = cfg_colors.get(str(cfg), "#888888")
                ax.plot(x, y, marker="o", label=cfg, alpha=0.9, color=c)
                ax.fill_between(
                    x,
                    np.clip(lo, 0, 1),
                    np.clip(hi, 0, 1),
                    color=c,
                    alpha=0.14,
                )
            if k == 0:
                ax.set_ylabel(mlabel)
            if j == 0:
                ax.set_title(model)
            ax.grid(alpha=0.25)
            ax.set_ylim(0.0, 1.05)
            if j == n_metrics - 1:
                ax.set_xlabel("Train size")
    for ax in axes.flat:
        ax.set_xticks(cat_x)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(cat_x[0] - train_size_cat_margin, cat_x[-1] + train_size_cat_margin)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=legend_bbox_to_anchor, fontsize=7)
    plt.tight_layout(rect=tight_layout_rect)
    return fig, axes


def format_kernel_aligned_baseline_display(
    kernel_baseline_summary: pd.DataFrame,
    *,
    round_metrics: int = 3,
) -> pd.DataFrame:
    """Rounded tidy table for display (kernel-aligned classical sweep)."""
    kernel_baseline_display_cols = [
        "model",
        "train_size",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "win_rate_mean",
        "win_rate_std",
        "n_runs",
    ]
    cols = [c for c in kernel_baseline_display_cols if c in kernel_baseline_summary.columns]
    out = kernel_baseline_summary[cols].copy()
    for col in out.columns[2:]:
        if col != "n_runs":
            out[col] = out[col].round(round_metrics)
    return out


def plot_kernel_aligned_baseline_curves(
    kernel_baseline_summary: pd.DataFrame,
    *,
    metric_specs: Sequence[tuple[str, str]] | None = None,
    model_colors: Mapping[str, str] | None = None,
    full_train_n: int = 215,
    figsize: tuple[float, float] = (10.0, 4.0),
    suptitle: str = "Kernel-aligned classical baseline — raw heaps, symmetry=none",
) -> tuple[object, np.ndarray]:
    """Figure 8 style: RBF vs angle kernel on raw heaps (two metrics, shared y)."""
    import matplotlib.pyplot as plt

    if metric_specs is None:
        metric_specs = [
            ("balanced_accuracy", "Balanced accuracy (OOD)"),
            ("win_rate", "Win rate vs random"),
        ]
    if model_colors is None:
        model_colors = KERNEL_ALIGNED_MODEL_COLORS

    kernel_train_sizes = sorted(kernel_baseline_summary["train_size"].unique())
    kernel_x = np.arange(len(kernel_train_sizes), dtype=float)
    kernel_xticks = [
        "full" if float(ts) == float(full_train_n) else str(int(ts)) for ts in kernel_train_sizes
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (mkey, mlabel) in zip(axes, metric_specs):
        mean_c = f"{mkey}_mean"
        lo_c, hi_c = f"{mkey}_ci_low", f"{mkey}_ci_high"
        std_c = f"{mkey}_std"
        for model, grp in kernel_baseline_summary.groupby("model", sort=False):
            grp = grp.sort_values("train_size")
            x = np.array([kernel_train_sizes.index(ts) for ts in grp["train_size"]], dtype=float)
            y = grp[mean_c].to_numpy()
            if lo_c in grp.columns and hi_c in grp.columns:
                lo = grp[lo_c].to_numpy()
                hi = grp[hi_c].to_numpy()
            else:
                e = grp[std_c].fillna(0).to_numpy()
                lo, hi = np.clip(y - e, 0, 1), np.clip(y + e, 0, 1)
            color = model_colors.get(str(model), "#888888")
            ax.plot(x, y, marker="o", color=color, label=str(model))
            ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=color, alpha=0.15)
        ax.set_xticks(kernel_x)
        ax.set_xticklabels(kernel_xticks)
        ax.set_xlabel("Train size")
        ax.set_ylabel(mlabel)
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(suptitle)
    plt.tight_layout()
    return fig, axes


def plot_parity_feature_ablation_balanced_accuracy(
    display_df: pd.DataFrame,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] = (9.0, 4.0),
    full_train_sort_key: int = 215,
) -> tuple[object, object]:
    """Grouped bar chart: balanced accuracy ± std by feature set and train size."""
    import matplotlib.pyplot as plt

    feat_order = display_df["feature_set"].unique().tolist()
    train_order = sorted(
        display_df["train_size"].unique(),
        key=lambda ts: full_train_sort_key if ts == "full" else int(ts),
    )
    idx = pd.MultiIndex.from_product(
        [train_order, feat_order], names=["train_size", "feature_set"]
    )
    wide = display_df.set_index(["train_size", "feature_set"]).reindex(idx)
    y_mat = wide["balanced_accuracy_mean"].to_numpy().reshape(
        len(train_order), len(feat_order)
    )
    err_mat = (
        wide["balanced_accuracy_std"]
        .where(wide["balanced_accuracy_mean"].notna(), 0)
        .to_numpy()
        .reshape(len(train_order), len(feat_order))
    )

    x = np.arange(len(feat_order))
    n_sizes = len(train_order)
    bar_w = min(0.22, 0.8 / max(n_sizes, 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, ts_key in enumerate(train_order):
        offset = (i - (n_sizes - 1) / 2.0) * bar_w
        ax.bar(
            x + offset,
            y_mat[i],
            width=bar_w,
            yerr=err_mat[i],
            capsize=2,
            label=str(ts_key),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(feat_order, rotation=20, ha="right")
    ax.set_ylabel("Balanced accuracy (OOD)")
    ax.set_title("Feature ablation — SVM (RBF), no symmetry augmentation")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Train size", fontsize=8)
    return fig, ax


__all__ = [
    "CLASSICAL_CFG_LEARNING_CURVE_COLORS",
    "KERNEL_ALIGNED_MODEL_COLORS",
    "format_classical_sweep_summary_display",
    "format_kernel_aligned_baseline_display",
    "plot_classical_sample_efficiency_curves",
    "plot_kernel_aligned_baseline_curves",
    "plot_parity_feature_ablation_balanced_accuracy",
]
