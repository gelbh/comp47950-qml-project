"""VQC tuning, VQC robustness, and architecture-diagnostics figures (§4–§5)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def plot_vqc_tuning_curves_by_encoding(
    vqc_summary: pd.DataFrame,
    *,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
    legend_fontsize: float = 7,
) -> tuple[Any, Any] | None:
    """Figure 5.1: mean ± std balanced accuracy vs train size; one panel per encoding."""
    import matplotlib.pyplot as plt

    required = {
        "encoding",
        "config_id",
        "train_size",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    }
    if vqc_summary.empty or not required.issubset(vqc_summary.columns):
        return None

    encodings_seen = sorted(vqc_summary["encoding"].astype(str).unique())
    if not encodings_seen:
        return None

    n = len(encodings_seen)
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, n, figsize=(w * n, h), sharey=True)
    ax_list = [axes] if n == 1 else list(np.ravel(np.asarray(axes)))

    for ax, encoding in zip(ax_list, encodings_seen):
        sub = vqc_summary.loc[vqc_summary["encoding"] == encoding]
        for cfg, grp in sub.groupby("config_id"):
            grp = grp.sort_values("train_size")
            x = np.arange(len(grp))
            y = grp["balanced_accuracy_mean"].to_numpy()
            e = grp["balanced_accuracy_std"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", label=str(cfg))
            ax.fill_between(x, np.maximum(0, y - e), np.minimum(1, y + e), alpha=0.15)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in grp["train_size"]])
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(f"VQC tuning — {encoding}")
        ax.set_xlabel("Train size")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=legend_fontsize)

    ax_list[0].set_ylabel("Balanced accuracy (OOD)")
    plt.tight_layout()
    return fig, axes


def plot_vqc_robustness_balanced_accuracy_vs_noise(
    vqc_robustness_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7.0, 4.0),
    legend_fontsize: float = 7,
) -> tuple[Any, Any] | None:
    """Figure 5.2: mean balanced accuracy vs depolarising noise at mid shot budget."""
    import matplotlib.pyplot as plt

    if vqc_robustness_df.empty or "noise_level" not in vqc_robustness_df.columns:
        return None

    metric_cols = (
        ("balanced_accuracy_raw", "raw"),
        ("balanced_accuracy_readout", "readout"),
        ("balanced_accuracy_zne", "zne"),
        ("balanced_accuracy_readout_zne", "readout+zne"),
    )
    mid_shots_vals = sorted(vqc_robustness_df["shots"].dropna().unique())
    if mid_shots_vals:
        mid = mid_shots_vals[len(mid_shots_vals) // 2]
        at_shots = vqc_robustness_df.loc[vqc_robustness_df["shots"] == mid].copy()
    else:
        at_shots = vqc_robustness_df
        mid = None

    plotted = False
    fig, ax = plt.subplots(figsize=figsize)
    for col, label in metric_cols:
        if col not in at_shots.columns:
            continue
        grp = (
            at_shots.groupby("noise_level")[col]
            .mean()
            .reset_index()
            .sort_values("noise_level")
        )
        ax.plot(grp["noise_level"], grp[col], marker="o", label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Depolarising noise level")
    ax.set_ylabel("Balanced accuracy (OOD)")
    ax.set_title(
        "VQC robustness — top config balanced accuracy vs noise"
        + (f" (shots={int(mid)})" if mid is not None else "")
    )
    ax.legend(fontsize=legend_fontsize)
    ax.grid(alpha=0.25)
    ax.set_ylim(0.0, 1.05)
    plt.tight_layout()
    return fig, ax


def plot_architecture_diagnostics_triptych(
    diag_df: pd.DataFrame,
    grad_df: pd.DataFrame,
    *,
    encodings_order: Sequence[str],
    ansatze: Sequence[str],
    figsize: tuple[float, float] = (15.0, 4.2),
    bar_width: float = 0.35,
    legend_fontsize_bars: float = 8,
    legend_fontsize_grad: float = 7,
) -> tuple[Any, Any] | None:
    """Figure 4.3: expressibility, entangling capability, and gradient-variance screen."""
    import matplotlib.pyplot as plt

    diag_need = {"encoding", "ansatz", "kl_to_haar", "mw_mean", "mw_std"}
    grad_need = {"encoding", "ansatz", "depth", "gradient_variance_mean"}
    if (
        diag_df.empty
        or grad_df.empty
        or not diag_need.issubset(diag_df.columns)
        or not grad_need.issubset(grad_df.columns)
        or not encodings_order
        or not ansatze
    ):
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    encodings_ord = [str(e) for e in encodings_order]
    x = np.arange(len(encodings_ord))
    width = bar_width

    for i, ansatz in enumerate(ansatze):
        sub = diag_df.loc[diag_df["ansatz"] == ansatz].set_index("encoding")
        axes[0].bar(
            x + (i - 0.5) * width,
            [float(sub.loc[e, "kl_to_haar"]) if e in sub.index else np.nan for e in encodings_ord],
            width=width,
            label=str(ansatz),
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(encodings_ord)
    axes[0].set_ylabel("KL(empirical || Haar)")
    axes[0].set_title("Expressibility (lower is better)")
    axes[0].legend(fontsize=legend_fontsize_bars)
    axes[0].grid(axis="y", alpha=0.25)

    for i, ansatz in enumerate(ansatze):
        sub = diag_df.loc[diag_df["ansatz"] == ansatz].set_index("encoding")
        axes[1].bar(
            x + (i - 0.5) * width,
            [float(sub.loc[e, "mw_mean"]) if e in sub.index else np.nan for e in encodings_ord],
            yerr=[float(sub.loc[e, "mw_std"]) if e in sub.index else 0.0 for e in encodings_ord],
            width=width,
            capsize=3,
            label=str(ansatz),
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(encodings_ord)
    axes[1].set_ylabel("Meyer–Wallach mean ± std")
    axes[1].set_title("Entangling capability")
    axes[1].legend(fontsize=legend_fontsize_bars)
    axes[1].grid(axis="y", alpha=0.25)

    for (encoding, ansatz), grp in grad_df.groupby(["encoding", "ansatz"]):
        grp = grp.sort_values("depth")
        axes[2].plot(
            grp["depth"],
            grp["gradient_variance_mean"],
            marker="o",
            label=f"{encoding}|{ansatz}",
        )
    axes[2].set_xlabel("Depth (n_layers)")
    axes[2].set_ylabel("Mean gradient variance")
    axes[2].set_yscale("log")
    axes[2].set_title("Barren-plateau screen")
    axes[2].legend(fontsize=legend_fontsize_grad, ncol=2)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    return fig, axes


__all__ = [
    "plot_architecture_diagnostics_triptych",
    "plot_vqc_robustness_balanced_accuracy_vs_noise",
    "plot_vqc_tuning_curves_by_encoding",
]
