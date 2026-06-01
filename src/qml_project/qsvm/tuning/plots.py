"""Section 6 QSVM plots: faceted metric / timing curves and full-train bars."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _use_logarithmic_y_axis(values: np.ndarray) -> bool:
    """Heuristic: pick a log y-axis when the positive max/min ratio exceeds 50."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if array.size == 0:
        return False
    vmax = float(np.max(array))
    vmin = float(np.maximum(np.min(array), 1e-9))
    return (vmax / vmin) > 50.0


def plot_qsvm_faceted_metric_curves(
    summary: pd.DataFrame,
    *,
    metric_mean_column: str,
    metric_std_column: str,
    y_axis_label: str,
    figure_suptitle: str,
    y_limits: tuple[float, float] | None,
    chance_line_y: float | None,
) -> None:
    """Rows = ``include_nim_sum``, columns = angle / amplitude / binary; lines = ``variant_id``.

    All variants in *summary* share the same train-size tick positions per panel
    (union of sizes present in that panel) so curves are not drawn on misaligned
    $x$ indices, which previously could stack two ``variant_id`` lines on top of
    each other.
    """
    if summary.empty:
        return
    encoding_order = ["angle", "amplitude", "binary"]
    nim_present = {bool(x) for x in summary["include_nim_sum"].dropna().unique()}
    nim_order = [flag for flag in (False, True) if flag in nim_present]
    if not nim_order:
        nim_order = [True]
    n_rows = max(1, len(nim_order))
    n_cols = len(encoding_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False,
    )
    variant_order = sorted(summary["variant_id"].dropna().unique())
    linestyles = ("-", "--", ":", "-.")
    for row_index, nim_value in enumerate(nim_order):
        for col_index, encoding_name in enumerate(encoding_order):
            ax = axes[row_index][col_index]
            panel = summary.loc[
                (summary["encoding"].astype(str) == encoding_name)
                & (summary["include_nim_sum"].astype(bool) == bool(nim_value))
            ]
            train_sizes_union = sorted(panel["train_size"].dropna().unique().tolist())
            if not train_sizes_union:
                ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{encoding_name} — {'Nim-sum on' if bool(nim_value) else 'Nim-sum off'}")
                continue
            x_positions = np.arange(len(train_sizes_union))

            def _row_for_size(group: pd.DataFrame, size_key: object) -> pd.Series | None:
                hit = group.loc[group["train_size"] == size_key]
                return hit.iloc[0] if len(hit) else None

            for variant_index, variant_label in enumerate(variant_order):
                group = panel.loc[panel["variant_id"] == variant_label].sort_values("train_size")
                if group.empty:
                    continue
                y_mean_list: list[float] = []
                y_std_list: list[float] = []
                for size_key in train_sizes_union:
                    row = _row_for_size(group, size_key)
                    if row is None:
                        y_mean_list.append(float("nan"))
                        y_std_list.append(0.0)
                    else:
                        y_mean_list.append(float(row[metric_mean_column]))
                        y_std_list.append(
                            float(row[metric_std_column])
                            if pd.notna(row[metric_std_column])
                            else 0.0
                        )
                y_mean = np.asarray(y_mean_list, dtype=float)
                y_std = np.asarray(y_std_list, dtype=float)
                if not np.any(np.isfinite(y_mean)):
                    continue
                color = f"C{variant_index % 10}"
                linestyle = linestyles[variant_index % len(linestyles)]
                ax.plot(
                    x_positions,
                    y_mean,
                    marker="o",
                    linestyle=linestyle,
                    color=color,
                    label=str(variant_label),
                )
                if y_limits is not None:
                    lo = np.maximum(y_limits[0], y_mean - y_std)
                    hi = np.minimum(y_limits[1], y_mean + y_std)
                else:
                    lo, hi = y_mean - y_std, y_mean + y_std
                ax.fill_between(
                    x_positions,
                    lo,
                    hi,
                    alpha=0.12,
                    color=color,
                    linestyle=linestyle,
                )
            nim_label = "Nim-sum on" if bool(nim_value) else "Nim-sum off"
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(s) for s in train_sizes_union])
            ax.set_title(f"{encoding_name} — {nim_label}")
            ax.set_xlabel("Train size")
            ax.grid(alpha=0.25)
            if chance_line_y is not None:
                ax.axhline(chance_line_y, color="grey", linestyle=":", alpha=0.5)
            if y_limits is not None:
                ax.set_ylim(y_limits)
            ax.legend(fontsize=6, loc="best")
    axes[0][0].set_ylabel(y_axis_label)
    fig.suptitle(figure_suptitle, y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_qsvm_train_and_kernel_time_curves(summary: pd.DataFrame) -> None:
    """Faceted like ``plot_qsvm_faceted_metric_curves``: Nim-sum rows, encoding columns.

    Two bands (SVC train wall time, then train Gram-matrix build time) share the
    same layout. Lines are ``variant_id``; within each panel, train sizes use the
    union of ladder steps present so curves stay aligned. Shaded bands use
    mean ± std over seeds when ``*_std`` columns exist. Each panel uses a
    **log** *y*-scale when the max/min ratio of positive plotted means exceeds 50
    (same rule as the previous two-row figure).
    """
    if summary.empty:
        return
    required = {"variant_id", "train_size", "train_time_s_mean", "encoding"}
    if not required.issubset(summary.columns):
        return
    variant_order = sorted(summary["variant_id"].dropna().unique())
    if not variant_order:
        return

    encoding_order = ["angle", "amplitude", "binary"]
    if "include_nim_sum" in summary.columns:
        nim_present = {bool(x) for x in summary["include_nim_sum"].dropna().unique()}
        nim_order = [flag for flag in (False, True) if flag in nim_present]
        if not nim_order:
            nim_order = [True]
    else:
        nim_order = [False]

    train_std_col = "train_time_s_std" if "train_time_s_std" in summary.columns else None
    metric_bands: list[tuple[str, str | None, str, str]] = [
        ("train_time_s_mean", train_std_col, "Train time (s)", "o"),
    ]
    if "kernel_matrix_time_s_mean" in summary.columns:
        kernel_std_col = (
            "kernel_matrix_time_s_std" if "kernel_matrix_time_s_std" in summary.columns else None
        )
        metric_bands.append(
            ("kernel_matrix_time_s_mean", kernel_std_col, "Kernel matrix time (s)", "s")
        )

    n_nim = len(nim_order)
    n_bands = len(metric_bands)
    n_rows = n_bands * n_nim
    n_cols = len(encoding_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.2 * n_rows),
        sharex="col",
        squeeze=False,
    )
    linestyles = ("-", "--", ":", "-.")

    def _encoding_nim_mask(encoding_name: str, nim_value: bool) -> pd.Series:
        mask = summary["encoding"].astype(str) == encoding_name
        if "include_nim_sum" in summary.columns:
            mask = mask & (summary["include_nim_sum"].astype(bool) == bool(nim_value))
        return mask

    def _row_for_size(group: pd.DataFrame, size_key: object) -> pd.Series | None:
        hit = group.loc[group["train_size"] == size_key]
        return hit.iloc[0] if len(hit) else None

    for band_index, (mean_col, std_col, y_axis_label, marker) in enumerate(metric_bands):
        for nim_index, nim_value in enumerate(nim_order):
            row = band_index * n_nim + nim_index
            for col_index, encoding_name in enumerate(encoding_order):
                ax = axes[row][col_index]
                panel = summary.loc[_encoding_nim_mask(encoding_name, nim_value)]
                train_sizes_union = sorted(panel["train_size"].dropna().unique().tolist())
                if not train_sizes_union:
                    title = (
                        f"{encoding_name} — {'Nim-sum on' if bool(nim_value) else 'Nim-sum off'}"
                        if "include_nim_sum" in summary.columns
                        else encoding_name
                    )
                    ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    continue
                x_positions = np.arange(len(train_sizes_union))

                all_vals: list[float] = []
                for variant_label in variant_order:
                    group = panel.loc[panel["variant_id"] == variant_label]
                    if group.empty:
                        continue
                    for size_key in train_sizes_union:
                        row_ser = _row_for_size(group, size_key)
                        if row_ser is None:
                            continue
                        mean_raw = row_ser[mean_col]
                        mean_scalar = (
                            mean_raw.iloc[0] if isinstance(mean_raw, pd.Series) else mean_raw
                        )
                        v = float(mean_scalar)
                        if np.isfinite(v) and v > 0:
                            all_vals.append(v)
                use_log = bool(all_vals) and _use_logarithmic_y_axis(
                    np.array(all_vals, dtype=float)
                )
                if use_log:
                    ax.set_yscale("log")
                scale_note = "log y" if use_log else "linear y"
                if "include_nim_sum" in summary.columns:
                    nim_title = "Nim-sum on" if bool(nim_value) else "Nim-sum off"
                    title_core = f"{encoding_name} — {nim_title}"
                else:
                    title_core = encoding_name
                ax.set_title(f"{title_core} ({scale_note})")

                for variant_index, variant_label in enumerate(variant_order):
                    group = panel.loc[panel["variant_id"] == variant_label].sort_values("train_size")
                    if group.empty:
                        continue
                    y_mean_list: list[float] = []
                    y_std_list: list[float] = []
                    for size_key in train_sizes_union:
                        row_ser = _row_for_size(group, size_key)
                        if row_ser is None:
                            y_mean_list.append(float("nan"))
                            y_std_list.append(0.0)
                        else:
                            mean_raw = row_ser[mean_col]
                            mean_scalar = (
                                mean_raw.iloc[0]
                                if isinstance(mean_raw, pd.Series)
                                else mean_raw
                            )
                            y_mean_list.append(float(mean_scalar))
                            if std_col is not None and std_col in row_ser.index:
                                raw_std = row_ser[std_col]
                                std_scalar = (
                                    raw_std.iloc[0]
                                    if isinstance(raw_std, pd.Series)
                                    else raw_std
                                )
                                y_std_list.append(
                                    float(std_scalar) if pd.notna(std_scalar) else 0.0
                                )
                            else:
                                y_std_list.append(0.0)
                    y_mean = np.asarray(y_mean_list, dtype=float)
                    y_std = np.asarray(y_std_list, dtype=float)
                    if not np.any(np.isfinite(y_mean)):
                        continue
                    color = f"C{variant_index % 10}"
                    linestyle = linestyles[variant_index % len(linestyles)]
                    if use_log:
                        y_plot = np.where((y_mean > 0) & np.isfinite(y_mean), y_mean, np.nan)
                    else:
                        y_plot = y_mean
                    ax.plot(
                        x_positions,
                        y_plot,
                        marker=marker,
                        linestyle=linestyle,
                        color=color,
                        label=str(variant_label),
                    )
                    lo = y_mean - y_std
                    hi = y_mean + y_std
                    if use_log:
                        eps = 1e-15
                        lo_d = np.maximum(lo, eps)
                        hi_d = np.maximum(hi, eps)
                        valid = (y_mean > 0) & np.isfinite(y_mean)
                        ax.fill_between(
                            x_positions,
                            lo_d,
                            hi_d,
                            where=valid,
                            alpha=0.12,
                            color=color,
                            linestyle=linestyle,
                        )
                    else:
                        ax.fill_between(
                            x_positions,
                            lo,
                            hi,
                            alpha=0.12,
                            color=color,
                            linestyle=linestyle,
                        )

                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(s) for s in train_sizes_union])
                ax.grid(alpha=0.25)
                ax.legend(fontsize=6, loc="best")
                if col_index == 0:
                    ax.set_ylabel(y_axis_label)
                if row == n_rows - 1:
                    ax.set_xlabel("Train size")

    fig.suptitle("QSVM timing — SVC fit and Gram-matrix build", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_qsvm_full_train_balanced_accuracy_bars(
    summary: pd.DataFrame,
    *,
    train_size: int | None = None,
    encoding_order: tuple[str, ...] = ("angle", "amplitude", "binary"),
) -> None:
    """Two panels (Nim-sum off / on); $x$ = encoding; grouped bars = ``variant_id``.

    Rows are restricted to one ``train_size``: *train_size* if given, otherwise
    the largest value present in *summary* (full ladder step when included).
    """
    if summary.empty or "train_size" not in summary.columns:
        return
    ts = int(train_size) if train_size is not None else summary["train_size"].max()
    at_full_summary = summary.loc[summary["train_size"] == ts].copy()
    if at_full_summary.empty:
        return
    variants_sorted = sorted(at_full_summary["variant_id"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    bar_width = 0.8 / max(len(variants_sorted), 1)
    x_base = np.arange(len(encoding_order), dtype=float)
    for ax, nim_flag, panel_title in zip(
        axes,
        (False, True),
        ("Nim-sum off (heap-only where applicable)", "Nim-sum on"),
        strict=False,
    ):
        panel = at_full_summary.loc[at_full_summary["include_nim_sum"].astype(bool) == nim_flag]
        for variant_index, variant_label in enumerate(variants_sorted):
            offset = (variant_index - (len(variants_sorted) - 1) / 2.0) * bar_width
            heights: list[float] = []
            errors: list[float] = []
            for encoding_name in encoding_order:
                row = panel.loc[
                    (panel["encoding"].astype(str) == encoding_name)
                    & (panel["variant_id"].astype(str) == str(variant_label))
                ]
                if len(row):
                    heights.append(float(row["balanced_accuracy_mean"].iloc[0]))
                    errors.append(float(row["balanced_accuracy_std"].fillna(0).iloc[0]))
                else:
                    heights.append(float("nan"))
                    errors.append(0.0)
            ax.bar(
                x_base + offset,
                heights,
                width=bar_width * 0.92,
                yerr=errors,
                capsize=2,
                label=str(variant_label),
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels(list(encoding_order))
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(panel_title)
        ax.set_xlabel("Encoding")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel("Balanced accuracy (OOD, full train)")
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("QSVM — full training set", y=1.02)
    plt.tight_layout()
    plt.show()


__all__ = [
    "plot_qsvm_faceted_metric_curves",
    "plot_qsvm_full_train_balanced_accuracy_bars",
    "plot_qsvm_train_and_kernel_time_curves",
]
