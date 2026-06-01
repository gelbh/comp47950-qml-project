"""Per-state adversarial/disagreement analysis helpers for Nim experiments."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _nim_sum_array(states: np.ndarray) -> np.ndarray:
    """Return Nim-sum for each state row."""
    arr = np.asarray(states, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError("states must be a 2D array")
    if arr.shape[1] == 0:
        raise ValueError("states must include at least one heap")
    out = arr[:, 0].copy()
    for j in range(1, arr.shape[1]):
        out = np.bitwise_xor(out, arr[:, j])
    return out


def min_distance_to_training_states(
    test_states: np.ndarray,
    train_states: np.ndarray,
    *,
    metric: str = "l1",
    m_max: int = 7,
) -> np.ndarray:
    """Return minimum train-distance for each test state.

    Distances are computed in normalised heap space (divide by ``m_max``),
    with per-row min over the training set.
    """
    test_arr = np.asarray(test_states, dtype=np.float64)
    train_arr = np.asarray(train_states, dtype=np.float64)
    if test_arr.ndim != 2 or train_arr.ndim != 2:
        raise ValueError("test_states and train_states must be 2D arrays")
    if test_arr.shape[1] != train_arr.shape[1]:
        raise ValueError("test and train states must have same feature dimension")
    if test_arr.size == 0 or train_arr.size == 0:
        return np.array([], dtype=np.float64)

    test_norm = test_arr / float(m_max)
    train_norm = train_arr / float(m_max)
    diff = np.abs(test_norm[:, None, :] - train_norm[None, :, :])
    if metric == "l1":
        dist = diff.sum(axis=2)
    elif metric == "l2":
        dist = np.sqrt((diff**2).sum(axis=2))
    else:
        raise ValueError("metric must be 'l1' or 'l2'")
    return dist.min(axis=1)


def pairwise_disagreement_labels(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> np.ndarray:
    """Classify pairwise outcome by correctness/disagreement type."""
    yt = np.asarray(y_true, dtype=np.int32)
    pa = np.asarray(y_pred_a, dtype=np.int32)
    pb = np.asarray(y_pred_b, dtype=np.int32)
    if yt.shape != pa.shape or yt.shape != pb.shape:
        raise ValueError("y_true and predictions must have same shape")

    ca = pa == yt
    cb = pb == yt

    labels = np.full(yt.shape, "both_wrong_same_label", dtype=object)
    labels[np.logical_and(ca, cb)] = "both_correct"
    labels[np.logical_and(ca, np.logical_not(cb))] = f"{label_a}_only_correct"
    labels[np.logical_and(np.logical_not(ca), cb)] = f"{label_b}_only_correct"
    both_wrong = np.logical_and(np.logical_not(ca), np.logical_not(cb))
    diff_wrong = np.logical_and(both_wrong, pa != pb)
    labels[diff_wrong] = "both_wrong_different_label"
    return labels


def build_adversarial_analysis_dataframe(
    test_states: np.ndarray,
    y_true: np.ndarray,
    predictions_by_pipeline: Mapping[str, np.ndarray],
    *,
    train_states: np.ndarray,
    distance_metric: str = "l1",
    m_max: int = 7,
) -> pd.DataFrame:
    """Create per-state analysis table with correctness and OOD features."""
    states = np.asarray(test_states, dtype=np.int32)
    y = np.asarray(y_true, dtype=np.int32)
    if states.ndim != 2:
        raise ValueError("test_states must be 2D")
    if y.shape[0] != states.shape[0]:
        raise ValueError("y_true length must match test_states rows")

    data: dict[str, object] = {}
    for j in range(states.shape[1]):
        data[f"h_{j + 1}"] = states[:, j]
    data["true_label"] = y
    nim_sum = _nim_sum_array(states)
    data["nim_sum"] = nim_sum
    data["nim_sum_abs"] = np.abs(nim_sum)
    data["max_heap"] = states.max(axis=1)
    data["min_train_distance"] = min_distance_to_training_states(
        states,
        np.asarray(train_states, dtype=np.int32),
        metric=distance_metric,
        m_max=m_max,
    )

    for name, pred in predictions_by_pipeline.items():
        arr = np.asarray(pred, dtype=np.int32)
        if arr.shape[0] != states.shape[0]:
            raise ValueError(f"prediction length mismatch for pipeline '{name}'")
        data[f"pred_{name}"] = arr
        data[f"correct_{name}"] = arr == y

    return pd.DataFrame(data)


def add_pairwise_disagreement_column(
    df: pd.DataFrame,
    *,
    pipeline_a: str,
    pipeline_b: str,
    out_col: str | None = None,
) -> pd.DataFrame:
    """Add pairwise disagreement-category column for two pipelines."""
    pred_a = f"pred_{pipeline_a}"
    pred_b = f"pred_{pipeline_b}"
    if pred_a not in df.columns or pred_b not in df.columns or "true_label" not in df.columns:
        raise ValueError("required columns missing for disagreement classification")
    col = out_col or f"disagreement_{pipeline_a}_vs_{pipeline_b}"
    out = df.copy()
    out[col] = pairwise_disagreement_labels(
        out["true_label"].to_numpy(),
        out[pred_a].to_numpy(),
        out[pred_b].to_numpy(),
        label_a=pipeline_a,
        label_b=pipeline_b,
    )
    return out


def summarise_disagreement_by_nim_sum_band(
    df: pd.DataFrame,
    *,
    disagreement_col: str,
    nim_sum_abs_col: str = "nim_sum_abs",
) -> pd.DataFrame:
    """Summarise disagreement rate by Nim-sum magnitude bands."""
    if disagreement_col not in df.columns or nim_sum_abs_col not in df.columns:
        raise ValueError("required columns missing for nim-sum summary")
    out = df.copy()
    out["nim_sum_band"] = pd.cut(
        pd.to_numeric(out[nim_sum_abs_col], errors="coerce"),
        bins=[-0.1, 0.5, 1.5, 3.5, np.inf],
        labels=["0", "1", "2-3", "4+"],
        include_lowest=True,
    )
    agg = (
        out.groupby("nim_sum_band", dropna=False, observed=False)
        .agg(
            n=("true_label", "size"),
            disagreement_rate=(disagreement_col, lambda s: float((s != "both_correct").mean())),
            both_wrong_rate=(disagreement_col, lambda s: float(s.astype(str).str.startswith("both_wrong").mean())),
        )
        .reset_index()
    )
    return agg


def summarise_disagreement_by_distance_quantile(
    df: pd.DataFrame,
    *,
    disagreement_col: str,
    distance_col: str = "min_train_distance",
    q: int = 4,
) -> pd.DataFrame:
    """Summarise disagreement rate by quantile of train-distance."""
    if disagreement_col not in df.columns or distance_col not in df.columns:
        raise ValueError("required columns missing for distance summary")
    out = df.copy()
    out = out.dropna(subset=[distance_col])
    if out.empty:
        return pd.DataFrame(columns=["distance_bin", "n", "disagreement_rate", "both_wrong_rate"])
    out["distance_bin"] = pd.qcut(out[distance_col], q=q, duplicates="drop")
    agg = (
        out.groupby("distance_bin", dropna=False, observed=False)
        .agg(
            n=("true_label", "size"),
            disagreement_rate=(disagreement_col, lambda s: float((s != "both_correct").mean())),
            both_wrong_rate=(disagreement_col, lambda s: float(s.astype(str).str.startswith("both_wrong").mean())),
            mean_distance=(distance_col, "mean"),
        )
        .reset_index()
    )
    return agg


_DEFAULT_DISAGREEMENT_GEO_CATEGORIES: tuple[str, ...] = (
    "both_correct",
    "classical_only_correct",
    "qsvm_only_correct",
    "both_wrong_same_label",
    "both_wrong_different_label",
)

_DEFAULT_DISAGREEMENT_GEO_COLORS: tuple[str, ...] = (
    "#2a9d8f",
    "#1d3557",
    "#e76f51",
    "#6c757d",
    "#f4a261",
)


def plot_disagreement_geography_h3_slices(
    df: pd.DataFrame,
    disagreement_col: str,
    *,
    h3_slices: tuple[int, ...] = (0, 3, 7),
    heap_coord_max: int = 7,
    categories: tuple[str, ...] | None = None,
    category_colors: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (13.0, 3.8),
    suptitle: str = "Confusion geography (classical vs QSVM disagreement categories)",
) -> "Figure":
    """Heatmaps of disagreement category on (h_1, h_2) for fixed ``h_3`` slices.

    Nim-sum zero cells are overlaid with white crosses. Requires columns
    ``h_1``, ``h_2``, ``h_3``, ``nim_sum``, and ``disagreement_col``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if disagreement_col not in df.columns:
        raise ValueError(f"column {disagreement_col!r} not in dataframe")
    cats = categories or _DEFAULT_DISAGREEMENT_GEO_CATEGORIES
    colors = category_colors or _DEFAULT_DISAGREEMENT_GEO_COLORS
    if len(colors) != len(cats):
        raise ValueError("categories and category_colors must have the same length")

    code_map = {k: i for i, k in enumerate(cats)}
    cmap = ListedColormap(list(colors))
    n = heap_coord_max + 1
    fig, axes = plt.subplots(1, len(h3_slices), figsize=figsize, sharex=True, sharey=True)
    if len(h3_slices) == 1:
        axes = np.asarray([axes])
    im = None
    for ax, h3 in zip(axes, h3_slices):
        sub = df[df["h_3"] == h3]
        grid = np.full((n, n), np.nan, dtype=np.float64)
        for _, r in sub.iterrows():
            h1, h2 = int(r["h_1"]), int(r["h_2"])
            if 0 <= h1 <= heap_coord_max and 0 <= h2 <= heap_coord_max:
                cat = str(r[disagreement_col])
                if cat in code_map:
                    grid[h2, h1] = code_map[cat]

        im = ax.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=len(cats) - 1,
            cmap=cmap,
        )
        nim0 = sub[sub["nim_sum"] == 0]
        if not nim0.empty:
            ax.scatter(nim0["h_1"], nim0["h_2"], marker="x", color="white", s=28, label="nim_sum=0")
        ax.set_title(f"h_3 = {h3}")
        ax.set_xlabel("h_1")
        ax.set_ylabel("h_2")
        ax.set_xticks(range(0, n))
        ax.set_yticks(range(0, n))
        ax.grid(alpha=0.15)

    assert im is not None
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_ticks(np.arange(len(cats)))
    cbar.set_ticklabels(list(cats))
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_disagreement_rate_clustering_bars(
    nim_summary: pd.DataFrame,
    dist_summary: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (11.0, 3.6),
) -> "Figure":
    """Bar charts: disagreement rate vs Nim-sum band and vs train-distance quantile."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(nim_summary["nim_sum_band"].astype(str), nim_summary["disagreement_rate"], color="#457b9d")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Disagreement rate vs Nim-sum magnitude")
    axes[0].set_xlabel("|Nim-sum| band")
    axes[0].set_ylabel("Rate")
    axes[0].grid(alpha=0.25)

    x2 = dist_summary["distance_bin"].astype(str)
    axes[1].bar(x2, dist_summary["disagreement_rate"], color="#bc6c25")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Disagreement rate vs train-distance quantile")
    axes[1].set_xlabel("Distance quantile")
    axes[1].set_ylabel("Rate")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def build_adversarial_section_precondition_df(
    adv_df: pd.DataFrame,
    geo_col: str,
    available_pipeline_names: Collection[str] | Mapping[str, object],
    *,
    required_pipeline_names: Collection[str] | None = None,
    h3_slices: tuple[int, ...] = (0, 3, 7),
    distance_quantile_q: int = 4,
    nim_summary: pd.DataFrame | None = None,
    dist_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Submission-style checks for the adversarial / disagreement notebook section.

    When ``nim_summary`` / ``dist_summary`` are provided (e.g. from the same
    section that prints clustering tables), they are used for the clustering
    diagnostic check instead of recomputing aggregations.
    """
    if isinstance(available_pipeline_names, Mapping):
        available = set(available_pipeline_names.keys())
    else:
        available = set(available_pipeline_names)
    required = frozenset(required_pipeline_names or ("classical", "qsvm"))
    missing = sorted(required - available)

    if geo_col in adv_df.columns and not adv_df.empty:
        non_empty_slices = int(
            sum(adv_df[adv_df["h_3"] == h].shape[0] > 0 for h in h3_slices)
        )
    else:
        non_empty_slices = 0

    if geo_col in adv_df.columns:
        if nim_summary is not None:
            nim_ok = not nim_summary.empty
        else:
            nim_ok = not summarise_disagreement_by_nim_sum_band(
                adv_df, disagreement_col=geo_col
            ).empty
        if dist_summary is not None:
            dist_ok = not dist_summary.empty
        else:
            dist_ok = not summarise_disagreement_by_distance_quantile(
                adv_df, disagreement_col=geo_col, q=distance_quantile_q
            ).empty
    else:
        nim_ok = False
        dist_ok = False

    return pd.DataFrame(
        [
            {
                "check": "required_pipelines_available",
                "ok": not bool(missing),
                "detail": None if not missing else f"missing={missing}",
            },
            {
                "check": "confusion_geography_slices_h3_0_3_7",
                "ok": bool(non_empty_slices == len(h3_slices)),
                "detail": f"non_empty_slices={non_empty_slices}",
            },
            {
                "check": "clustering_diagnostics_nim_sum_distance",
                "ok": bool(nim_ok and dist_ok),
                "detail": None,
            },
            {
                "check": "optional_sim_vqc_comparator_enabled",
                "ok": bool("sim_vqc" in available),
                "detail": None,
            },
        ]
    )
