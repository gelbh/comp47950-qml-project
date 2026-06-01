"""Classical baseline charts, win/loss facets, PCA grids, and training-time bars."""

from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_WIN_LOSS_HEATMAP_SCALE: list[list[Any]] = [
    [0.0, "#d3d3d3"],
    [0.5, "#f4a261"],
    [1.0, "#264653"],
]


def nim_h1_h2_winloss_heatmaps(*, M: int = 7, k: int = 3) -> go.Figure:
    """2×4 facet heatmaps: $(h_1, h_2)$ plane for each fixed $h_3$ (Section 02 style)."""
    if k != 3:
        raise ValueError("facet layout is fixed for k=3 heaps")
    size = M + 1
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[f"h₃ = {h3}" for h3 in range(8)],
        vertical_spacing=0.14,
        horizontal_spacing=0.06,
    )
    for h3 in range(8):
        grid = np.zeros((size, size), dtype=np.int8)
        for h1 in range(size):
            for h2 in range(size):
                if (h1, h2, h3) == (0, 0, 0) and h3 == 0:
                    grid[h1, h2] = -1
                else:
                    grid[h1, h2] = 1 if (h1 ^ h2 ^ h3) != 0 else 0
        row = h3 // 4 + 1
        col = h3 % 4 + 1
        hm_kwargs: dict[str, Any] = dict(
            z=grid,
            x=list(range(size)),
            y=list(range(size)),
            colorscale=_WIN_LOSS_HEATMAP_SCALE,
            zmin=-1,
            zmax=1,
            hovertemplate="h₁=%{y}<br>h₂=%{x}<br>code %{z}<extra></extra>",
            showscale=h3 == 0,
        )
        if h3 == 0:
            hm_kwargs["colorbar"] = dict(
                title="code",
                tickvals=[-1, 0, 1],
                ticktext=["terminal", "losing", "winning"],
                len=0.4,
                y=0.55,
            )
        fig.add_trace(go.Heatmap(**hm_kwargs), row=row, col=col)
        fig.update_xaxes(title_text="h₂", row=row, col=col, dtick=1)
        fig.update_yaxes(title_text="h₁", row=row, col=col, dtick=1)
    fig.update_layout(
        title="Nim state space: win / loss in (h₁, h₂) for each h₃",
        height=520,
        margin=dict(l=30, r=30, t=80, b=20),
        plot_bgcolor="white",
    )
    return fig


def classical_train_pca_scatter_grid(
    z_raw: np.ndarray,
    z_parity: np.ndarray,
    z_bit: np.ndarray,
    y: np.ndarray,
    *,
    titles: Sequence[str] = (
        "Raw (normalised heaps)",
        "Parity (12 features)",
        "Bit parity (6 features)",
    ),
    highlight_raw: np.ndarray | None = None,
    highlight_parity: np.ndarray | None = None,
    highlight_bit: np.ndarray | None = None,
) -> go.Figure:
    """2D PCA scatter for OOD train points in three feature spaces (Section 03 style)."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=list(titles),
        horizontal_spacing=0.07,
    )
    yv = np.asarray(y).astype(int).reshape(-1)
    panels: list[tuple[np.ndarray, np.ndarray | None, int]] = [
        (z_raw, highlight_raw, 1),
        (z_parity, highlight_parity, 2),
        (z_bit, highlight_bit, 3),
    ]
    for Z, highlight, col in panels:
        Z = np.asarray(Z, dtype=np.float64)
        for cls, color, name in (
            (0, "#f4a261", "Losing"),
            (1, "#264653", "Winning"),
        ):
            mask = yv == cls
            fig.add_trace(
                go.Scatter(
                    x=Z[mask, 0],
                    y=Z[mask, 1],
                    mode="markers",
                    marker=dict(color=color, size=7, opacity=0.72, line=dict(width=0)),
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    hovertemplate="PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        if highlight is not None:
            pt = np.asarray(highlight, dtype=np.float64).reshape(2)
            fig.add_trace(
                go.Scatter(
                    x=[pt[0]],
                    y=[pt[1]],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=16,
                        color="#e63946",
                        line=dict(width=1, color="white"),
                    ),
                    name="Preview board",
                    legendgroup="board",
                    showlegend=col == 1,
                    hovertemplate="Preview board<extra></extra>",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="PC 1", row=1, col=col)
        fig.update_yaxes(title_text="PC 2", row=1, col=col)
    fig.update_layout(
        title=dict(
            text="OOD train set in 2D PCA — raw vs parity vs bit-parity features",
            y=0.995,
            x=0.5,
            xanchor="center",
            pad=dict(b=6),
        ),
        height=450,
        margin=dict(l=40, r=20, t=108, b=100),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(color="#212529", size=13),
        ),
    )
    # Subplot column titles from make_subplots sit too low by default; nudge up.
    fig.update_annotations(yshift=22)
    return fig


_CLASSICAL_MODEL_PALETTE: dict[str, str] = {
    "Logistic Regression": "#457b9d",
    "SVM (RBF)": "#e76f51",
    "Random Forest": "#2a9d8f",
    "SVM (Angle Kernel)": "#6a4c93",
}

_CLASSICAL_MODEL_ORDER: tuple[str, ...] = (
    "Logistic Regression",
    "SVM (RBF)",
    "Random Forest",
    "SVM (Angle Kernel)",
)


def _classical_ood_metric_df(
    classical_df: pd.DataFrame | None, *, metric: str
) -> pd.DataFrame | None:
    if classical_df is None or classical_df.empty or metric not in classical_df.columns:
        return None
    df = classical_df.copy()
    if "regime" in df.columns and (df["regime"] == "ood").any():
        df = cast(pd.DataFrame, df.loc[df["regime"] == "ood"])
    return df


def classical_baseline_models_bar(
    classical_df: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """One bar per sklearn model: mean metric over feature sets and sweep axes."""
    fig = go.Figure()
    df = _classical_ood_metric_df(classical_df, metric=metric)
    if df is None or "model" not in df.columns:
        fig.update_layout(
            title="Classical baselines — model comparison",
            annotations=[
                dict(
                    text="No classical sweep data found.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig
    grouped = df.groupby("model", as_index=False)[metric].mean()
    agg = grouped.assign(mean_metric=grouped[metric]).drop(columns=[metric])
    order_rank = {m: i for i, m in enumerate(_CLASSICAL_MODEL_ORDER)}
    agg["_sort"] = agg["model"].map(lambda m: order_rank.get(str(m), 999))
    agg = agg.sort_values("_sort").drop(columns="_sort")
    fig.add_trace(
        go.Bar(
            x=agg["model"],
            y=agg["mean_metric"],
            marker_color=[
                _CLASSICAL_MODEL_PALETTE.get(str(m), "#6c757d") for m in agg["model"]
            ],
            text=[f"{v:.2f}" for v in agg["mean_metric"]],
            textposition="outside",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Classical baselines — mean {metric.replace('_', ' ')} (avg. over feature sets)",
        yaxis=dict(range=[0, 1.05], title=metric.replace("_", " ")),
        xaxis_title="classifier",
        margin=dict(l=40, r=20, t=50, b=80),
        height=300,
        plot_bgcolor="white",
    )
    return fig


def classical_feature_ablation_bar(
    classical_df: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Grouped bars: x = feature set, one series per model (ablation over encoding)."""
    fig = go.Figure()
    df = _classical_ood_metric_df(classical_df, metric=metric)
    if df is None or "model" not in df.columns or "feature_set" not in df.columns:
        fig.update_layout(
            title="Feature-set ablation",
            annotations=[
                dict(
                    text="No classical sweep data found.",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig
    grouped = df.groupby(["model", "feature_set"], as_index=False)[metric].mean()
    agg = grouped.assign(mean_metric=grouped[metric]).drop(columns=[metric])
    model_order = [m for m in _CLASSICAL_MODEL_ORDER if m in set(agg["model"])]
    model_order += sorted(str(m) for m in set(agg["model"]) - set(model_order))
    for model in model_order:
        g = cast(pd.DataFrame, agg.loc[agg["model"] == model].sort_values(by="feature_set"))
        if g.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=g["feature_set"],
                y=g["mean_metric"],
                name=str(model),
                marker_color=_CLASSICAL_MODEL_PALETTE.get(str(model), "#6c757d"),
                text=[f"{v:.2f}" for v in g["mean_metric"]],
                textposition="outside",
            )
        )
    fig.update_layout(
        title=f"Feature-set ablation — mean {metric.replace('_', ' ')}",
        yaxis=dict(range=[0, 1.05], title=metric.replace("_", " ")),
        xaxis_title="feature set",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
        plot_bgcolor="white",
    )
    return fig


def classical_sweep_bar(
    classical_df: pd.DataFrame,
    *,
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """Deprecated alias for ``classical_feature_ablation_bar``."""
    return classical_feature_ablation_bar(classical_df, metric=metric)


def _workflow_mean_training_time_by_size(df: pd.DataFrame) -> pd.Series | None:
    # Prefer canonical name from add_cost_metric_contract_columns (notebook Section 7.2).
    time_col = next(
        (c for c in ("training_time_s", "train_time_s", "training_time") if c in df.columns),
        None,
    )
    if time_col is None or "train_size" not in df.columns:
        return None
    g = df.groupby("train_size", as_index=False)[time_col].mean()
    return g.set_index("train_size")[time_col].rename("mean_time_s")


def _classical_mean_train_time_by_size(df: pd.DataFrame) -> pd.Series | None:
    """Mean ``train_time_s`` by ``train_size`` for OOD rows when ``regime`` exists."""
    if df.empty or "train_time_s" not in df.columns or "train_size" not in df.columns:
        return None
    sub = df
    if "regime" in sub.columns and (sub["regime"] == "ood").any():
        sub = sub[sub["regime"] == "ood"]
    if sub.empty:
        return None
    g = sub.groupby("train_size", as_index=False)["train_time_s"].mean()
    return g.set_index("train_size")["train_time_s"].rename("mean_time_s")


def _classical_winner_config_keys(df: pd.DataFrame) -> dict[str, Any] | None:
    """Pick ``model`` / ``feature_set`` / ``symmetry`` with highest mean OOD BA."""
    if df.empty or "balanced_accuracy" not in df.columns:
        return None
    sub = df
    if "regime" in sub.columns and (sub["regime"] == "ood").any():
        sub = sub[sub["regime"] == "ood"]
    if sub.empty:
        return None
    gcols = [c for c in ("model", "feature_set", "symmetry") if c in sub.columns]
    if not gcols:
        return None
    means = sub.groupby(gcols, as_index=False)["balanced_accuracy"].mean()
    if means.empty:
        return None
    best = means.loc[means["balanced_accuracy"].idxmax()]
    return {c: best[c] for c in gcols}


def _classical_winner_mean_train_time_by_size(df: pd.DataFrame) -> pd.Series | None:
    """Mean ``train_time_s`` by ``train_size`` for the classical best-BA config only."""
    keys = _classical_winner_config_keys(df)
    if keys is None:
        return None
    sub = df
    if "regime" in sub.columns and (sub["regime"] == "ood").any():
        sub = sub[sub["regime"] == "ood"]
    mask = pd.Series(True, index=sub.index)
    for col, val in keys.items():
        if col not in sub.columns:
            return None
        if pd.isna(val):
            mask &= sub[col].isna()
        else:
            mask &= sub[col].astype(str) == str(val)
    sub = sub.loc[mask]
    if sub.empty or "train_time_s" not in sub.columns or "train_size" not in sub.columns:
        return None
    g = sub.groupby("train_size", as_index=False)["train_time_s"].mean()
    return g.set_index("train_size")["train_time_s"].rename("mean_time_s")


def vqc_qsvm_training_time_grouped_bar(
    vqc_df: pd.DataFrame | None,
    qsvm_df: pd.DataFrame | None,
    classical_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Grouped bars: mean training time vs train size for three pipelines.

    **Quantum:** pass the Section 07 ``quantum_winner_rows_*`` slice per pipeline
    when available (one config, all train sizes); otherwise pass the full
    workflow frame (this function then averages over all configs/seeds).

    **Classical:** prefers the single sweep configuration with highest mean
    OOD ``balanced_accuracy`` (grouped by model / feature_set / symmetry when
    present), then averages ``train_time_s`` over seeds at each train size;
    falls back to averaging over the whole OOD sweep if a winner cannot be
    resolved.

    Uses a logarithmic y-axis so very different scales stay on one chart.
    """
    fig = go.Figure()
    vqc_s = (
        _workflow_mean_training_time_by_size(vqc_df)
        if vqc_df is not None and not vqc_df.empty
        else None
    )
    qsvm_s = (
        _workflow_mean_training_time_by_size(qsvm_df)
        if qsvm_df is not None and not qsvm_df.empty
        else None
    )
    classical_s: pd.Series | None = None
    if classical_df is not None and not classical_df.empty:
        classical_s = _classical_winner_mean_train_time_by_size(classical_df)
        if classical_s is None:
            classical_s = _classical_mean_train_time_by_size(classical_df)
    if vqc_s is None and qsvm_s is None and classical_s is None:
        fig.update_layout(
            title="Mean training time vs train size",
            annotations=[
                dict(
                    text="No cached timings (workflow / winner-row parquets, "
                    "or classical_df with train_time_s).",
                    showarrow=False,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                )
            ],
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            plot_bgcolor="white",
        )
        return fig

    parts: list[pd.Series] = []
    if vqc_s is not None:
        parts.append(vqc_s.rename("VQC"))
    if qsvm_s is not None:
        parts.append(qsvm_s.rename("QSVM"))
    if classical_s is not None:
        parts.append(classical_s.rename("Classical"))
    wide = pd.concat(parts, axis=1).sort_index()
    xs = [str(i) for i in wide.index]
    colors = {"VQC": "#264653", "QSVM": "#2a9d8f", "Classical": "#e9c46a"}
    which = ", ".join(str(c) for c in wide.columns)
    for col in wide.columns:
        yvals = wide[col]
        y_plot = yvals.mask((yvals.isna()) | (yvals <= 0))
        fig.add_trace(
            go.Bar(
                name=str(col),
                x=xs,
                y=y_plot,
                marker_color=colors.get(str(col), "#6c757d"),
                text=[f"{float(v):.1f}s" if pd.notna(v) else "" for v in yvals],
                textposition="outside",
            )
        )
    fig.update_layout(
        title=f"Mean training time vs train size — {which} (log y)",
        barmode="group",
        xaxis_title="train size",
        yaxis_title="mean training time (s, log scale)",
        height=360,
        margin=dict(l=40, r=20, t=52, b=88),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#dee2e6",
            borderwidth=1,
            font=dict(color="#212529", size=12),
        ),
    )
    fig.update_yaxes(type="log", exponentformat="power", showexponent="all")
    return fig
