"""QSVM kernel heatmaps and support-vector contribution bars."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go

from qml_project.nim.game import NimMove

from ._constants import (
    COLOR_OPTIMAL,
    KERNEL_HEATMAP_AXIS_TICKS_MAX_N,
    KERNEL_HEATMAP_RICH_HOVER_MAX_N,
)
from .board import move_label


def kernel_row_heatmap(
    kernel_row: np.ndarray,
    support_mask: np.ndarray,
    moves: Sequence[NimMove],
) -> go.Figure:
    """Heatmap of kernel similarity (candidate, training point).

    Support vectors are indicated by a small marker above the column.
    """
    move_labels = [move_label(m) for m in moves]
    train_labels = [str(i) for i in range(kernel_row.shape[1])]
    fig = go.Figure(
        go.Heatmap(
            z=kernel_row,
            x=train_labels,
            y=move_labels,
            colorscale="Viridis",
            colorbar=dict(title="k(cand, train)"),
            zmin=0.0,
            zmax=1.0,
        )
    )
    sv_cols = np.flatnonzero(np.asarray(support_mask, dtype=bool))
    if sv_cols.size:
        fig.add_trace(
            go.Scatter(
                x=[str(i) for i in sv_cols],
                y=[move_labels[0]] * sv_cols.size,
                mode="markers",
                marker=dict(symbol="triangle-down", size=8, color=COLOR_OPTIMAL),
                name="support vector",
                yaxis="y2",
                hoverinfo="text",
                text=[f"SV #{i}" for i in sv_cols],
            )
        )
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                showticklabels=False,
                range=[-0.5, 0.5],
            )
        )
    fig.update_layout(
        title="Kernel row |\u27E8\u03C8(cand)|\u03C8(train)\u27E9|\u00B2",
        xaxis_title="training state index",
        yaxis_title="candidate move",
        margin=dict(l=20, r=40, t=50, b=30),
        height=max(220, 40 + 28 * len(moves)),
    )
    return fig


def sv_contributions_bar(
    kernel_row_for_move: np.ndarray,
    support_mask: np.ndarray,
    dual_coef: np.ndarray,
    intercept: float,
    *,
    support_vectors_raw: np.ndarray,
) -> go.Figure:
    """Per-support-vector contribution ``dual_coef_i * k(cand, sv_i)``.

    Sums (with the intercept) to the decision function value.
    """
    sv_idx = np.flatnonzero(np.asarray(support_mask, dtype=bool))
    if sv_idx.size == 0:
        return go.Figure()
    k_vals = kernel_row_for_move[sv_idx]
    contributions = dual_coef * k_vals
    labels = [
        f"sv #{i} {tuple(int(v) for v in support_vectors_raw[row])}"
        for row, i in enumerate(sv_idx)
    ]
    colors = ["#2a9d8f" if c < 0 else "#e76f51" for c in contributions]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=contributions,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in contributions],
            textposition="outside",
        )
    )
    total = float(np.sum(contributions) + intercept)
    fig.update_layout(
        title=(
            "Per-SV contribution to decision function "
            f"(sum + b = {total:+.3f}; b = {intercept:+.3f})"
        ),
        xaxis_title="support vector",
        yaxis_title="dual_coef \u00D7 k(cand, sv)",
        margin=dict(l=20, r=20, t=50, b=40),
        height=320,
    )
    fig.update_xaxes(tickangle=-35)
    fig.add_hline(y=0.0, line_color="#adb5bd")
    return fig


def _nim_sum_int(h: np.ndarray) -> int:
    return int(h[0]) ^ int(h[1]) ^ int(h[2])


def _kernel_cell_hover_html(
    i: int,
    j: int,
    heap_rows: np.ndarray,
    kval: float,
) -> str:
    """Rich hover for QSVM kernel: heaps, Nim-sum, win/loss for player to move."""
    hi = heap_rows[i]
    hj = heap_rows[j]
    ni, nj = _nim_sum_int(hi), _nim_sum_int(hj)
    wi = "W to move" if ni else "L to move"
    wj = "W to move" if nj else "L to move"
    return (
        f"<b>Row #{i}</b> heaps ({hi[0]},{hi[1]},{hi[2]}) nim={ni} ({wi})<br>"
        f"<b>Col #{j}</b> heaps ({hj[0]},{hj[1]},{hj[2]}) nim={nj} ({wj})<br>"
        f"<b>k(x,x')</b> = {kval:.4f}"
    )


def kernel_matrix_heatmap(
    K: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = None,
    heap_rows: np.ndarray | None = None,
) -> go.Figure:
    """Full N×N kernel-similarity heatmap for the QSVM Learn page.

    When *heap_rows* is passed with shape ``(n, 3)`` (heap triples in row order),
    small ``n`` uses ``#i`` ticks plus rich hovers (heaps, Nim-sum, W/L, ``k``).
    Large ``n`` hides tick labels and drops custom hovers (Plotly shows indices
    and ``z`` only) to keep the figure responsive.
    """
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("kernel matrix must be square")
    n = K.shape[0]
    hovertext: list[list[str]] | None = None
    show_ticklabels = True
    tickvals: list[int] | None = None
    ticktext: list[str] | None = None
    if heap_rows is not None:
        hr = np.asarray(heap_rows, dtype=np.int32)
        if hr.shape != (n, 3):
            raise ValueError("heap_rows must have shape (n, 3) matching kernel size")
        ticks_x = list(range(n))
        ticks_y = list(range(n))
        if n <= KERNEL_HEATMAP_AXIS_TICKS_MAX_N:
            tickvals = list(range(n))
            ticktext = [f"#{i}" for i in range(n)]
        else:
            show_ticklabels = False
        if n <= KERNEL_HEATMAP_RICH_HOVER_MAX_N:
            hovertext = [
                [_kernel_cell_hover_html(i, j, hr, float(K[i, j])) for j in range(n)]
                for i in range(n)
            ]
    else:
        ticks_x = list(labels) if labels is not None else [str(i) for i in range(n)]
        ticks_y = ticks_x

    hm_kw: dict[str, Any] = dict(
        z=K,
        x=ticks_x,
        y=ticks_y,
        colorscale="Viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="k(x, x')"),
    )
    if hovertext is not None:
        hm_kw["hovertext"] = hovertext
        hm_kw["hovertemplate"] = "%{hovertext}<extra></extra>"

    fig = go.Figure(go.Heatmap(**hm_kw))
    axis_title = (
        "sample index (#row / #col)" if heap_rows is not None else "training state"
    )
    # Slightly taller for large N so cells stay a few pixels wide in the browser.
    height = int(min(920, max(400, 28.0 * float(np.sqrt(n)))))
    xaxis: dict[str, Any] = dict(title=axis_title, tickangle=0)
    yaxis: dict[str, Any] = dict(title=axis_title, autorange="reversed")
    if heap_rows is not None:
        xaxis["showticklabels"] = show_ticklabels
        yaxis["showticklabels"] = show_ticklabels
        if tickvals is not None and ticktext is not None:
            xaxis["tickmode"] = "array"
            xaxis["tickvals"] = tickvals
            xaxis["ticktext"] = ticktext
            yaxis["tickmode"] = "array"
            yaxis["tickvals"] = tickvals
            yaxis["ticktext"] = ticktext
    fig.update_layout(
        title=title,
        xaxis=xaxis,
        yaxis=yaxis,
        margin=dict(l=60, r=20, t=40 if title else 10, b=50),
        height=height,
        plot_bgcolor="white",
    )
    return fig
