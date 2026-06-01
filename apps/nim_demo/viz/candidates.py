"""Per-move score bars, class-probability stacks, and classical feature bars."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from qml_project.nim.game import NimMove

from ._constants import COLOR_CHOSEN, COLOR_NEUTRAL, COLOR_OPTIMAL
from .board import move_label


def candidate_score_bar(
    moves: Sequence[NimMove],
    score: np.ndarray,
    *,
    best_index: int,
    optimal_move: NimMove | None,
    score_label: str,
    title: str | None = None,
) -> go.Figure:
    """Bar chart of per-candidate scores with the chosen and optimal moves marked."""
    labels = [move_label(m) for m in moves]
    colors = []
    for i, m in enumerate(moves):
        if i == best_index:
            colors.append(COLOR_CHOSEN)
        elif optimal_move is not None and tuple(m) == tuple(optimal_move):
            colors.append(COLOR_OPTIMAL)
        else:
            colors.append(COLOR_NEUTRAL)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=score,
            marker_color=colors,
            text=[f"{v:.3f}" for v in score],
            textposition="outside",
        )
    )
    annotations = [
        dict(
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.15,
            text=(
                f"<span style='color:{COLOR_CHOSEN}'>\u25A0 chosen</span>"
                f"  <span style='color:{COLOR_OPTIMAL}'>\u25A0 Nim-sum optimum</span>"
            ),
            showarrow=False,
            font=dict(size=11),
            align="left",
        )
    ]
    layout_kwargs: dict = {
        "yaxis_title": score_label,
        "xaxis_title": "candidate move",
        "margin": dict(l=20, r=20, t=60, b=40),
        "height": 330,
        "plot_bgcolor": "white",
        "annotations": annotations,
    }
    if title is not None:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(tickangle=-30)
    return fig


def stacked_class_probs(
    moves: Sequence[NimMove],
    class_probs: np.ndarray,
    *,
    best_index: int,
    optimal_move: NimMove | None,
    class_names: Sequence[str] = ("losing (good for me)", "winning (bad for me)"),
) -> go.Figure:
    """Stacked bar of class probabilities per candidate, highlighting chosen."""
    labels = [move_label(m) for m in moves]
    n_classes = class_probs.shape[1]
    fig = go.Figure()
    palette = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"]
    for cls in range(n_classes):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=class_probs[:, cls],
                name=class_names[cls] if cls < len(class_names) else f"class {cls}",
                marker_color=palette[cls % len(palette)],
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[labels[best_index]],
            y=[1.02],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color=COLOR_CHOSEN),
            text=["chosen"],
            textposition="top center",
            showlegend=False,
        )
    )
    if optimal_move is not None:
        try:
            opt_idx = [tuple(m) for m in moves].index(tuple(optimal_move))
            fig.add_trace(
                go.Scatter(
                    x=[labels[opt_idx]],
                    y=[1.08],
                    mode="markers+text",
                    marker=dict(symbol="star", size=14, color=COLOR_OPTIMAL),
                    text=["optimum"],
                    textposition="top center",
                    showlegend=False,
                )
            )
        except ValueError:
            pass
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="probability", range=[0, 1.2]),
        xaxis_title="candidate move",
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, x=0),
        margin=dict(l=20, r=20, t=30, b=40),
        height=380,
        plot_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def classical_feature_bar(
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    best_index: int,
    moves: Sequence[NimMove],
) -> go.Figure:
    """Plot the feature vector for the chosen candidate move as a labelled bar."""
    values = features[best_index]
    labels = list(feature_names) if len(feature_names) == values.size else [
        f"f{i}" for i in range(values.size)
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#457b9d",
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Features used for chosen move {move_label(moves[best_index])}",
        xaxis_title="feature",
        yaxis_title="value",
        margin=dict(l=20, r=20, t=50, b=40),
        height=280,
    )
    fig.update_xaxes(tickangle=-35)
    return fig
