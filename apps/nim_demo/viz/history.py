"""Agreement-with-optimal history and on-device accuracy bars."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def agreement_history_figure(history_df: pd.DataFrame) -> go.Figure:
    """Plot running agreement-with-optimal per pipeline as a line chart."""
    fig = go.Figure()
    if history_df.empty:
        fig.update_layout(
            title="Agreement with Nim-sum optimum (history)",
            xaxis_title="turn number",
            yaxis_title="running agreement rate",
            margin=dict(l=20, r=20, t=50, b=40),
            height=320,
        )
        return fig
    pipeline_palette = {"VQC": "#264653", "QSVM": "#2a9d8f", "Classical": "#e9c46a"}
    fallback_cycle = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    fallback_idx = 0
    for pipeline, g in history_df.groupby("pipeline"):
        color = pipeline_palette.get(pipeline)
        if color is None:
            color = fallback_cycle[fallback_idx % len(fallback_cycle)]
            fallback_idx += 1
        fig.add_trace(
            go.Scatter(
                x=g["turn"],
                y=g["running_agreement"],
                mode="lines+markers",
                name=pipeline,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title="Agreement with Nim-sum optimum (running)",
        xaxis_title="turn number",
        yaxis_title="running agreement rate",
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        margin=dict(l=20, r=20, t=50, b=40),
        height=340,
        plot_bgcolor="white",
    )
    return fig


def device_history_bar(device_df: pd.DataFrame) -> go.Figure:
    """Show the static Section 10 on-device balanced accuracies."""
    fig = go.Figure()
    if device_df is None or device_df.empty:
        fig.update_layout(
            title="On-device balanced accuracy (Section 10)",
            margin=dict(l=20, r=20, t=50, b=40),
            height=240,
        )
        return fig
    palette = {"VQC": "#264653", "QSVM": "#2a9d8f", "vqc": "#264653", "qsvm": "#2a9d8f"}
    for pipeline, g in device_df.groupby("pipeline"):
        labels = []
        for _, row in g.iterrows():
            labels.append(f"n={int(row['train_size'])}")
        fig.add_trace(
            go.Bar(
                x=labels,
                y=g["balanced_accuracy"],
                name=str(pipeline).upper(),
                marker_color=palette.get(str(pipeline), "#6c757d"),
                text=[f"{v:.2f}" for v in g["balanced_accuracy"]],
                textposition="outside",
            )
        )
    fig.update_layout(
        title="On-device balanced accuracy (Section 10)",
        yaxis=dict(range=[0, 1.05], title="balanced accuracy"),
        xaxis_title="refit train size",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        margin=dict(l=20, r=20, t=50, b=40),
        height=320,
        plot_bgcolor="white",
    )
    return fig
