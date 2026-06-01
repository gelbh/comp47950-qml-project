"""Feature heatmaps and Qiskit matplotlib circuit figures for Learn pages."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go

from qml_project.nim.game import NimMove

from .board import move_label


def encoding_heatmap(
    features: np.ndarray,
    moves: Sequence[NimMove],
    *,
    title: str = "Encoded feature vectors",
) -> go.Figure:
    """Heatmap of (candidate, feature) for amplitude/angle encoded inputs."""
    move_labels = [move_label(m) for m in moves]
    feature_labels = [f"f{i}" for i in range(features.shape[1])]
    fig = go.Figure(
        go.Heatmap(
            z=features,
            x=feature_labels,
            y=move_labels,
            colorscale="Viridis",
            colorbar=dict(title="value"),
        )
    )
    layout_kwargs: dict = {
        "xaxis_title": "feature index",
        "yaxis_title": "candidate move",
        "margin": dict(l=20, r=20, t=40, b=30),
        "height": max(180, 40 + 22 * len(moves)),
    }
    if title is not None:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)
    return fig


def _shrink_matplotlib_figure(
    fig: Any,
    *,
    max_width_in: float,
    max_height_in: float,
) -> None:
    """Uniform scale-down so Streamlit never receives a 20+ inch tall/wide figure.

    Qiskit's mpl drawer sizes the canvas from gate count; ``st.pyplot`` then
    rasterises that canvas, so capping inches is the reliable fix for huge
    scroll areas (especially many fold segments on binary encodings).
    """
    w, h = fig.get_size_inches()
    if w <= 0 or h <= 0:
        return
    factor = min(1.0, max_width_in / w, max_height_in / h)
    if factor < 1.0:
        fig.set_size_inches(w * factor, h * factor, forward=True)


def cap_qiskit_mpl_figure(
    fig: Any,
    *,
    max_width_in: float,
    max_height_in: float,
) -> None:
    """Public wrapper: cap figure size in inches after ``circuit.draw(output='mpl')``."""
    _shrink_matplotlib_figure(fig, max_width_in=max_width_in, max_height_in=max_height_in)


def render_qiskit_circuit(circuit, *, fold: int = 40, scale: float = 0.9):
    """Return a matplotlib figure drawing the circuit.

    Kept tiny and folded so the demo never produces a giant unreadable image.
    """
    fig = circuit.draw(
        output="mpl",
        fold=fold,
        scale=scale,
        style={"backgroundcolor": "#ffffff"},
    )
    return fig


def render_encoding_circuit(circuit) -> Any:
    """Matplotlib figure for the Encoding learn page — scales down wide circuits.

    Binary encoding (9 qubits + many CZs) otherwise dominates the column;
    amplitude (2 qubits) can stay larger and more legible.

    Uses a **larger fold** for heavy circuits than before: a small ``fold``
    wraps often and stacks many horizontal strips, which makes mpl figures
    extremely tall. Smaller ``scale`` plus a hard cap on figure inches keeps
    the Learn column usable.
    """
    n = int(circuit.num_qubits)
    depth = int(circuit.depth())
    if n >= 9 or depth > 45:
        fold, scale = 22, 0.22
    elif n >= 6 or depth > 32:
        fold, scale = 20, 0.34
    elif n <= 2:
        fold, scale = 36, 0.62
    else:
        fold, scale = 28, 0.48
    if depth > 28:
        scale = max(0.16, scale * 0.82)
    fig = render_qiskit_circuit(circuit, fold=fold, scale=scale)
    # Right-hand column ~half page; keep raster footprint modest for Streamlit.
    _shrink_matplotlib_figure(fig, max_width_in=5.6, max_height_in=3.4)
    return fig
