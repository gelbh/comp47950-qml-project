"""Board drawings, move/state labels, and Nim-sum bit table for the demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from qml_project.nim.game import NimMove, NimState

from ._constants import COLOR_HEAP, COLOR_STONE_EMPTY


def move_label(move: NimMove) -> str:
    heap, amount = move
    return f"heap {heap + 1}: −{amount}"


def state_label(state: NimState) -> str:
    return "(" + ", ".join(str(int(h)) for h in state) + ")"


def board_figure(
    state: NimState,
    max_heap_size: int = 7,
    *,
    interactive: bool = False,
) -> go.Figure:
    """Render heaps as horizontal rows of coloured squares.

    When ``interactive`` is True, every remaining stone becomes a clickable
    marker carrying ``customdata = [heap_index, take_amount]``. Hook this
    figure up to ``st.plotly_chart(..., on_select="rerun")`` to let the
    player take stones by clicking on the board itself.
    """
    k = len(state)
    fig = go.Figure()
    fig.add_annotation(
        x=max_heap_size / 2,
        y=k + 0.35,
        text=(
            "Click a stone to take it plus everything to its right in the same heap."
            if interactive
            else ""
        ),
        showarrow=False,
        font=dict(size=11, color="#6c757d"),
    )

    for i, h in enumerate(state):
        for j in range(max_heap_size):
            filled = j < h
            fig.add_shape(
                type="rect",
                x0=j + 0.05,
                x1=j + 0.95,
                y0=k - i - 0.4,
                y1=k - i + 0.4,
                line=dict(color="#adb5bd" if not filled else "#212529", width=1),
                fillcolor=(
                    COLOR_HEAP[i % len(COLOR_HEAP)] if filled else COLOR_STONE_EMPTY
                ),
                layer="below",
            )
        fig.add_annotation(
            x=-0.5,
            y=k - i,
            text=f"heap {i + 1}",
            showarrow=False,
            font=dict(size=14, color="#212529"),
            xanchor="right",
        )
        fig.add_annotation(
            x=max_heap_size + 0.2,
            y=k - i,
            text=f"{int(h)} left",
            showarrow=False,
            font=dict(size=12, color="#495057"),
            xanchor="left",
        )

    if interactive:
        xs: list[float] = []
        ys: list[float] = []
        customdata: list[list[int]] = []
        hover: list[str] = []
        for i, h in enumerate(state):
            for j in range(int(h)):
                amount = h - j
                xs.append(j + 0.5)
                ys.append(k - i)
                customdata.append([i, int(amount)])
                hover.append(f"Take {amount} from heap {i + 1}")
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        size=28,
                        color="rgba(0,0,0,0)",
                        line=dict(color="rgba(0,0,0,0)", width=0),
                    ),
                    hovertext=hover,
                    hoverinfo="text",
                    customdata=customdata,
                    showlegend=False,
                    cliponaxis=False,
                )
            )

    fig.update_xaxes(visible=False, range=[-2.2, max_heap_size + 2.2])
    fig.update_yaxes(visible=False, range=[0.2, k + 0.9])
    fig.update_layout(
        height=80 + 60 * k,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        showlegend=False,
        dragmode=False,
        clickmode="event+select" if interactive else None,
    )
    return fig


def board_editor_figure(state: NimState, max_heap_size: int = 7) -> go.Figure:
    """Three heaps as rows of stones; click **+** to add one, **−** to remove one.

    Plotly selections in Streamlit do not expose mouse buttons, so add/remove
    are two marker traces (same layout as :func:`board_figure`). Selection
    ``customdata`` is ``[heap_index, delta]`` with ``delta`` ``+1`` or ``-1``.
    """
    k = len(state)
    state_t = tuple(int(x) for x in state)
    fig = go.Figure()
    fig.add_annotation(
        x=max_heap_size / 2,
        y=k + 0.35,
        text=(
            "Left-click **+** on the first empty slot to add a stone, "
            "or **−** on the top stone to remove one."
        ),
        showarrow=False,
        font=dict(size=11, color="#6c757d"),
    )

    for i, h in enumerate(state_t):
        for j in range(max_heap_size):
            filled = j < h
            fig.add_shape(
                type="rect",
                x0=j + 0.05,
                x1=j + 0.95,
                y0=k - i - 0.4,
                y1=k - i + 0.4,
                line=dict(color="#adb5bd" if not filled else "#212529", width=1),
                fillcolor=(
                    COLOR_HEAP[i % len(COLOR_HEAP)] if filled else COLOR_STONE_EMPTY
                ),
                layer="below",
            )
        fig.add_annotation(
            x=-0.5,
            y=k - i,
            text=f"heap {i + 1}",
            showarrow=False,
            font=dict(size=14, color="#212529"),
            xanchor="right",
        )
        fig.add_annotation(
            x=max_heap_size + 0.2,
            y=k - i,
            text=f"{int(h)} stones",
            showarrow=False,
            font=dict(size=12, color="#495057"),
            xanchor="left",
        )

    xs_add: list[float] = []
    ys_add: list[float] = []
    cd_add: list[list[int]] = []
    hover_add: list[str] = []
    xs_sub: list[float] = []
    ys_sub: list[float] = []
    cd_sub: list[list[int]] = []
    hover_sub: list[str] = []
    for i, h in enumerate(state_t):
        yi = float(k - i)
        if h < max_heap_size:
            xs_add.append(h + 0.5)
            ys_add.append(yi)
            cd_add.append([i, 1])
            hover_add.append(f"Add 1 stone to heap {i + 1}")
        if h > 0:
            xs_sub.append((h - 1) + 0.5)
            ys_sub.append(yi)
            cd_sub.append([i, -1])
            hover_sub.append(f"Remove 1 stone from heap {i + 1}")

    if xs_add:
        fig.add_trace(
            go.Scatter(
                x=xs_add,
                y=ys_add,
                mode="markers+text",
                text=["+"] * len(xs_add),
                textposition="middle center",
                textfont=dict(size=14, color="#1d6f5c", family="Arial Black"),
                marker=dict(
                    size=26,
                    color="rgba(216, 243, 220, 0.95)",
                    line=dict(color="#2a9d8f", width=2),
                    symbol="circle",
                ),
                customdata=cd_add,
                hovertext=hover_add,
                hoverinfo="text",
                name="add",
                showlegend=False,
            )
        )
    if xs_sub:
        fig.add_trace(
            go.Scatter(
                x=xs_sub,
                y=ys_sub,
                mode="markers+text",
                text=["−"] * len(xs_sub),
                textposition="middle center",
                textfont=dict(size=18, color="#c1121f", family="Arial Black"),
                marker=dict(
                    size=26,
                    color="rgba(255, 221, 210, 0.95)",
                    line=dict(color="#e76f51", width=2),
                    symbol="circle",
                ),
                customdata=cd_sub,
                hovertext=hover_sub,
                hoverinfo="text",
                name="remove",
                showlegend=False,
            )
        )

    fig.update_xaxes(visible=False, range=[-2.2, max_heap_size + 2.2])
    fig.update_yaxes(visible=False, range=[0.2, k + 0.9])
    fig.update_layout(
        height=100 + 60 * k,
        margin=dict(l=10, r=10, t=36, b=10),
        plot_bgcolor="white",
        showlegend=False,
        dragmode=False,
        clickmode="event+select",
    )
    return fig


def nim_sum_table(state: NimState, *, n_bits: int | None = None) -> pd.DataFrame:
    """Build a heap-by-bit table with the column XOR annotated as the Nim-sum."""
    arr = np.asarray(state, dtype=np.int32)
    if n_bits is None:
        m = int(arr.max()) if arr.size else 0
        n_bits = max(1, int(np.ceil(np.log2(m + 1))) if m > 0 else 1)
    bit_cols = [f"bit {b}" for b in reversed(range(n_bits))]
    rows = []
    for i, h in enumerate(arr):
        bits = [((int(h) >> b) & 1) for b in reversed(range(n_bits))]
        rows.append({"heap": f"h{i + 1} = {int(h)}", **dict(zip(bit_cols, bits))})
    nim_s = int(np.bitwise_xor.reduce(arr)) if arr.size else 0
    xor_bits = [((nim_s >> b) & 1) for b in reversed(range(n_bits))]
    rows.append({"heap": f"XOR = {nim_s}", **dict(zip(bit_cols, xor_bits))})
    return pd.DataFrame(rows)
