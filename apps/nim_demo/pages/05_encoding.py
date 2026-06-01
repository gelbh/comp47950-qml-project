"""Encoding page — pick encoding, set heaps/symmetry, preview vector + circuit."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import viz  # type: ignore[import-not-found]
from content import (  # type: ignore[import-not-found]
    ENCODING_EXPLANATIONS,
    PAGE_BLURBS,
    PAGE_TITLES,
)

from qml_project.nim.encoding import (
    amplitude_vector,
    angle_parameters,
    binary_angle_features_matrix,
    build_encoding_circuit,
)

# Default-first order for the encoding picker (horizontal radio).
_ENCODING_ORDER: tuple[str, ...] = ("amplitude", "angle", "binary")


def _feature_vector(
    encoding: str,
    state: tuple[int, ...],
    symmetry: str,
    *,
    include_nim_sum: bool,
) -> np.ndarray:
    if encoding == "angle":
        return angle_parameters(
            state, M=7, include_nim_sum=include_nim_sum, symmetry=symmetry
        )
    if encoding == "amplitude":
        return amplitude_vector(
            state, M=7, include_nim_sum=include_nim_sum, symmetry=symmetry
        )
    if encoding == "binary":
        row = binary_angle_features_matrix(
            np.array([state], dtype=np.int32),
            bits_per_heap=3,
            include_nim_sum=include_nim_sum,
            symmetry=symmetry,  # type: ignore[arg-type]
        )
        return (row[0] / np.pi).astype(np.float64)
    raise ValueError(f"unsupported encoding: {encoding!r}")


def _show_encoding_circuit_mpl(fig: Any, *, encoding: str) -> None:
    """High-DPI PNG + CSS max-height so Streamlit does not stretch-blur tall Qiskit figures."""
    max_height_px, alt = {
        "angle": (220, "Angle encoding circuit"),
        "amplitude": (260, "Amplitude encoding circuit"),
        "binary": (480, "Binary encoding circuit"),
    }.get(encoding, (320, "Encoding circuit"))
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" alt="{alt}" '
        f'style="max-height:{max_height_px}px;width:auto;max-width:100%;'
        'display:block;margin-bottom:1rem;">',
        unsafe_allow_html=True,
    )


st.title(PAGE_TITLES["encoding"])
st.markdown(PAGE_BLURBS["encoding"])

encoding = st.radio(
    "Encoding",
    options=list(_ENCODING_ORDER),
    index=0,
    horizontal=True,
    format_func=lambda x: str(x).capitalize(),
    key="encoding_page_family",
)

st.subheader(f"How **{encoding}** encoding works")
st.markdown(ENCODING_EXPLANATIONS[encoding])

default_state = st.session_state.setdefault("learn_encoding_state", (1, 3, 5))
st.caption(
    "Heaps, symmetry, and **Include Nim-sum** change the **feature vector** "
    "and **circuit** below (same toggle as the VQC/QSVM sweeps)."
)
ctrl = st.columns([1, 1, 1, 2.2], gap="small")
h1 = ctrl[0].number_input(
    "h₁",
    min_value=0,
    max_value=7,
    value=int(default_state[0]),
    step=1,
    key="encoding_page_heap1",
)
h2 = ctrl[1].number_input(
    "h₂",
    min_value=0,
    max_value=7,
    value=int(default_state[1]),
    step=1,
    key="encoding_page_heap2",
)
h3 = ctrl[2].number_input(
    "h₃",
    min_value=0,
    max_value=7,
    value=int(default_state[2]),
    step=1,
    key="encoding_page_heap3",
)
symmetry = ctrl[3].selectbox(
    "Symmetry",
    options=("none", "canonical", "equivariant"),
    index=1,
    key="encoding_page_symmetry",
    help=(
        "**none** — heap order matches h₁, h₂, h₃.\n\n"
        "**canonical** — sort heaps ascending before encoding (all three encodings).\n\n"
        "**equivariant** — same heap order as none for angle/amplitude; for binary, "
        "adds extra CZ gates between heap wires."
    ),
)
include_nim_sum = st.checkbox(
    "Include Nim-sum in encoding",
    value=True,
    key="encoding_page_include_nim_sum",
    help=(
        "Same ablation as Parts 05/06: with Nim-sum off, angle and amplitude use "
        "three heap channels only; binary leaves the Nim-sum register in |0⟩. "
        "With Nim-sum on, a fourth angle / amplitude component and binary "
        "Nim-sum bits are included."
    ),
)
state = (int(h1), int(h2), int(h3))
st.session_state["learn_encoding_state"] = state

features = np.asarray(
    _feature_vector(encoding, state, symmetry, include_nim_sum=include_nim_sum)
).reshape(1, -1)

try:
    circuit = build_encoding_circuit(
        encoding,  # type: ignore[arg-type]
        state,
        M=7,
        bits_per_heap=3,
        iqp_reps=2,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,  # type: ignore[arg-type]
    )
except Exception as exc:
    circuit = None
    st.warning(f"Could not build encoding circuit: {exc}")

left, right = st.columns([0.45, 0.55], gap="large")

with left:
    _fv_note = (
        "Values are **heap bits then Nim-sum register bits** in {0,1} "
        "(same order as VQC binary features)."
        if encoding == "binary"
        else "Values are **radians** (angle) or **normalised amplitudes** (amplitude)."
    )
    st.markdown(
        f"**Feature vector** — {features.shape[1]} value(s) for state "
        f"`{state}` · encoding `{encoding}` · symmetry `{symmetry}` · "
        f"Nim-sum **{'on' if include_nim_sum else 'off'}**. {_fv_note}"
    )
    heatmap_fig = viz.encoding_heatmap(
        features,
        [(0, int(sum(state)))],
        title=None,
    )
    heatmap_fig.update_yaxes(showticklabels=False, title=None)
    heatmap_fig.update_layout(height=160)
    st.plotly_chart(heatmap_fig, width="stretch")
    if circuit is not None:
        st.caption(
            f"Circuit uses **{circuit.num_qubits} qubits** with depth "
            f"**{circuit.depth()}**."
        )

with right:
    st.markdown("**Encoding circuit**")
    if circuit is not None:
        try:
            fig = viz.render_encoding_circuit(circuit)
            _show_encoding_circuit_mpl(fig, encoding=encoding)
        except Exception as exc:
            st.warning(f"Could not render circuit: {exc}")
