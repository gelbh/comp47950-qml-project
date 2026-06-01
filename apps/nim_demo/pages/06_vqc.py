"""VQC architecture page — live `build_circuit` explorer."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from qiskit.circuit import Parameter, QuantumCircuit

import viz  # type: ignore[import-not-found]
from content import (  # type: ignore[import-not-found]
    ANSATZ_EXPLANATIONS,
    CZ_STRATEGY_EXPLANATIONS,
    PAGE_BLURBS,
    PAGE_TITLES,
    VQC_READOUT_MARKDOWN,
)

from qml_project.circuit import AnsatzName, build_circuit

_BITS_PER_HEAP = 3

_ANSATZ_LABELS = {"basic_block": "Basic block", "ry_rz": "RY–RZ"}
_CZ_LABELS = {"linear": "Linear chain", "all": "All pairs", "random": "Random pairs"}


def _fmt_ansatz(value: str) -> str:
    return _ANSATZ_LABELS.get(value, value)


def _fmt_cz(value: str) -> str:
    return _CZ_LABELS.get(value, value)


def _ansatz_slot_circuit(ansatz: AnsatzName) -> QuantumCircuit:
    """One-qubit block matching ``build_circuit`` (parameterised angle θ)."""
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    if ansatz == "basic_block":
        qc.rx(np.pi / 2, 0)
        qc.rz(theta, 0)
        qc.rx(np.pi / 2, 0)
    else:
        qc.ry(theta, 0)
        qc.rz(theta, 0)
    return qc


def _cz_layer_circuit_from_pairs(
    n_qubits: int,
    pairs: list[tuple[int, int]],
) -> QuantumCircuit:
    """CZ-only layer from an explicit pair list (e.g. first block of the live circuit)."""
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in pairs:
        qc.cz(q1, q2)
    return qc


def _st_circuit_sketch(fig, *, width: str | int = "stretch") -> None:
    """``width='content'`` keeps intrinsic figure size (no column upscale blur)."""
    st.pyplot(fig, width=width)
    plt.close(fig)


def _vqc_explainer_sketch_params(
    n_qubits: int, n_cz: int
) -> tuple[int, float, float, float, float]:
    """CZ fold/scale plus **shared** max figure inches (ansatz + CZ use same cap)."""
    if n_qubits <= 2 and n_cz <= 1:
        cz_fold, cz_scale, mw, mh = 36, 0.32, 2.65, 0.78
    elif n_qubits <= 2:
        cz_fold, cz_scale, mw, mh = 30, 0.38, 2.85, 0.88
    elif n_qubits == 3 and n_cz <= 2:
        cz_fold, cz_scale, mw, mh = 26, 0.46, 3.15, 1.02
    elif n_qubits == 3:
        cz_fold, cz_scale, mw, mh = 22, 0.52, 3.45, 1.18
    elif n_cz >= 6:
        cz_fold, cz_scale, mw, mh = 8, 0.48, 4.35, 2.05
    elif n_qubits >= 4:
        cz_fold, cz_scale, mw, mh = 14, 0.58, 3.95, 1.52
    else:
        cz_fold, cz_scale, mw, mh = 20, 0.55, 3.5, 1.25
    # One-qubit ansatz: scale with CZ so both feel similar after the same inch cap.
    ans_scale = max(0.34, min(0.82, cz_scale * 1.08))
    return cz_fold, cz_scale, mw, mh, ans_scale


def _part05_vqc_dims(
    encoding: str, *, include_nim_sum: bool
) -> tuple[int, int, int]:
    """Return ``(n_qubits, n_features, min_layers)`` matching Section 05 profiles."""
    B = _BITS_PER_HEAP
    if encoding == "angle":
        n = 4 if include_nim_sum else 3
        min_layers = max(2, 2 * math.ceil(n / n) - 1)
        return n, n, min_layers
    if encoding == "amplitude":
        n_qubits, n_features = 2, 4
        min_layers = max(2, 2 * math.ceil(n_features / n_qubits) - 1)
        return n_qubits, n_features, max(4, min_layers)
    if encoding == "binary":
        n_qubits = n_features = 4 * B
        min_layers = max(2, 2 * math.ceil(n_features / n_qubits) - 1)
        return n_qubits, n_features, min_layers
    raise ValueError(f"unknown encoding: {encoding!r}")


st.title(PAGE_TITLES["vqc"])
st.markdown(PAGE_BLURBS["vqc"])

with st.container(border=True):
    st.markdown("**Live circuit build**")
    st.caption(
        "Everything here is passed straight into `build_circuit` and the "
        "diagram below. **Play** loads checkpoints from the Section 08 cache "
        "(`config_id`, `n_features`, encoding, `include_nim_sum`) — they need "
        "not match the exploratory sliders here."
    )
    st.markdown("**Section 05 preset (encoding × Nim-sum)**")
    _pc1, _pc2, _pc3 = st.columns([1.1, 1.0, 1.0])
    with _pc1:
        _preset_enc = st.selectbox(
            "Encoding",
            ("angle", "amplitude", "binary"),
            index=0,
            key="vqc_page_preset_encoding",
            help="Dimensions used in the VQC workflow grid for that encoding.",
        )
    with _pc2:
        _preset_ns = st.checkbox(
            "Include Nim-sum",
            value=True,
            key="vqc_page_preset_include_nim_sum",
            help="Same flag as Section 05 / Section 06 (`include_nim_sum`).",
        )
    with _pc3:
        if st.button("Apply preset to qubits / features / min layers", type="primary"):
            nq, nf, ml = _part05_vqc_dims(_preset_enc, include_nim_sum=_preset_ns)
            st.session_state["vqc_page_n_qubits"] = int(nq)
            st.session_state["vqc_page_n_features"] = int(nf)
            st.session_state["vqc_page_n_layers"] = max(
                int(ml), int(st.session_state.get("vqc_page_n_layers", ml))
            )
            st.rerun()
    st.caption(
        "Angle: **3** qubits / features when Nim-sum is off, **4** when on. "
        "Amplitude: **2** qubits, **4** features. Binary: **12** qubits / features "
        f"(`4×{_BITS_PER_HEAP}` bits including Nim-sum register)."
    )
    n_features = st.slider(
        "Number of features",
        2,
        16,
        4,
        key="vqc_page_n_features",
    )
    st.markdown(
        "**Ansatz** and **CZ strategy** set the parameter-layer rotations and "
        "which wires get CZ entanglement after each block."
    )
    c_ansatz, c_cz = st.columns(2, gap="large")
    with c_ansatz:
        ansatz = st.selectbox(
            "Ansatz",
            ("basic_block", "ry_rz"),
            index=0,
            key="vqc_page_ansatz",
            format_func=_fmt_ansatz,
            help="Per-qubit rotation pattern inside each trainable layer.",
        )
    with c_cz:
        cz_strategy = st.selectbox(
            "CZ strategy",
            ("linear", "all", "random"),
            index=0,
            key="vqc_page_cz_strategy",
            format_func=_fmt_cz,
            help="How qubit pairs are chosen for CZ gates after each parameter layer.",
        )

st.markdown("**Live circuit**")
circuit_col, stats_col = st.columns([0.62, 0.38], gap="large")

with stats_col:
    n_qubits = st.number_input(
        "Qubits",
        min_value=2,
        max_value=12,
        value=3,
        step=1,
        key="vqc_page_n_qubits",
        help=(
            "Circuit width (Section 05 uses up to **12** for binary encoding). "
            "Wide circuits are slower to render."
        ),
    )
    # Even indices are feature layers; need enough of them for every x[k].
    _min_layers = max(2, 2 * math.ceil(n_features / n_qubits) - 1)
    _ly_key = "vqc_page_n_layers"
    if _ly_key not in st.session_state:
        st.session_state[_ly_key] = max(4, _min_layers)
    elif st.session_state[_ly_key] < _min_layers:
        st.session_state[_ly_key] = _min_layers
    n_layers = st.number_input(
        "Layers",
        min_value=_min_layers,
        max_value=10,
        step=1,
        key=_ly_key,
        help=(
            "Alternating feature / parameter blocks. Minimum rises with feature "
            "count so every feature angle appears in the circuit."
        ),
    )

try:
    vqc = build_circuit(
        n_qubits=n_qubits,
        n_features=n_features,
        n_classes=2,
        n_layers=n_layers,
        cz_strategy=cz_strategy,  # type: ignore[arg-type]
        ansatz=ansatz,  # type: ignore[arg-type]
    )
except Exception as exc:
    st.error(f"Could not build circuit: {exc}")
    st.stop()

with circuit_col:
    try:
        fig = viz.render_qiskit_circuit(vqc.circuit, fold=22)
        _st_circuit_sketch(fig)
    except Exception as exc:
        st.warning(f"Could not render circuit: {exc}")

with stats_col:
    depth = vqc.circuit.depth()
    n_trainable = int(getattr(vqc, "n_trainable", 0) or len(vqc.trainable_params))
    c1, c2 = st.columns(2)
    c1.metric("Circuit depth", depth)
    c2.metric("Trainable θ", n_trainable)
    st.caption(
        "Even-indexed layers are **feature** layers (data angles); odd-indexed "
        "layers are **parameter** layers (trainable "
        r"$\boldsymbol{\theta}$). Ansatz and CZ strategy are explained below."
    )

_cz_pairs0 = vqc.cz_pairs_per_layer[0] if vqc.cz_pairs_per_layer else []
_sk_n_cz = len(_cz_pairs0)
_cz_fold, _cz_scale, _sk_mw, _sk_mh, _ans_scale = _vqc_explainer_sketch_params(
    n_qubits, _sk_n_cz
)

st.markdown(f"### `{ansatz}` — ansatz")
_ans_col_t, _ans_col_d = st.columns(
    [0.58, 0.42], gap="medium", vertical_alignment="center"
)
with _ans_col_t:
    st.markdown(ANSATZ_EXPLANATIONS[ansatz])
with _ans_col_d:
    st.caption("Single-qubit slot (feature or trainable angle)")
    try:
        _ans_fig = viz.render_qiskit_circuit(
            _ansatz_slot_circuit(ansatz),  # type: ignore[arg-type]
            fold=44,
            scale=_ans_scale,
        )
        viz.cap_qiskit_mpl_figure(_ans_fig, max_width_in=_sk_mw, max_height_in=_sk_mh)
        _st_circuit_sketch(_ans_fig, width="content")
    except Exception as exc:
        st.caption(f"Sketch unavailable: {exc}")

st.markdown(f"### `{cz_strategy}` — CZ strategy")
_cz_col_t, _cz_col_d = st.columns(
    [0.58, 0.42], gap="medium", vertical_alignment="center"
)
with _cz_col_t:
    st.markdown(CZ_STRATEGY_EXPLANATIONS[cz_strategy])
with _cz_col_d:
    st.caption(
        f"CZ layer on **{n_qubits}** qubits (first entangling block in the live circuit)"
    )
    try:
        _cz_fig = viz.render_qiskit_circuit(
            _cz_layer_circuit_from_pairs(n_qubits, _cz_pairs0),
            fold=_cz_fold,
            scale=_cz_scale,
        )
        viz.cap_qiskit_mpl_figure(_cz_fig, max_width_in=_sk_mw, max_height_in=_sk_mh)
        _st_circuit_sketch(_cz_fig, width="content")
    except Exception as exc:
        st.caption(f"Sketch unavailable: {exc}")

with st.container(border=True):
    st.markdown("**Measurement → class label**")
    st.markdown(VQC_READOUT_MARKDOWN)
