"""QSVM architecture page — live quantum-kernel matrix on a configurable sample."""

from __future__ import annotations

import numpy as np
import streamlit as st

import viz  # type: ignore[import-not-found]
from content import (  # type: ignore[import-not-found]
    DEMO_ENCODINGS,
    PAGE_BLURBS,
    PAGE_TITLES,
)

from qml_project.nim.data import enumerate_states
from qml_project.qsvm import quantum_kernel_matrix

_QSVM_M = 7
# All legal Nim boards for k=3, M=7 (same as workflow notebooks / Play).
_QSVM_STATES_ALL = np.asarray(list(enumerate_states(3, _QSVM_M)), dtype=np.int32)
_QSVM_N_STATES_MAX = int(_QSVM_STATES_ALL.shape[0])
# Cap interactive N×N size (full 511×511 is too slow for Streamlit).
_QSVM_KERNEL_UI_MAX_N = min(150, _QSVM_N_STATES_MAX)
# Fixed permutation so increasing N extends the same ordered subset (reproducible).
_QSVM_PERM = np.random.default_rng(42).permutation(_QSVM_N_STATES_MAX).astype(
    np.intp, copy=False
)

_QSVM_ENCODING_HINT: dict[str, str] = {
    "angle": (
        "**Angle:** each heap maps to a data angle $\\theta_i = h_i\\pi/M$; "
        "the encoding is a product of single-qubit **RY** rotations."
    ),
    "amplitude": (
        "**Amplitude:** normalised heap components (and optionally Nim-sum) are "
        "loaded as **computational-basis amplitudes** on $\\lceil\\log_2|\\mathbf{v}|\\rceil$ qubits."
    ),
    "binary": (
        "**Binary:** each heap is written as bits; **X** flips 1-bits and **CZ** "
        "wires tie heaps together; the Nim-sum register can stay in $|0\\rangle$ or "
        "encode the XOR heap pattern."
    ),
}
_QSVM_SYMMETRY_HINT: dict[str, str] = {
    "none": "**Symmetry none:** heaps are encoded in the order sampled.",
    "canonical": (
        "**Symmetry canonical:** heaps are **sorted ascending** before the "
        "encoding circuit (removes trivial reorderings)."
    ),
    "equivariant": (
        "**Symmetry equivariant:** the circuit treats heap labels in a "
        "permutation-aware way (see Encoding page for the binary map)."
    ),
}

st.title(PAGE_TITLES["qsvm"])
st.markdown(PAGE_BLURBS["qsvm"])

with st.container(border=True):
    st.markdown("**Kernel preview**")
    st.caption(
        "Choose **encoding**, **symmetry**, **Nim-sum**, and **N** (number of Nim "
        "boards in the kernel). **N = max** uses every legal board once in a "
        "fixed random order (extending N keeps the same prefix of that order)."
    )
    _enc_col, _sym_col = st.columns(2, gap="large")
    with _enc_col:
        encoding = st.selectbox(
            "Encoding",
            options=list(DEMO_ENCODINGS),
            index=list(DEMO_ENCODINGS).index("amplitude"),
            key="qsvm_page_encoding",
        )
    with _sym_col:
        symmetry = st.selectbox(
            "Symmetry",
            options=("none", "canonical", "equivariant"),
            index=1,
            key="qsvm_page_symmetry",
        )
    include_nim_sum = st.checkbox(
        "Include Nim-sum in encoding",
        value=True,
        key="qsvm_page_include_nim_sum",
        help="Same ablation as Section 06 (`include_nim_sum` in workflow rows).",
    )
    n_sample = st.slider(
        "Sample size N",
        min_value=6,
        max_value=_QSVM_KERNEL_UI_MAX_N,
        value=20,
        key="qsvm_page_n_sample",
        help=(
            f"N×N kernel on the first **N** boards in a fixed permutation of all "
            f"**{_QSVM_N_STATES_MAX}** legal `k=3, M={_QSVM_M}` states. "
            f"Slider caps at **{_QSVM_KERNEL_UI_MAX_N}** for responsiveness."
        ),
    )
    st.markdown(
        f"{_QSVM_ENCODING_HINT[encoding]} {_QSVM_SYMMETRY_HINT[symmetry]}"
    )
    if not include_nim_sum:
        st.caption(
            "**Nim-sum off:** angle uses **3** qubits; amplitude uses **3** "
            "normalised heaps (padded to 4 amplitudes on **2** qubits); binary "
            "leaves the Nim-sum register in $|0\\rangle$."
        )

if n_sample >= 100:
    st.info(
        "Large **N**: the first **N×N** kernel build for this setting "
        "(encoding, symmetry, Nim-sum, N) can take noticeable CPU time; Streamlit "
        "**caches** the matrix so revisits stay fast."
    )


@st.cache_data(show_spinner="Computing quantum kernel…")
def _kernel(
    encoding: str,
    symmetry: str,
    n_sample: int,
    include_nim_sum: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_take = int(np.clip(n_sample, 6, _QSVM_KERNEL_UI_MAX_N))
    sel = _QSVM_PERM[:n_take]
    X = _QSVM_STATES_ALL[sel]
    K = quantum_kernel_matrix(
        X,
        X,
        encoding=encoding,  # type: ignore[arg-type]
        M=_QSVM_M,
        bits_per_heap=3,
        iqp_reps=2,
        include_nim_sum=include_nim_sum,
        symmetry=symmetry,  # type: ignore[arg-type]
        estimator_mode="exact_statevector",
        kernel_backend="manual",
        validate=False,
    )
    return K, X


K, X = _kernel(encoding, symmetry, n_sample, include_nim_sum)
_n = int(K.shape[0])
_rich = _n <= viz.KERNEL_HEATMAP_RICH_HOVER_MAX_N
_ticks = _n <= viz.KERNEL_HEATMAP_AXIS_TICKS_MAX_N

left, right = st.columns([0.6, 0.4], gap="large")

with left:
    st.plotly_chart(
        viz.kernel_matrix_heatmap(
            K,
            heap_rows=X,
            title=(
                f"K({encoding}, sym={symmetry}, nim_sum={include_nim_sum}) "
                f"— {_n}×{_n}"
            ),
        ),
        width="stretch",
    )
    if _ticks and _rich:
        st.caption(
            "Axes are **#0 … #N−1** along the fixed permutation. **Hover** a cell "
            "for both heap triples, **Nim-sum** (bitwise XOR of heaps), **W/L** for "
            "the player to move, and **k**."
        )
    elif _ticks:
        st.caption(
            "Axes show **#0 … #N−1**. **Hover** shows **k** only (full heap tooltips "
            f"return when **N ≤ {viz.KERNEL_HEATMAP_RICH_HOVER_MAX_N}**)."
        )
    else:
        st.caption(
            "**Tick labels hidden** at this size; axes are still row/column index "
            f"0 … {_n - 1}. **Hover** shows **k** (and indices) only."
        )

with right:
    st.markdown("**How to read it**")
    _hover_line = (
        "**Hover** the heatmap for heaps, Nim-sum, and win/loss on the two "
        "positions that define that cell."
        if _rich
        else "**Hover** shows matrix indices and **k** only at this N (large "
        "matrices skip per-cell annotations)."
    )
    st.markdown(
        "Each cell is `|⟨ψ(x)|ψ(x')⟩|²` — the squared state overlap "
        "between two encoded Nim positions. The diagonal is 1 by "
        "construction. Brighter off-diagonal blocks indicate states the "
        "encoding treats as similar. "
        + _hover_line
    )
    c1, c2 = st.columns(2)
    c1.metric("Min k", f"{float(K.min()):.2f}")
    c2.metric("Mean k (off-diag)", f"{float(K[~np.eye(K.shape[0], dtype=bool)].mean()):.2f}")
    st.caption(
        "A pure block-diagonal matrix is useless (every state looks unique). "
        "A fully bright matrix is also useless (every state looks the same). "
        "The useful regime is in between — structured similarity. "
        "Workflow sweeps also vary $C$, symmetry, and shot vs statevector "
        "estimators; this page uses **exact_statevector** only."
    )
