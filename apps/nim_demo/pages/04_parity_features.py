"""Classical parity-style features — Learn page 04; heatmaps + PCA (notebook Sections 02/03)."""

from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA

import viz  # type: ignore[import-not-found]
from content import (  # type: ignore[import-not-found]
    PAGE_BLURBS,
    PAGE_TITLES,
    PARITY_FEATURES_MARKDOWN,
)

from qml_project.baselines.features import prepare_features
from qml_project.nim.data import prepare_experiment_data


@st.cache_data(show_spinner=False)
def _ood_train_feature_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw / parity / bit-parity design matrices and labels for the OOD train pool."""
    data = prepare_experiment_data(k=3, M=7, random_state=42)
    split = data.split
    X_raw = prepare_features(split.X_train, "raw", M=7)
    X_parity = prepare_features(split.X_train, "parity", M=7)
    X_bit = prepare_features(split.X_train, "bit_parity", M=7)
    return X_raw, X_parity, X_bit, split.y_train


def _apply_heap_delta(
    state: tuple[int, int, int], heap_i: int, delta: int, *, M: int
) -> tuple[int, int, int]:
    heaps = list(state)
    heaps[heap_i] = max(0, min(M, heaps[heap_i] + delta))
    return tuple(heaps)


M_BOARD = 7
st.session_state.setdefault("learn_encoding_state", (1, 3, 5))
state = tuple(int(x) for x in st.session_state["learn_encoding_state"])

st.title(PAGE_TITLES["parity_features"])
st.markdown(PAGE_BLURBS["parity_features"])
st.markdown(PARITY_FEATURES_MARKDOWN)

st.markdown("**Win / loss in the $(h_1, h_2)$ planes**")
st.caption(
    "Same layout as Section 02: losing (Nim-sum $= 0$) in orange, winning in teal, "
    "terminal $(0,0,0)$ in grey on the $h_3=0$ panel."
)
st.plotly_chart(viz.nim_h1_h2_winloss_heatmaps(M=M_BOARD, k=3), width="stretch")

X_raw, X_parity, X_bit, y_train = _ood_train_feature_matrices()
_n_ood_train = int(len(y_train))
st.markdown("**OOD train set — 2D PCA in three feature spaces**")
st.caption(
    "Matches Section 03: PCA is fit **inside** each feature space on the same "
    f"**{_n_ood_train}** OOD training states; points are coloured by win/loss. "
    "The **red star** uses the board you set **below** (same heaps as the Encoding page)."
)
pca_r = PCA(n_components=2).fit(X_raw)
pca_p = PCA(n_components=2).fit(X_parity)
pca_b = PCA(n_components=2).fit(X_bit)
Z_raw = pca_r.transform(X_raw)
Z_parity = pca_p.transform(X_parity)
Z_bit = pca_b.transform(X_bit)
row = np.array([state], dtype=np.int64)
pt_r = pca_r.transform(prepare_features(row, "raw", M=M_BOARD)).reshape(2)
pt_p = pca_p.transform(prepare_features(row, "parity", M=M_BOARD)).reshape(2)
pt_b = pca_b.transform(prepare_features(row, "bit_parity", M=M_BOARD)).reshape(2)
st.plotly_chart(
    viz.classical_train_pca_scatter_grid(
        Z_raw,
        Z_parity,
        Z_bit,
        y_train,
        highlight_raw=pt_r,
        highlight_parity=pt_p,
        highlight_bit=pt_b,
    ),
    width="stretch",
)

st.markdown("**Preview board**")
st.caption(
    "Same heap layout as **Play**: stacks grow to the right. "
    "Streamlit cannot read right-clicks on Plotly — use the **+** on the first "
    "empty slot to add a stone, or **−** on the top stone to remove one "
    "(left-click only)."
)
fig_board = viz.board_editor_figure(state, max_heap_size=M_BOARD)
board_key = f"parity_board_{state[0]}_{state[1]}_{state[2]}"
board_event = st.plotly_chart(
    fig_board,
    width="stretch",
    on_select="rerun",
    selection_mode=["points"],
    key=board_key,
)
if board_event is not None:
    sel = getattr(board_event, "selection", None)
    if isinstance(sel, dict):
        pts = sel.get("points") or []
        if pts:
            cd = pts[0].get("customdata")
            if cd is not None and len(cd) >= 2:
                heap_i = int(cd[0])
                delta = int(cd[1])
                if delta in (-1, 1) and 0 <= heap_i < 3:
                    new_state = _apply_heap_delta(state, heap_i, delta, M=M_BOARD)
                    st.session_state["learn_encoding_state"] = new_state
                    st.rerun()

h1, h2, h3 = state
nim_sum = int(h1 ^ h2 ^ h3)
st.caption(
    f"Heaps `({h1}, {h2}, {h3})` — Nim-sum "
    f"$h_1 \\oplus h_2 \\oplus h_3 = {nim_sum}$."
)

st.caption(
    "Sweep numbers for these feature sets live on **Classical baselines**; "
    "quantum **Encoding** is separate from this sklearn-side engineering."
)
