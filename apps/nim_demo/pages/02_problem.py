"""Problem page — Nim rules, Nim-sum, and optimal play."""

from __future__ import annotations

import numpy as np
import streamlit as st

import viz  # type: ignore[import-not-found]
from content import PAGE_BLURBS, PAGE_TITLES  # type: ignore[import-not-found]

from qml_project.nim.game import is_terminal, is_winning, nim_sum, optimal_move

st.title(PAGE_TITLES["problem"])
st.markdown(PAGE_BLURBS["problem"])

_prev = st.session_state.get("learn_problem_state", (1, 3, 5))
with st.container(border=True):
    st.markdown("**Board state**")
    st.caption(
        "Set the three heap heights (heaps go up to **7** here, matching the "
        "project’s default **M**)."
    )
    _hp1, _hp2, _hp3 = st.columns(3, gap="medium")
    with _hp1:
        h1 = st.slider("h₁", 0, 7, int(_prev[0]), key="problem_page_h1")
    with _hp2:
        h2 = st.slider("h₂", 0, 7, int(_prev[1]), key="problem_page_h2")
    with _hp3:
        h3 = st.slider("h₃", 0, 7, int(_prev[2]), key="problem_page_h3")
state = (int(h1), int(h2), int(h3))
st.session_state["learn_problem_state"] = state

board_col, info_col = st.columns([0.55, 0.45], gap="large")

with board_col:
    st.markdown("**Board**")
    st.plotly_chart(
        viz.board_figure(state, max_heap_size=max(7, max(state)), interactive=False),
        width="stretch",
        theme=None,
    )

with info_col:
    ns = nim_sum(state)
    if is_terminal(state):
        verdict = "Terminal state (no stones left)."
        opt_label = "—"
    elif is_winning(state):
        move = optimal_move(state, rng=np.random.default_rng(0))
        verdict = r"**Winning** for the player to move ($h_1 \oplus h_2 \oplus h_3 \neq 0$)."
        opt_label = viz.move_label(move)
    else:
        verdict = r"**Losing** for the player to move ($h_1 \oplus h_2 \oplus h_3 = 0$)."
        opt_label = "any legal move (position is lost)"
    m1, m2 = st.columns(2)
    m1.metric("Nim-sum", ns)
    m2.metric("Optimal move", opt_label)
    st.markdown(verdict)

    n_bits = max(3, int(np.ceil(np.log2(max(state) + 1))) if any(state) else 1)
    st.markdown("**Binary breakdown**")
    st.dataframe(
        viz.nim_sum_table(state, n_bits=n_bits),
        width="stretch",
        hide_index=True,
    )
