"""Data page — class balance and OOD split for the Nim dataset."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from content import PAGE_BLURBS, PAGE_TITLES  # type: ignore[import-not-found]

from qml_project.nim.data import (
    class_balance_table,
    enumerate_states,
    ood_split,
    split_class_balance,
)

st.title(PAGE_TITLES["data"])
st.markdown(PAGE_BLURBS["data"])

with st.container(border=True):
    st.markdown("**Dataset scope**")
    st.caption(
        "**M** is the maximum heap size in the full Nim catalogue; **M_train** "
        "sets the in-distribution cap (train on heaps ≤ M_train, OOD test when "
        "any heap exceeds it)."
    )
    _m_col, _mt_col = st.columns(2, gap="large")
    with _m_col:
        M = st.slider(
            "Max heap size M",
            min_value=3,
            max_value=7,
            value=7,
            key="data_page_M",
        )
    with _mt_col:
        # Streamlit requires slider min_value < max_value; for M=3 only M_train=2 is valid.
        if M <= 3:
            M_train = 2
            st.metric("Train cutoff M_train", str(M_train))
            st.caption("Fixed: only one value works with **M ≤ 3**.")
        else:
            M_train = st.slider(
                "Train cutoff M_train",
                min_value=2,
                max_value=M - 1,
                value=min(5, M - 1),
                key="data_page_M_train",
            )


@st.cache_data(show_spinner=False)
def _balance(M: int) -> pd.DataFrame:
    return class_balance_table(M_values=range(1, M + 1), k=3)


@st.cache_data(show_spinner=False)
def _ood_rows(M: int, M_train: int) -> pd.DataFrame:
    split = ood_split(k=3, M_train=M_train, M_test=M)
    rows = split_class_balance(
        {
            "Train (heaps ≤ M_train)": split.y_train,
            "OOD Test (heap > M_train)": split.y_test,
        }
    )
    return rows


balance_df = _balance(M)
ood_df = _ood_rows(M, M_train)

left, right = st.columns([0.55, 0.45], gap="large")

with left:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="losing",
            x=balance_df["M"],
            y=balance_df["losing"],
            marker_color="#e76f51",
        )
    )
    fig.add_trace(
        go.Bar(
            name="winning",
            x=balance_df["M"],
            y=balance_df["winning"],
            marker_color="#2a9d8f",
        )
    )
    fig.update_layout(
        barmode="stack",
        title=f"State counts by M (k=3, total = {len(enumerate_states(3, M))})",
        xaxis_title="M (max heap size)",
        yaxis_title="number of states",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Minority class stays around 12% of states — that is why we rely on "
        "balanced accuracy rather than raw accuracy."
    )

with right:
    st.markdown("**OOD split**")
    st.caption(
        f"Train on heaps ≤ {M_train}; OOD test on any state with a heap > {M_train}."
    )
    st.dataframe(ood_df, width="stretch", hide_index=True)
    st.markdown("**Class balance by M**")
    st.dataframe(balance_df, width="stretch", hide_index=True, height=230)
