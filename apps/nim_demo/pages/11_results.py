"""Results page — scatter views (simulation sweeps + IBM device)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import viz  # type: ignore[import-not-found]
from content import (  # type: ignore[import-not-found]
    PAGE_BLURBS,
    PAGE_TITLES,
)
from loaders import (  # type: ignore[import-not-found]
    load_device_history,
    load_summary_dataframes,
)

st.title(PAGE_TITLES["results"])
st.markdown(PAGE_BLURBS["results"])

summary = load_summary_dataframes()
classical_df: pd.DataFrame | None = summary.get("classical_df")
selection_df: pd.DataFrame | None = summary.get("selection_table")
winners_df: pd.DataFrame | None = summary.get("quantum_winners_summary")
device_df: pd.DataFrame | None = load_device_history()

_combined = viz.build_combined_selection_view(
    selection_df,
    classical_df,
    device_df=device_df,
    vqc_workflow_df=summary.get("vqc_workflow_df"),
    qsvm_workflow_df=summary.get("qsvm_workflow_df"),
)

st.markdown("**Quantum winners**")
if winners_df is None or winners_df.empty:
    st.info("No `quantum_winners_summary.parquet` cached yet.")
else:
    show_cols = [
        c
        for c in (
            "pipeline",
            "encoding",
            "include_nim_sum",
            "config_id",
            "variant_id",
            "mean_accuracy",
            "std_accuracy",
            "train_size_used",
            "overall_top",
        )
        if c in winners_df.columns
    ]
    view = winners_df[show_cols].copy()
    if "mean_accuracy" in view.columns:
        view["mean_accuracy"] = view["mean_accuracy"].round(3)
    if "std_accuracy" in view.columns:
        view["std_accuracy"] = view["std_accuracy"].round(3)
    st.dataframe(view, width="stretch", hide_index=True)

if not _combined.empty:
    st.markdown("---")
    st.markdown("**OOD balanced accuracy vs time (simulation + IBM device)**")
    st.caption(
        "Classical configs (max train, OOD), Section 07 quantum selection rows, and "
        "**IBM device** pickles (stars). Markers split **classical raw heaps vs "
        "parity/engineered features**, and **quantum simulation** by whether **Nim-sum "
        "is in the encoding** (circles = off, squares = on). Simulation rows prefer "
        "``include_nim_sum`` merged from ``vqc_workflow_df`` / ``qsvm_workflow_df`` "
        "when Section 07 omits it; otherwise ``|ns=T`` / ``|ns=F`` in ids. **X-axis:** "
        "seconds (log when spread is wide) — simulation uses **mean training time**; "
        "device uses **runtime_summary** when the pickle exposes it (otherwise that "
        "device point is omitted)."
    )
    st.plotly_chart(
        viz.combined_selection_cost_scatter(_combined),
        width="stretch",
    )
else:
    st.info(
        "No rows for scatter plots yet. Run Sections 05–07 (selection/classical "
        "parquets) and optionally Section 10 (``*_device_result_n*.pkl``)."
    )
