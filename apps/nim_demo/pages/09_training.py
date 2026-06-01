"""Training page — recipe blurb and cached workflow training-time chart."""

from __future__ import annotations

import streamlit as st

import viz  # type: ignore[import-not-found]
from content import PAGE_BLURBS, PAGE_TITLES  # type: ignore[import-not-found]
from loaders import load_summary_dataframes  # type: ignore[import-not-found]

st.title(PAGE_TITLES["training"])
_blurb = PAGE_BLURBS["training"]
if _blurb:
    st.markdown(_blurb)

summary = load_summary_dataframes()

left, right = st.columns([0.45, 0.55], gap="large")

with left:
    st.markdown("**Training recipe**")
    st.markdown(
        "- **VQC**: COBYLA on class-weighted softmax-NLL over bitstring probs.\n"
        "- **QSVM**: sklearn `SVC(kernel='precomputed')` on a pre-computed "
        "`N×N` kernel matrix.\n"
        "- **Classical**: sklearn defaults with `class_weight='balanced'`.\n"
        "- **Seeds**: `random_state=42` for all data splits; multi-seed "
        "sweeps report mean ± std."
    )

with right:
    st.markdown("**Training time vs train size (cached sweep)**")
    qv_w = summary.get("quantum_winner_rows_vqc")
    qs_w = summary.get("quantum_winner_rows_qsvm")
    vqc_df = (
        qv_w
        if qv_w is not None and not qv_w.empty
        else summary.get("vqc_workflow_df")
    )
    qsvm_df = (
        qs_w
        if qs_w is not None and not qs_w.empty
        else summary.get("qsvm_workflow_df")
    )
    classical_df = summary.get("classical_df")
    st.plotly_chart(
        viz.vqc_qsvm_training_time_grouped_bar(
            vqc_df, qsvm_df, classical_df=classical_df
        ),
        width="stretch",
    )
    st.caption(
        "Each bar is **mean wall-clock at that train size** after choosing a "
        "single configuration per pipeline. **VQC / QSVM** use Section 07 "
        "`quantum_winner_rows_*.parquet` when present (winner slice over seeds); "
        "otherwise the full `*_workflow_df` (mean over all configs and seeds). "
        "**Classical** uses the sweep row with highest **mean OOD balanced accuracy** "
        "(model × feature_set × symmetry), then averages `train_time_s` over seeds; "
        "if that cannot be resolved, it falls back to the OOD sweep mean."
    )
