"""Noise & device page — clean sim vs noisy sim vs real hardware."""

from __future__ import annotations

import streamlit as st

import viz  # type: ignore[import-not-found]
from content import (  # type: ignore[import-not-found]
    NOISE_PAGE_PLOT_FOOTNOTE,
    PAGE_BLURBS,
    PAGE_TITLES,
)
from loaders import load_device_history, workflow_cache_dir  # type: ignore[import-not-found]

st.title(PAGE_TITLES["noise"])
st.markdown(PAGE_BLURBS["noise"])

device_df = load_device_history()

left, right = st.columns([0.6, 0.4], gap="large")

with left:
    if device_df is None:
        cache = workflow_cache_dir()
        st.warning(
            "No on-device balanced-accuracy rows could be read from the "
            "workflow cache. Expected pickles like "
            "`vqc_device_result_n*.pkl` / `qsvm_device_result_n*.pkl` "
            "(written by the notebook device inference section in "
            "`notebooks/qml_project.ipynb`).\n\n"
            f"**Resolved cache directory:** `{cache}`\n\n"
            "If the folder is empty or missing, run the device inference section "
            "with submissions enabled; if pickles exist but this message persists, "
            "they may use an older layout — re-run that section to refresh."
        )
    else:
        st.plotly_chart(viz.device_history_bar(device_df), width="stretch")
        st.caption(NOISE_PAGE_PLOT_FOOTNOTE)

with right:
    st.markdown("**Three tiers of evaluation**")
    st.markdown(
        "1. **Clean simulation** — `StatevectorSampler`; gives the "
        "best-case label accuracy for a given model.\n"
        "2. **Noisy simulation** — Aer `SamplerV2` with a fake-Brisbane "
        "noise model (readout + depolarising). Use for ablations.\n"
        "3. **Real device** — `qiskit_ibm_runtime.SamplerV2` on an IBM "
        "backend. Inference only; training remains classical."
    )
    st.caption(
        "The drop between clean sim and real device is the headline "
        "noise cost — smaller drops mean the circuit tolerates hardware "
        "noise better; large drops point to a circuit or encoding that "
        "does not match the device well."
    )
