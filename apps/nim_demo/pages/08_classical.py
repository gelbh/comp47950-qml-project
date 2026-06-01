"""Classical baselines page — sweep charts and a live-fit table over feature sets."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import viz  # type: ignore[import-not-found]
from content import PAGE_BLURBS, PAGE_TITLES  # type: ignore[import-not-found]
from loaders import load_classical_bundle, load_summary_dataframes  # type: ignore[import-not-found]

from qml_project.baselines.features import (
    FEATURE_SET_DESCRIPTIONS,
    PARITY_ABLATION_FEATURE_SETS,
)

st.title(PAGE_TITLES["classical"])
st.markdown(PAGE_BLURBS["classical"])

summary = load_summary_dataframes()
classical_df: pd.DataFrame | None = summary.get("classical_df")
cdf = classical_df if classical_df is not None else pd.DataFrame()

left, right = st.columns([0.55, 0.45], gap="large")

with left:
    st.markdown("**Classical baselines (model comparison)**")
    st.plotly_chart(viz.classical_baseline_models_bar(cdf), width="stretch")
    st.caption(
        "Mean OOD balanced accuracy for each sklearn model, averaged over "
        "feature sets, training sizes, and seeds in the cached sweep."
    )
    st.markdown("**Feature-set ablation**")
    st.plotly_chart(viz.classical_feature_ablation_bar(cdf), width="stretch")
    st.caption(
        "Same sweep broken down by engineered feature vector; grouped bars "
        "are classifiers. Parity-style features are what make the baselines "
        "hard to beat."
    )

with right:
    st.markdown("**Probe settings**")
    classifier = st.selectbox(
        "Classifier",
        options=("Logistic Regression", "SVM (RBF)", "Random Forest"),
        index=0,
        help="Used for the live fits in the table and the sample-efficiency grid.",
        key="classical_page_classifier",
    )

    st.markdown("**Live fit — all feature sets**")
    st.caption(
        "Fresh sklearn fit on the OOD split for each engineered feature vector."
    )
    probe_rows: list[dict[str, object]] = []
    for fs in PARITY_ABLATION_FEATURE_SETS:
        try:
            bundle = load_classical_bundle(model_name=classifier, feature_set=fs)
        except Exception as exc:
            probe_rows.append(
                {
                    "feature_set": fs,
                    "description": FEATURE_SET_DESCRIPTIONS.get(fs, ""),
                    "train BA": None,
                    "OOD test BA": None,
                    "# features": None,
                    "error": str(exc),
                }
            )
        else:
            probe_rows.append(
                {
                    "feature_set": fs,
                    "description": FEATURE_SET_DESCRIPTIONS.get(fs, ""),
                    "train BA": round(bundle.train_balanced_accuracy, 3),
                    "OOD test BA": round(bundle.test_balanced_accuracy, 3),
                    "# features": len(bundle.feature_names),
                    "error": "",
                }
            )
    probe_df = pd.DataFrame(probe_rows)
    err_mask = probe_df["error"].astype(str).str.len() > 0
    display_df = probe_df.drop(columns=["error"]) if not err_mask.any() else probe_df
    st.dataframe(display_df, width="stretch", hide_index=True)
    if err_mask.any():
        failed = probe_df.loc[err_mask, "feature_set"].tolist()
        st.warning(
            "Could not fit classical bundle for: "
            + ", ".join(str(x) for x in failed)
        )

    if classical_df is not None and not classical_df.empty:
        sub = classical_df[classical_df["model"] == classifier].copy()
        if (
            not sub.empty
            and "train_size" in sub.columns
            and "feature_set" in sub.columns
            and "balanced_accuracy" in sub.columns
        ):
            grid = (
                sub.groupby(["train_size", "feature_set"], as_index=False)[
                    "balanced_accuracy"
                ]
                .mean()
                .pivot(
                    index="train_size",
                    columns="feature_set",
                    values="balanced_accuracy",
                )
            )
            ordered = [c for c in PARITY_ABLATION_FEATURE_SETS if c in grid.columns]
            extra = [c for c in grid.columns if c not in PARITY_ABLATION_FEATURE_SETS]
            grid = grid[ordered + extra].reset_index()
            st.markdown("**Sample-efficiency (mean OOD BA)**")
            st.caption(
                "Training-set sizes from the cached sweep; one column per feature set."
            )
            st.dataframe(grid.round(3), width="stretch", hide_index=True)
