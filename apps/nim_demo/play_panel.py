"""Play page: right-hand explanation tabs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import engine  # type: ignore[import-not-found]
import viz  # type: ignore[import-not-found]
from qml_project.nim.game import apply_move, is_terminal

import play_session as ps
import play_turns as pt
from play_config import OpponentName, Slot


def slot_tab_spec(pipeline: OpponentName | None) -> list[str]:
    base_tail = ["Compare", "History"]
    if pipeline == "VQC":
        return ["Decision", "Circuit", "Encoding", "Class probs", *base_tail]
    if pipeline == "QSVM":
        return ["Decision", "Kernel", "SV contributions", *base_tail]
    if pipeline == "Classical":
        return ["Decision", "Features", "Class probs", *base_tail]
    return ["Decision", *base_tail]


def render_right_panel() -> None:
    slot = ps.panel_slot()
    pipeline = ps.slot_pipeline(slot)
    labels = slot_tab_spec(pipeline)
    tabs = st.tabs(labels)
    idx = {label: i for i, label in enumerate(labels)}

    state = st.session_state.game["state"]
    turn = pt.explain_slot(state, slot)

    with tabs[idx["Decision"]]:
        tab_decision(turn, slot)
    if "Circuit" in idx:
        with tabs[idx["Circuit"]]:
            tab_vqc_circuit(turn)
    if "Encoding" in idx:
        with tabs[idx["Encoding"]]:
            tab_vqc_encoding(turn)
    if "Class probs" in idx and pipeline == "VQC":
        with tabs[idx["Class probs"]]:
            tab_vqc_class_probs(turn)
    if "Kernel" in idx:
        with tabs[idx["Kernel"]]:
            tab_qsvm_kernel(turn)
    if "SV contributions" in idx:
        with tabs[idx["SV contributions"]]:
            tab_qsvm_contributions(turn)
    if "Features" in idx:
        with tabs[idx["Features"]]:
            tab_classical_features(turn)
    if "Class probs" in idx and pipeline == "Classical":
        with tabs[idx["Class probs"]]:
            tab_classical_class_probs(turn)
    with tabs[idx["Compare"]]:
        tab_compare()
    with tabs[idx["History"]]:
        tab_history()


_CONFIDENCE_CAPTION = {
    "VQC": "probability that the move leaves the opponent in a losing position",
    "QSVM": (
        "sign-flipped SVM margin — higher means the resulting state is "
        "classified more confidently as 'losing' for the opponent"
    ),
    "Classical": "classifier probability that the move leaves the opponent in a losing position",
}


def tab_decision(turn: engine.TurnExplanation, slot: Slot) -> None:
    if is_terminal(turn.state):
        st.info("Game is over. Start a new game from the sidebar.")
        return
    pipeline = ps.slot_pipeline(slot)
    if pipeline is None:
        st.info("No model is to move right now.")
        return
    exp = pt.get_exp_for(turn, pipeline)
    if exp is None:
        st.warning(f"No {pipeline} checkpoint available.")
        return

    label = ps.slot_label(slot)
    st.markdown(f"#### Under the hood: **{label}**")

    opt = turn.optimal
    scores = exp.scores
    best_move = scores.best_move
    confidence = float(scores.score[scores.best_index])
    agrees = (
        bool(tuple(best_move) == tuple(opt.optimal_move))
        if opt.is_winning_for_player_to_move
        else None
    )
    tmap = turn.pipeline_timings_ms or {}
    cur_ms = tmap.get(pipeline)
    n_metrics = 4 if cur_ms is not None else 3
    cols = st.columns(n_metrics)
    cols[0].metric("Pick", viz.move_label(best_move))
    cols[1].metric("Confidence", f"{confidence:.2f}")
    if agrees is None:
        cols[2].metric("Vs. optimum", "n/a")
    else:
        cols[2].metric("Vs. optimum", "agree" if agrees else "disagree")
    if cur_ms is not None:
        cols[3].metric("Inference", f"{cur_ms:.1f} ms")
    st.caption(_CONFIDENCE_CAPTION.get(pipeline, ""))

    if tmap:
        st.markdown("**Inference time** (wall clock, local simulator, this board position)")
        timing_rows = [
            {
                "Pipeline": (
                    f"{name} (policy to move)" if name == pipeline else name
                ),
                "ms": round(ms, 2),
            }
            for name, ms in sorted(tmap.items(), key=lambda x: x[0])
        ]
        st.dataframe(pd.DataFrame(timing_rows), width="stretch", hide_index=True)
        st.caption(
            "VQC includes finite-shot estimation; QSVM uses an exact statevector kernel row; "
            "classical is a single sklearn forward pass. Times refresh when the board changes."
        )

    st.plotly_chart(
        viz.candidate_score_bar(
            scores.moves,
            scores.score,
            best_index=scores.best_index,
            optimal_move=opt.optimal_move if opt.is_winning_for_player_to_move else None,
            score_label="score per candidate move",
            title=None,
        ),
        width="stretch",
    )


def tab_vqc_circuit(turn: engine.TurnExplanation) -> None:
    exp = turn.vqc
    if exp is None:
        st.warning("No VQC payload available.")
        return
    st.markdown(
        f"**Quantum circuit** — {exp.n_qubits} qubits · "
        f"θ of length {exp.theta.size} · {exp.shots} shots · "
        f"decision rule `{exp.decision_rule}`."
    )
    st.caption(
        "This is the exact circuit the model submits for the chosen move. "
        "Feature layers encode the resulting state; parameter layers carry "
        "the trained angles θ."
    )
    try:
        fig = viz.render_qiskit_circuit(exp.circuit, fold=24)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    except Exception as exc:
        st.warning(f"Could not render circuit: {exc}")


def tab_vqc_encoding(turn: engine.TurnExplanation) -> None:
    exp = turn.vqc
    if exp is None:
        st.warning("No VQC payload available.")
        return
    st.markdown("**Encoded feature vectors** — one row per candidate move.")
    st.caption(
        "Values ≈ π × amplitude of each feature. Each row is the input "
        "fed to the feature layer before the trained θ is applied."
    )
    st.plotly_chart(
        viz.encoding_heatmap(exp.features, exp.scores.moves),
        width="stretch",
    )


def tab_vqc_class_probs(turn: engine.TurnExplanation) -> None:
    exp = turn.vqc
    if exp is None:
        st.warning("No VQC payload available.")
        return
    opt_move = (
        turn.optimal.optimal_move
        if turn.optimal.is_winning_for_player_to_move
        else None
    )
    st.markdown("**Class probabilities per candidate move**")
    st.caption(
        "Stacked bar: the `losing` class probability (higher is better for "
        "the model's pick) plus the `winning` class probability. The chosen "
        "move is the one with the largest `losing` bar."
    )
    st.plotly_chart(
        viz.stacked_class_probs(
            exp.scores.moves,
            exp.class_probs,
            best_index=exp.scores.best_index,
            optimal_move=opt_move,
        ),
        width="stretch",
    )


def tab_qsvm_kernel(turn: engine.TurnExplanation) -> None:
    exp = turn.qsvm
    if exp is None:
        st.warning("No QSVM payload available.")
        return
    st.markdown(
        f"**Kernel similarity row** — encoding `{exp.encoding}`, "
        f"symmetry `{exp.symmetry}`."
    )
    st.caption(
        "Each row is a candidate resulting state; each column is a training "
        "state. Brighter = higher quantum-state overlap. Support vectors are "
        "marked on the x-axis."
    )
    st.plotly_chart(
        viz.kernel_row_heatmap(exp.kernel_row, exp.support_mask, exp.scores.moves),
        width="stretch",
    )


def tab_qsvm_contributions(turn: engine.TurnExplanation) -> None:
    exp = turn.qsvm
    if exp is None:
        st.warning("No QSVM payload available.")
        return
    st.markdown("**Per support vector contribution to the chosen move**")
    st.caption(
        "Decomposes `f(x) = sum_i α_i y_i k(x, sv_i) + b` for the chosen "
        "move's resulting state. Positive bars push toward the 'losing' "
        "class, negative bars push the other way."
    )
    st.plotly_chart(
        viz.sv_contributions_bar(
            exp.kernel_row[exp.scores.best_index],
            exp.support_mask,
            exp.dual_coef,
            exp.intercept,
            support_vectors_raw=exp.support_vectors_raw,
        ),
        width="stretch",
    )


def tab_classical_features(turn: engine.TurnExplanation) -> None:
    exp = turn.classical
    if exp is None:
        st.warning("Classical baseline is not available.")
        return
    st.markdown(
        f"**Feature vector for the chosen move** — feature set `{exp.feature_set}` "
        f"({exp.features.shape[1]} features)."
    )
    st.caption(
        "Hand-engineered inputs to the classical baseline: heap sizes, "
        "Nim-sum, parity, and similar hand-crafted features."
    )
    st.plotly_chart(
        viz.classical_feature_bar(
            exp.features,
            exp.feature_names,
            best_index=exp.scores.best_index,
            moves=exp.scores.moves,
        ),
        width="stretch",
    )


def tab_classical_class_probs(turn: engine.TurnExplanation) -> None:
    exp = turn.classical
    if exp is None:
        st.warning("Classical baseline is not available.")
        return
    if exp.probabilities is None or exp.probabilities.ndim != 2:
        st.info("This classifier does not expose calibrated probabilities.")
        return
    opt_move = (
        turn.optimal.optimal_move
        if turn.optimal.is_winning_for_player_to_move
        else None
    )
    st.markdown("**Class probabilities per candidate move**")
    st.caption(
        "Scikit-learn `predict_proba` for each candidate resulting state. "
        "The chosen move is the one with the largest `losing` probability."
    )
    st.plotly_chart(
        viz.stacked_class_probs(
            exp.scores.moves,
            exp.probabilities,
            best_index=exp.scores.best_index,
            optimal_move=opt_move,
        ),
        width="stretch",
    )


def tab_compare() -> None:
    game = st.session_state.game
    state = game["state"]
    settings = st.session_state.settings
    if is_terminal(state):
        st.info("Game is over — no comparison for terminal state.")
        return

    rows: list[dict] = []
    opt = None

    if settings["mode"] == "You vs Model":
        turn = pt.build_turn_for_variant(state, settings["variant_opp"])
        opt = turn.optimal
        tmap = turn.pipeline_timings_ms or {}
        for name, exp in (
            ("VQC", turn.vqc),
            ("QSVM", turn.qsvm),
            ("Classical", turn.classical),
        ):
            if exp is None:
                continue
            picked = exp.scores.best_move
            agree = (
                bool(tuple(picked) == tuple(opt.optimal_move))
                if opt.is_winning_for_player_to_move
                else True
            )
            ms = tmap.get(name)
            rows.append(
                dict(
                    pipeline=name,
                    picked_move=viz.move_label(picked),
                    resulting_state=viz.state_label(tuple(apply_move(state, picked))),
                    agrees_with_optimal="✓" if agree else "✗",
                    **{"Inference (ms)": f"{ms:.1f}" if ms is not None else "—"},
                )
            )
    else:
        for slot in ("a", "b"):
            pipeline = ps.slot_pipeline(slot)
            if pipeline is None:
                continue
            turn = pt.explain_slot(state, slot)
            exp = pt.get_exp_for(turn, pipeline)
            if exp is None:
                continue
            opt = turn.optimal
            picked = exp.scores.best_move
            agree = (
                bool(tuple(picked) == tuple(opt.optimal_move))
                if opt.is_winning_for_player_to_move
                else True
            )
            tmap = turn.pipeline_timings_ms or {}
            ms = tmap.get(pipeline) if pipeline is not None else None
            rows.append(
                dict(
                    pipeline=ps.slot_label(slot),
                    picked_move=viz.move_label(picked),
                    resulting_state=viz.state_label(tuple(apply_move(state, picked))),
                    agrees_with_optimal="✓" if agree else "✗",
                    **{"Inference (ms)": f"{ms:.1f}" if ms is not None else "—"},
                )
            )

    if opt is None:
        st.warning("No pipelines available.")
        return

    opt_move = opt.optimal_move if opt.is_winning_for_player_to_move else None
    st.markdown(
        f"**Current state:** {viz.state_label(state)}  ·  "
        f"**Nim-sum:** {opt.nim_sum}  ·  "
        f"**Optimal move:** "
        f"{viz.move_label(opt.optimal_move) if opt_move is not None else 'any (losing position)'}"
    )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning("No pipelines available.")

    n_bits = int(np.ceil(np.log2(max(state) + 1))) if any(state) else 1
    st.markdown("**Nim-sum binary breakdown**")
    st.dataframe(
        viz.nim_sum_table(state, n_bits=max(3, n_bits)),
        width="stretch",
        hide_index=True,
    )


def tab_history() -> None:
    game = st.session_state.game
    df = pt.agreement_history_df()
    st.caption(
        "Running agreement with the Nim-sum optimum. Losing positions "
        "(every move is suboptimal) are excluded."
    )
    st.plotly_chart(viz.agreement_history_figure(df), width="stretch")

    if game["move_log"]:
        log_df = pd.DataFrame(game["move_log"])
        log_df["state_before"] = log_df["state_before"].apply(viz.state_label)
        log_df["state_after"] = log_df["state_after"].apply(viz.state_label)
        log_df["move"] = log_df["move"].apply(viz.move_label)
        log_df = log_df[["turn", "label", "state_before", "move", "state_after"]]
        log_df = log_df.rename(columns={"label": "player"})
        st.markdown("**Game log**")
        st.dataframe(log_df, width="stretch", hide_index=True, height=220)
