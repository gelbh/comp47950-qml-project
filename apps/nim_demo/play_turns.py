"""Turn building, agreement logging, and model move selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import engine  # type: ignore[import-not-found]
from loaders import (  # type: ignore[import-not-found]
    load_classical_bundle,
    load_qsvm_payload,
    load_vqc_payload,
)
from qml_project.nim.game import NimMove, NimState, is_terminal

import play_session as ps
from play_config import OpponentName, Slot


def record_agreement(
    turn_no: int,
    label: str,
    picked: NimMove,
    optimal: NimMove,
    state: NimState,
    was_winning_position: bool,
) -> None:
    game = st.session_state.game
    agree = bool(tuple(picked) == tuple(optimal))
    game["agreement_rows"].append(
        dict(
            turn=turn_no,
            pipeline=label,
            picked_move=picked,
            optimal_move=optimal,
            agree=agree,
            state=state,
            was_winning_position=was_winning_position,
        )
    )


def agreement_history_df() -> pd.DataFrame:
    rows = st.session_state.game["agreement_rows"]
    if not rows:
        return pd.DataFrame(columns=["turn", "pipeline", "agree", "running_agreement"])
    df = pd.DataFrame(rows)
    df = df[df["was_winning_position"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["turn", "pipeline", "agree", "running_agreement"])
    df = df.sort_values(["pipeline", "turn"]).reset_index(drop=True)
    df["running_agreement"] = (
        df.groupby("pipeline")["agree"].expanding().mean().reset_index(level=0, drop=True)
    )
    return df


def track_agreement_for_state(state: NimState, turn_no: int) -> None:
    if is_terminal(state):
        return
    settings = st.session_state.settings
    if settings["mode"] == "You vs Model":
        turn = build_turn_for_variant(state, settings["variant_opp"])
        opt = turn.optimal
        was_winning = opt.is_winning_for_player_to_move
        for name, exp in (
            ("VQC", turn.vqc),
            ("QSVM", turn.qsvm),
            ("Classical", turn.classical),
        ):
            if exp is not None:
                record_agreement(
                    turn_no, name, exp.scores.best_move, opt.optimal_move, state, was_winning
                )
    else:
        for slot in ("a", "b"):
            pipeline = ps.slot_pipeline(slot)
            if pipeline is None:
                continue
            turn = build_turn_for_variant(state, ps.slot_variant(slot))
            exp = get_exp_for(turn, pipeline)
            if exp is None:
                continue
            opt = turn.optimal
            record_agreement(
                turn_no,
                ps.slot_label(slot),
                exp.scores.best_move,
                opt.optimal_move,
                state,
                opt.is_winning_for_player_to_move,
            )


def build_turn_for_variant(state: NimState, variant: dict) -> engine.TurnExplanation:
    """Build a 3-pipeline TurnExplanation for one variant config."""
    settings = st.session_state.settings
    shots = int(variant.get("shots", settings.get("shots", 512)))
    key = (
        tuple(int(h) for h in state),
        variant.get("vqc_size"),
        variant.get("qsvm_size"),
        variant.get("classical_name"),
        shots,
    )
    cache: dict = st.session_state.setdefault("_turn_cache", {})
    if key in cache:
        return cache[key]

    vqc_payload = None
    qsvm_payload = None
    classical_bundle = None
    if variant.get("vqc_size") is not None:
        try:
            vqc_payload = load_vqc_payload(int(variant["vqc_size"]))
        except Exception as exc:
            st.warning(f"Could not load VQC payload: {exc}")
    if variant.get("qsvm_size") is not None:
        try:
            qsvm_payload = load_qsvm_payload(int(variant["qsvm_size"]))
        except Exception as exc:
            st.warning(f"Could not load QSVM payload: {exc}")
    try:
        classical_bundle = load_classical_bundle(
            model_name=variant.get("classical_name", "Logistic Regression")
        )
    except Exception as exc:
        st.warning(f"Could not build classical baseline: {exc}")

    turn = engine.build_turn_explanation(
        state,
        vqc_payload=vqc_payload,
        qsvm_payload=qsvm_payload,
        classical_bundle=classical_bundle,
        shots=shots,
        seed=0,
    )
    cache[key] = turn
    return turn


def explain_slot(state: NimState, slot: Slot) -> engine.TurnExplanation:
    return build_turn_for_variant(state, ps.slot_variant(slot))


def get_exp_for(turn: engine.TurnExplanation, pipeline: OpponentName | None):
    if pipeline == "VQC":
        return turn.vqc
    if pipeline == "QSVM":
        return turn.qsvm
    if pipeline == "Classical":
        return turn.classical
    return None


def pick_slot_move(slot: Slot, state: NimState) -> NimMove:
    from qml_project.nim.game import random_policy

    pipeline = ps.slot_pipeline(slot)
    if pipeline is None:
        return random_policy(state, np.random.default_rng(0))
    turn = build_turn_for_variant(state, ps.slot_variant(slot))
    exp = get_exp_for(turn, pipeline)
    if exp is not None:
        return exp.scores.best_move
    return random_policy(state, np.random.default_rng(0))
