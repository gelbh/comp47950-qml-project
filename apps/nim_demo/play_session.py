"""Streamlit session state for the Play page: game dict, settings, slot labels."""

from __future__ import annotations

import streamlit as st

from qml_project.nim.game import NimMove, NimState, apply_move as nim_apply_move, is_terminal

from play_config import (
    CLASSICAL_CHOICES,
    Slot,
    OpponentName,
    classical_short,
    resolve_preset_state,
)


def blank_variant() -> dict:
    return {
        "vqc_size": None,
        "qsvm_size": None,
        "classical_name": "Logistic Regression",
        "shots": 512,
    }


def new_game_state() -> dict:
    return {
        "state": (1, 3, 5),
        "initial_state": (1, 3, 5),
        "turn_number": 0,
        "turn_index": 0,
        "players": ("human", "opp"),
        "winner": None,
        "move_log": [],
        "agreement_rows": [],
    }


def init_state() -> None:
    if "game" not in st.session_state:
        st.session_state.game = new_game_state()
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "mode": "You vs Model",
            "opponent": "VQC",
            "model_a": "VQC",
            "model_b": "QSVM",
            "variant_opp": blank_variant(),
            "variant_a": blank_variant(),
            "variant_b": blank_variant(),
            "starting_preset": "(1, 3, 5) — classic",
            "human_first": True,
            "a_first": True,
            "auto_play": True,
            "move_delay": 3.0,
        }


def slot_pipeline(slot: Slot) -> OpponentName | None:
    s = st.session_state.settings
    if slot == "human":
        return None
    if slot == "opp":
        return s["opponent"]
    if slot == "a":
        return s["model_a"]
    if slot == "b":
        return s["model_b"]
    raise ValueError(f"unknown slot: {slot}")


def slot_variant(slot: Slot) -> dict:
    return st.session_state.settings.get(f"variant_{slot}", blank_variant())


def slot_tag(slot: Slot) -> str:
    if st.session_state.settings["mode"] != "Model vs Model":
        return ""
    if slot == "a":
        return "A: "
    if slot == "b":
        return "B: "
    return ""


def slot_label(slot: Slot) -> str:
    if slot == "human":
        return "You"
    pipeline = slot_pipeline(slot)
    variant = slot_variant(slot)
    tag = slot_tag(slot)
    if pipeline == "VQC":
        n = variant.get("vqc_size")
        return f"{tag}VQC (n = {n})" if n is not None else f"{tag}VQC"
    if pipeline == "QSVM":
        n = variant.get("qsvm_size")
        return f"{tag}QSVM (n = {n})" if n is not None else f"{tag}QSVM"
    if pipeline == "Classical":
        name = variant.get("classical_name", CLASSICAL_CHOICES[0])
        if name not in CLASSICAL_CHOICES:
            name = CLASSICAL_CHOICES[0]
        return f"{tag}{classical_short(name)}"
    return f"{tag}{pipeline}"


def current_slot() -> Slot:
    g = st.session_state.game
    return g["players"][g["turn_index"] % 2]


def panel_slot() -> Slot:
    g = st.session_state.game
    settings = st.session_state.settings
    if settings["mode"] == "You vs Model":
        return "opp"
    cur = current_slot()
    if cur != "human":
        return cur
    if g["move_log"]:
        return g["move_log"][-1]["slot"]
    return g["players"][0]


def reset_game() -> None:
    settings = st.session_state.settings
    game = new_game_state()
    start = resolve_preset_state(settings["starting_preset"])
    game["state"] = tuple(int(h) for h in start)
    game["initial_state"] = tuple(int(h) for h in start)
    if settings["mode"] == "You vs Model":
        game["players"] = (
            ("human", "opp") if settings["human_first"] else ("opp", "human")
        )
    else:
        game["players"] = ("a", "b") if settings["a_first"] else ("b", "a")
    game["turn_index"] = 0
    st.session_state.game = game
    st.session_state.pop("_turn_cache", None)


def apply_move(move: NimMove, slot: Slot) -> None:
    from play_turns import track_agreement_for_state

    game = st.session_state.game
    state_before = game["state"]
    turn_no = game["turn_number"] + 1
    track_agreement_for_state(state_before, turn_no)
    state_after = nim_apply_move(state_before, move)
    game["turn_number"] = turn_no
    game["move_log"].append(
        dict(
            turn=turn_no,
            slot=slot,
            label=slot_label(slot),
            state_before=state_before,
            move=move,
            state_after=state_after,
        )
    )
    game["state"] = state_after
    if is_terminal(state_after):
        game["winner"] = slot
    else:
        game["turn_index"] = (game["turn_index"] + 1) % 2
